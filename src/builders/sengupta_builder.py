from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pywt
from neurokit2.ppg.ppg_findpeaks import ppg_findpeaks
from neurokit2.signal.signal_autocor import signal_autocor
from neurokit2.signal.signal_psd import signal_psd
from scipy import fft, stats
from statsmodels.regression.linear_model import yule_walker

from src.extractors.base import (
    ADDITIONAL_SIG_NAME,
    ATTR_ADDITIONAL_SIG,
    META_FS_HZ,
    META_GLUCOSE,
)
from src.processors.base import KEY_PPG_CLEAN
from src.utils import get_group, load_dataset_array

from .base import BaseTabularBuilder, register_builder

BUILDER_NAME_TABULAR = "sen_gupta_builder"
VERSION = 1


@register_builder(BUILDER_NAME_TABULAR)
class SenGuptaFeatureBuilder(BaseTabularBuilder):
    """Build tabular training features from windowed PPG segments."""

    def __init__(self, config: dict):
        builder_defaults = {
            "window_sec": 3,
            "flat_std_threshold": 0.001,
            "hr_min": 40,
            "hr_max": 210,
        }

        super().__init__(config, defaults=builder_defaults)

        self.window_sec = self.config["window_sec"]
        self.flat_std_threshold = self.config["flat_std_threshold"]
        self.hr_min = self.config["hr_min"]
        self.hr_max = self.config["hr_max"]

    def _get_kteo(self, signal):
        """Compute the Kaiser-Teager Energy Operator (KTEO) sequence.

        Args:
            signal: One-dimensional signal window.

        Returns:
            KTEO-transformed signal of the same effective length.
        """
        x = np.pad(signal, pad_width=1, mode="edge")
        return x[1:-1] ** 2 - x[:-2] * x[2:]

    def _process_single_group(
        self, key, in_path, fs, window_samples: int, detected_signals
    ):
        """Extract feature rows for one input group.

        Args:
            key: HDF5 group key.
            in_path: Source processed HDF5 path.
            window_samples: Number of samples per feature window.
            detected_signals: Additional aligned signal names to aggregate.

        Returns:
            Dictionary with features/metadata for one group, or ``None``.
        """
        try:
            with h5py.File(in_path, "r") as f_in:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    return None

                ppg_signal = load_dataset_array(grp_in, KEY_PPG_CLEAN, verbose=False)
                glucose = grp_in.attrs.get(META_GLUCOSE)
                if ppg_signal is None or glucose is None:
                    return None

                bmi = float(grp_in.attrs.get("BMI", np.nan))
                age = float(grp_in.attrs.get("age", np.nan))
                gender = float(grp_in.attrs.get("gender", np.nan))

                try:
                    all_peaks = np.asarray(
                        ppg_findpeaks(ppg_signal, sampling_rate=fs)["PPG_Peaks"],
                        dtype=np.int64,
                    )
                except Exception:
                    all_peaks = np.array([], dtype=np.int64)

                num_win = len(ppg_signal) // window_samples
                if num_win == 0:
                    return None

                X = ppg_signal[: num_win * window_samples].reshape(
                    num_win, window_samples
                )

                X_mean = np.mean(X, axis=1, keepdims=True)
                X_std = np.std(X, axis=1, keepdims=True)
                # Per-window normalization stabilizes feature scales.
                X_norm = (X - X_mean) / (X_std + 1e-9)

                all_skews = stats.skew(X_norm, axis=1)
                fft_matrix = np.abs(
                    np.fft.fft(X_norm, axis=1)[:, : window_samples // 2]
                )
                fft_skews = stats.skew(fft_matrix, axis=1)
                fft_kurts = stats.kurtosis(fft_matrix, axis=1)

                extra_signals_data = {
                    sig: load_dataset_array(
                        grp_in, f"{sig}{ADDITIONAL_SIG_NAME}", False
                    )
                    for sig in detected_signals
                }

                features_list = []
                for i in range(num_win):
                    if X_std[i, 0] < self.flat_std_threshold:
                        self.log(
                            f"feature builder dismiss key {key} at window {i}",
                            level="debug",
                        )
                        continue

                    start, end = i * window_samples, (i + 1) * window_samples
                    peaks = all_peaks[(all_peaks >= start) & (all_peaks < end)] - start
                    if len(peaks) < 2:
                        continue

                    hr_mean = (60.0 * fs) / np.mean(np.diff(peaks))
                    if not (self.hr_min <= hr_mean <= self.hr_max):
                        continue

                    x_norm = X_norm[i]

                    sign_changes = np.sign(x_norm[:-1]) != np.sign(x_norm[1:])
                    zcr = np.mean(sign_changes)

                    feat_dict = {
                        "ZCR_Count": float(zcr),
                        "ACR": float(signal_autocor(x_norm, lag=1)[0]),
                        "Signal_Skewness": float(all_skews[i]),
                        "FFT_Skewness": float(fft_skews[i]),
                        "FFT_Kurtosis": float(fft_kurts[i]),
                        "BMI": bmi,
                        "age": age,
                        "Gender": gender,
                    }

                    # PSD
                    psd_res = signal_psd(
                        x_norm,
                        sampling_rate=fs,
                        method="welch",
                        show=False,
                        normalize=True,
                    )
                    psd = psd_res["Power"].to_numpy(dtype=float)
                    feat_dict.update(
                        {
                            "PSD_Mean": float(np.mean(psd)),
                            "PSD_Variance": float(np.var(psd)),
                            "PSD_Kurtosis": float(stats.kurtosis(psd)),
                        }
                    )

                    # KTEO
                    kte = self._get_kteo(x_norm)
                    feat_dict.update(
                        {
                            "KTE_Mean": float(np.mean(kte)),
                            "KTE_Variance": float(np.var(kte)),
                            "KTE_Kurtosis": float(stats.kurtosis(kte)),
                            "KTE_Skewness": float(stats.skew(kte)),
                        }
                    )

                    # CWT
                    cwt_coeffs, _ = pywt.cwt(
                        x_norm, scales=np.arange(2, 32), wavelet="mexh"
                    )
                    feat_dict["CWT_Mean"] = float(np.mean(np.abs(cwt_coeffs)))

                    # AR
                    res = yule_walker(x_norm, order=5)
                    ar_params = res[0]
                    feat_dict["AR1"] = float(ar_params[0])
                    feat_dict["AR2"] = float(ar_params[1])
                    feat_dict["AR3"] = float(ar_params[2])
                    feat_dict["AR4"] = float(ar_params[3])
                    feat_dict["AR5"] = float(ar_params[4])

                    feat_dict["SAD"] = float(np.sum(np.abs(np.diff(x_norm))))

                    for sig_name in detected_signals:
                        data = extra_signals_data.get(sig_name)
                        if data is not None and len(data) > 0:
                            feat_dict[f"{sig_name}_mean"] = float(
                                np.mean(data[start:end])
                            )
                        else:
                            feat_dict[f"{sig_name}_mean"] = np.nan

                    features_list.append(feat_dict)

                return {
                    "key": key,
                    "features": features_list,
                    "glucose": glucose,
                    "meta": dict(grp_in.attrs),
                }
        except Exception as e:
            self.handle_exception(e, f"process_single_group with key: {key}")
            return None

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        """Build a tabular HDF5 artifact from processed input groups."""
        with h5py.File(out_path, "w") as f_out:
            self._write_provenance(f_out, BUILDER_NAME_TABULAR, VERSION)
            self._save_metadata(f_out)

            with h5py.File(in_path, "r") as f_in:
                fs_raw = f_in.attrs.get(META_FS_HZ, None)

                if fs_raw is None:
                    raise ValueError(
                        f"Missing required attribute '{META_FS_HZ}' in input file: {in_path}"
                    )
                fs = int(fs_raw)

                keys = list(f_in.keys())
                detected_signals = list(f_in.attrs.get(ATTR_ADDITIONAL_SIG, []))
                self.log(
                    f"Auto-detected additional signals: {detected_signals}",
                    level="debug",
                )

            window_samples = int(fs * self.window_sec)

            self.log("Using sequential processing", level="info")
            total_windows_all_cases = 0
            removed_count = 0

            pbar = self.pbar(
                keys, desc=f"Building Tabular ({BUILDER_NAME_TABULAR})", unit="grp"
            )

            for key in pbar:
                res = self._process_single_group(
                    key, in_path, fs, window_samples, detected_signals
                )

                if res and res["features"]:
                    df = pd.DataFrame(res["features"]).fillna(0)
                    self.save_tabular_group(
                        f_out=f_out,
                        key=res["key"],
                        df_features=df,
                        target_val=res["glucose"],
                        meta_attrs=res["meta"],
                    )
                    total_windows_all_cases += len(df)
                else:
                    removed_count += 1

                pbar.set_postfix(
                    total_windows=total_windows_all_cases, removed=removed_count
                )

            f_out.attrs["total_windows_dataset"] = total_windows_all_cases

            self.log(
                f"{BUILDER_NAME_TABULAR} created {total_windows_all_cases} windows ({self.window_sec}s). Removed {removed_count} recordings.",
                level="info",
            )

        return out_path
