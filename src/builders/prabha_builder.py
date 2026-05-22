from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd
from scipy import fft
from scipy import signal as scipy_sig

from src.extractors.base import META_AGE, META_GLUCOSE, META_HEIGHT, META_WEIGHT
from src.processors.base import KEY_PPG_CLEAN
from src.utils import get_group, load_dataset_array

from .base import BaseTabularBuilder, register_builder

VERSION = 1
BUILDER_NAME_MFCC_TABULAR = "prabha_tabular_features"


def _hz_to_mel(freq_hz: Union[float, np.ndarray]) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (freq_hz / 700.0))


def _mel_to_hz(mel: Union[float, np.ndarray]) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    *,
    fs: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    fmax = float(fmax if fmax is not None else fs / 2.0)

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor(((n_fft + 1) * hz_points) / fs).astype(int)
    bins = np.clip(bins, 0, (n_fft // 2) + 1)

    fbanks = np.zeros((n_mels, (n_fft // 2) + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        if center <= left:
            center = min(left + 1, fbanks.shape[1] - 1)
        if right <= center:
            right = min(center + 1, fbanks.shape[1])

        for k in range(left, center):
            fbanks[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            fbanks[m - 1, k] = (right - k) / max(right - center, 1)

    return fbanks


def _extract_mfcc_features(
    signal_window: np.ndarray,
    *,
    fs: int,
    frame_len_samples: int,
    frame_hop_samples: int,
    n_mels: int,
    n_mfcc: int,
) -> dict[str, float]:
    """
    For each 5-second segment, this computes:
    - 13 MFCC mean coefficients across frames
    - 91 upper-triangular covariance entries (excluding diagonal)
    Total MFCC-derived features: 104.
    """
    x = np.asarray(signal_window, dtype=np.float32)
    if len(x) < frame_len_samples or frame_len_samples <= 0 or frame_hop_samples <= 0:
        return {}

    x_filtered = scipy_sig.lfilter([1.0, -0.97], [1.0], x)
    x = np.asarray(x_filtered, dtype=np.float32)

    # Overlapping framing (paper-style MFCC matrix with shape 13 x n).
    starts = np.arange(0, len(x) - frame_len_samples + 1, frame_hop_samples, dtype=int)
    if len(starts) == 0:
        return {}

    frames = np.stack([x[s : s + frame_len_samples] for s in starts], axis=0)
    frames = frames * np.hamming(frame_len_samples)

    n_fft = max(256, 1 << int(np.ceil(np.log2(frame_len_samples))))
    power_spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    power_spec = power_spec / max(n_fft, 1)

    mel_fb = _build_mel_filterbank(fs=fs, n_fft=n_fft, n_mels=n_mels)
    mel_energies = np.maximum(power_spec @ mel_fb.T, 1e-9)
    log_mel = np.log(mel_energies)
    dct_result = np.asarray(fft.dct(log_mel, type=2, axis=1, norm="ortho"))
    mfcc = dct_result[:, :n_mfcc]

    feat_dict: dict[str, float] = {}

    # 13 mean MFCC features across frames.
    mfcc_means = np.mean(mfcc, axis=0)
    for coeff_idx in range(n_mfcc):
        feat_dict[f"mfcc_mean_c{coeff_idx + 1:02d}"] = float(mfcc_means[coeff_idx])

    # 91 upper-diagonal covariance features for 13 coefficients.
    # Covariance is computed across frames for each coefficient pair.
    cov = np.cov(mfcc, rowvar=False)
    upper_i, upper_j = np.triu_indices(n_mfcc, k=1)
    for i, j in zip(upper_i, upper_j):
        feat_dict[f"mfcc_cov_u_c{i + 1:02d}_c{j + 1:02d}"] = float(cov[i, j])

    return feat_dict


@register_builder(BUILDER_NAME_MFCC_TABULAR)
class MFCCTabularFeatureBuilder(BaseTabularBuilder):
    def __init__(self, config: dict):
        defaults = {
            "fs": 100,
            "window_sec": 5.0,
            "flat_std_threshold": 0.001,
            "frame_len_sec": 1.0,
            "frame_hop_sec": 0.4,
            "n_mels": 26,
            "n_mfcc": 13,
            "force": False,
        }
        super().__init__(config, defaults=defaults)

        self.fs = self.config["fs"]
        self.window_sec = self.config["window_sec"]
        self.flat_std_threshold = self.config["flat_std_threshold"]
        self.frame_len_sec = self.config["frame_len_sec"]
        self.frame_hop_sec = self.config["frame_hop_sec"]
        self.n_mels = self.config["n_mels"]
        self.n_mfcc = self.config["n_mfcc"]

    def _process_single_group(self, key, in_path, window_samples):
        try:
            with h5py.File(in_path, "r") as f_in:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    return None

                ppg_signal = load_dataset_array(grp_in, KEY_PPG_CLEAN, verbose=False)
                glucose = grp_in.attrs.get(META_GLUCOSE)
                if ppg_signal is None or glucose is None:
                    return None

                age = float(grp_in.attrs.get(META_AGE, np.nan))
                height = float(grp_in.attrs.get(META_HEIGHT, np.nan))
                weight = float(grp_in.attrs.get(META_WEIGHT, np.nan))

                num_win = len(ppg_signal) // window_samples
                if num_win == 0:
                    return None

                X = ppg_signal[: num_win * window_samples].reshape(
                    num_win, window_samples
                )
                X_mean = np.mean(X, axis=1, keepdims=True)
                X_std = np.std(X, axis=1, keepdims=True)
                X_norm = (X - X_mean) / (X_std + 1e-9)

                features_list = []
                for i in range(num_win):
                    if X_std[i, 0] < self.flat_std_threshold:
                        self.log(
                            f"mfcc builder dismiss key {key} at window {i}",
                            level="debug",
                        )
                        continue

                    feat_dict = _extract_mfcc_features(
                        X_norm[i],
                        fs=self.fs,
                        frame_len_samples=max(1, int(round(self.frame_len_sec * self.fs))),
                        frame_hop_samples=max(1, int(round(self.frame_hop_sec * self.fs))),
                        n_mels=self.n_mels,
                        n_mfcc=self.n_mfcc,
                    )
                    if not feat_dict:
                        continue

                    feat_dict.update(
                        {
                            "age": age,
                            "height": height,
                            "weight": weight,
                        }
                    )
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
        window_samples = int(self.fs * self.window_sec)

        with h5py.File(out_path, "w") as f_out:
            self._write_provenance(f_out, BUILDER_NAME_MFCC_TABULAR, VERSION)
            self._save_metadata(f_out)

            with h5py.File(in_path, "r") as f_in:
                keys = list(f_in.keys())

            self.log("Using sequential processing", level="info")
            total_windows_all_cases = 0
            removed_count = 0

            pbar = self.pbar(keys, desc="Building MFCC Tabular", unit="grp")

            for key in pbar:
                res = self._process_single_group(key, in_path, window_samples)

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
                f"{BUILDER_NAME_MFCC_TABULAR} created {total_windows_all_cases} windows ({self.window_sec}s). Removed {removed_count} recordings.",
                level="info",
            )

        return out_path
