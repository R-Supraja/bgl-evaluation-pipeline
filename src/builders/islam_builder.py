from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from scipy import signal as scipy_sig

from src.extractors.base import META_FS_HZ, META_GLUCOSE
from src.processors.base import KEY_PPG_CLEAN
from src.utils import get_group, load_dataset_array

from .base import BaseTabularBuilder, register_builder

BUILDER_NAME_ISLAM = "islam_builder"
VERSION = 1


@register_builder(BUILDER_NAME_ISLAM)
class IslamTabularBuilder(BaseTabularBuilder):
    """
    Builds tabular training features based on Islam et al. (2021).
    """

    def __init__(self, config: dict):
        defaults = {"window_sec": 50.0, "force": False, "feature_mode": "compact"}
        super().__init__(config, defaults=defaults)
        self.window_sec = self.config["window_sec"]
        self.feature_mode = str(self.config["feature_mode"]).strip().lower()
        if self.feature_mode not in {"compact", "padded"}:
            raise ValueError(
                "Invalid feature_mode for islam_builder. "
                "Use one of: 'compact', 'padded'."
            )

    def _extract_islam_features(self, x_window: np.ndarray, fs: int) -> Optional[dict]:
        """Extracts the features described by Islam et al."""

        # 1. Calculate 1st Derivative using 2-point central difference method
        d1 = np.zeros_like(x_window)
        d1[1:-1] = 0.5 * (x_window[2:] - x_window[:-2])

        # 2. Calculate 2nd Derivative using 3-point central difference method
        d2 = np.zeros_like(x_window)
        d2[1:-1] = x_window[2:] - 2 * x_window[1:-1] + x_window[:-2]

        # 3. Find peaks
        try:
            sys_peaks, _ = scipy_sig.find_peaks(x_window, distance=fs * 0.3)
            dias_peaks, _ = scipy_sig.find_peaks(-x_window, distance=fs * 0.3)
            d1_peaks, _ = scipy_sig.find_peaks(d1)
            d2_peaks, _ = scipy_sig.find_peaks(d2)

        except IndexError as e:
            # self.handle_exception(
            #     e, context=f"{BUILDER_NAME_ISLAM}", show_traceback=False
            # )
            return None
        except Exception as e:
            self.handle_exception(e, context=f"{BUILDER_NAME_ISLAM} PPG peak detection")
            return None

        if len(sys_peaks) < 2 or len(d1_peaks) < 2 or len(d2_peaks) < 2:
            return None

        # 4. Calculate DelT
        delT_list = []
        for sp in sys_peaks:
            subsequent_dias = dias_peaks[dias_peaks > sp]
            if len(subsequent_dias) > 0:
                dp = subsequent_dias[0]
                delT_list.append((dp - sp) / fs)

        if not delT_list:
            return None

        if self.feature_mode == "compact":
            return {
                "Systolic_Peak_Mean": float(np.mean(x_window[sys_peaks])),
                "Systolic_Peak_Std": float(np.std(x_window[sys_peaks])),
                "Diastolic_Peak_Mean": float(
                    np.mean(x_window[dias_peaks]) if len(dias_peaks) > 0 else 0.0
                ),
                "DelT_Mean": float(np.mean(delT_list)),
                "DelT_Std": float(np.std(delT_list)),
                "D1_Peak_Mean": float(np.mean(d1[d1_peaks])),
                "D1_Peak_Std": float(np.std(d1[d1_peaks])),
                "D1_Peak_Count": float(len(d1_peaks)),
                "D2_Peak_Mean": float(np.mean(d2[d2_peaks])),
                "D2_Peak_Std": float(np.std(d2[d2_peaks])),
                "D2_Peak_Count": float(len(d2_peaks)),
            }

        MAX_PEAKS = 150

        def _pad_or_trunc(arr):
            arr = np.asarray(arr)
            if len(arr) >= MAX_PEAKS:
                return arr[:MAX_PEAKS]
            return np.pad(arr, (0, MAX_PEAKS - len(arr)), "constant")

        sys_vals = _pad_or_trunc(x_window[sys_peaks])
        dias_vals = (
            _pad_or_trunc(x_window[dias_peaks])
            if len(dias_peaks) > 0
            else np.zeros(MAX_PEAKS)
        )
        delt_vals = _pad_or_trunc(delT_list)
        d1_vals = _pad_or_trunc(d1[d1_peaks])
        d2_vals = _pad_or_trunc(d2[d2_peaks])

        feat_dict = {}
        for idx in range(MAX_PEAKS):
            feat_dict[f"Sys_Peak_{idx}"] = float(sys_vals[idx])
            feat_dict[f"Dias_Peak_{idx}"] = float(dias_vals[idx])
            feat_dict[f"DelT_{idx}"] = float(delt_vals[idx])
            feat_dict[f"D1_Peak_{idx}"] = float(d1_vals[idx])
            feat_dict[f"D2_Peak_{idx}"] = float(d2_vals[idx])

        return feat_dict

    def _process_single_group(self, key, in_path, fs, window_samples):
        try:
            with h5py.File(in_path, "r") as f_in:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    return None

                ppg_signal = load_dataset_array(grp_in, KEY_PPG_CLEAN, verbose=False)
                glucose = grp_in.attrs.get(META_GLUCOSE)

                if ppg_signal is None or glucose is None:
                    return None

                num_win = len(ppg_signal) // window_samples
                if num_win == 0:
                    return None

                X = ppg_signal[: num_win * window_samples].reshape(
                    num_win, window_samples
                )

                features_list = []
                for i in range(num_win):

                    feat_dict = self._extract_islam_features(X[i], fs)

                    if not feat_dict:
                        continue

                    features_list.append(feat_dict)

                return {
                    "key": key,
                    "features": features_list,
                    "total_possible_chunks": num_win,
                    "glucose": glucose,
                    "meta": dict(grp_in.attrs),
                }
        except Exception as e:
            self.handle_exception(e, f"process_single_group with key: {key}")
            return None

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        with h5py.File(out_path, "w") as f_out:
            self._write_provenance(f_out, BUILDER_NAME_ISLAM, VERSION)
            self._save_metadata(f_out)

            with h5py.File(in_path, "r") as f_in:
                fs_raw = f_in.attrs.get(META_FS_HZ, None)
                if fs_raw is None:
                    raise ValueError(
                        f"Missing required attribute '{META_FS_HZ}' in {in_path}"
                    )
                fs = int(fs_raw)
                keys = list(f_in.keys())

            window_samples = int(fs * self.window_sec)

            self.log(f"Building features for {len(keys)} recordings", level="info")
            total_windows_created = 0
            skipped_windows = 0
            removed_count = 0
            pbar = self.pbar(
                keys, desc=f"Building Tabular ({BUILDER_NAME_ISLAM})", unit="grp"
            )

            for key in pbar:
                res = self._process_single_group(key, in_path, fs, window_samples)

                if res and res["features"]:

                    total_in_grp = res.get(
                        "total_possible_chunks", len(res["features"])
                    )
                    skipped_in_grp = total_in_grp - len(res["features"])
                    skipped_windows += skipped_in_grp

                    df = pd.DataFrame(res["features"]).fillna(0)
                    self.save_tabular_group(
                        f_out=f_out,
                        key=res["key"],
                        df_features=df,
                        target_val=res["glucose"],
                        meta_attrs=res["meta"],
                    )
                    total_windows_created += len(df)
                else:
                    removed_count += 1

                pbar.set_postfix(
                    wins=total_windows_created,
                    skip_w=skipped_windows,
                    rem_rec=removed_count,
                )

            f_out.attrs["total_windows_dataset"] = total_windows_created
            f_out.attrs["total_skipped_windows"] = skipped_windows

            self.log(
                f"{BUILDER_NAME_ISLAM} DONE: Created {total_windows_created} windows. "
                f"Skipped {skipped_windows} bad ({self.window_sec}s) windows. "
                f"Removed {removed_count} empty recordings.",
                level="info",
            )

        return out_path
