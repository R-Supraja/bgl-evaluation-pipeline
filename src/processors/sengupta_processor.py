from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from neurokit2.signal.signal_filter import signal_filter

from src.algorithms.fsw import _fsw_algorithm
from src.extractors.base import (
    ATTR_ADDITIONAL_SIG,
    ATTR_TOTAL_RECS,
    KEY_PPG_RAW,
    KEY_SPO2_RAW,
    META_FS_HZ,
)
from src.utils import get_group, load_dataset_array

from .base import ATTR_PROCESSOR_TYPE, BaseProcessor, register_processor

HIGH_CUT = 8.0
FILTER_ORDER = 6

PROCESSOR_NAME_STANDARD = "sen_gupta_processor"


@register_processor(PROCESSOR_NAME_STANDARD)
class SenGuptaPPGFilter(BaseProcessor):
    """Apply Sen Gupta et al. (2021) specific filtering to PPG recordings."""

    def __init__(self, config):
        super().__init__(config)

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:

            f_out.attrs[ATTR_PROCESSOR_TYPE] = PROCESSOR_NAME_STANDARD

            fs = int(f_in.attrs.get(META_FS_HZ, -1))
            if fs <= 0:
                raise ValueError(f"Invalid or missing sampling rate: {fs}")

            f_out.attrs[META_FS_HZ] = fs
            f_out.attrs[ATTR_ADDITIONAL_SIG] = f_in.attrs.get(ATTR_ADDITIONAL_SIG, [])

            keys = list(f_in.keys())
            total_recordings = len(keys)
            pbar = self.pbar(
                keys,
                desc=f"Processing ({PROCESSOR_NAME_STANDARD})",
                total=len(keys),
                unit="grp",
            )

            removed_count = 0
            removed_quality = 0

            def update_postfix():
                pbar.set_postfix(total=total_recordings, removed=removed_count)

            for key in pbar:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    continue

                try:

                    ppg_raw = load_dataset_array(grp_in, KEY_PPG_RAW, verbose=False)
                    spo2_raw = load_dataset_array(grp_in, KEY_SPO2_RAW, verbose=False)

                    if ppg_raw is None or len(ppg_raw) == 0:
                        removed_count += 1
                        update_postfix()
                        continue

                    # 1. Baseline Wandering Removal
                    ppg_detrended = _fsw_algorithm(ppg_raw, fs)

                    # 2. Butterworth filter
                    ppg_clean = signal_filter(
                        ppg_detrended,
                        sampling_rate=fs,
                        lowcut=None,
                        highcut=HIGH_CUT,
                        method="butterworth",
                        order=FILTER_ORDER,
                    )

                    quality_dummy = np.ones_like(ppg_clean, dtype=np.float32)

                    self.save_processed_group(
                        f_out=f_out,
                        key=key,
                        ppg_clean=ppg_clean,
                        spo2_raw=spo2_raw,
                        quality=quality_dummy,
                        meta_attrs=dict(grp_in.attrs),
                        bad_ratio=0.0,
                    )

                    grp_out = f_out[key]
                    self._copy_additional_signals(grp_in, grp_out)

                except Exception as e:
                    self.handle_exception(e, f"{PROCESSOR_NAME_STANDARD} with {key}")
                    removed_count += 1
                    update_postfix()
                    continue

            f_out.attrs[ATTR_TOTAL_RECS] = int(total_recordings - removed_count)
            f_out.attrs["total_removed_recordings"] = int(removed_count)
            f_out.attrs["total_removed_quality"] = int(removed_quality)

            update_postfix()
        return out_path
