from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from neurokit2.ppg.ppg_quality import ppg_quality
from neurokit2.signal.signal_filter import signal_filter

from src.extractors.base import (
    ATTR_ADDITIONAL_SIG,
    ATTR_TOTAL_RECS,
    KEY_PPG_RAW,
    KEY_SPO2_RAW,
    META_FS_HZ,
    ATTR_TOTAL_RECS,
    ATTR_TOTAL_SUBJS,
    META_SUBJECTID,
)
from src.utils import get_group, load_dataset_array

from .base import ATTR_PROCESSOR_TYPE, BaseProcessor, register_processor

PROCESSOR_NAME_STANDARD = "zeynali_processor"

CONF_FS = "fs_hz"
CONF_LOWCUT = "lowcut"
CONF_HIGHCUT = "highcut"
CONF_FILTER_ORDER = "filter_order"
CONF_QUALITY_THRESH = "quality_threshold"

NK_QUALITY_KEY = "PPG_Quality"


@register_processor(PROCESSOR_NAME_STANDARD)
class ZeynaliPPGFilter(BaseProcessor):
    """Apply standard filtering and quality-based rejection to PPG recordings."""

    def __init__(self, config):
        super().__init__(config)
        self.lowcut = float(config.get(CONF_LOWCUT, 0.5))
        self.highcut = float(config.get(CONF_HIGHCUT, 8.0))
        self.filter_order = int(config.get(CONF_FILTER_ORDER, 3))
        self.quality_threshold = float(config.get(CONF_QUALITY_THRESH, 1.0))

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        """Process all groups from raw input into cleaned output.

        Args:
            in_path: Source HDF5 path with raw extractor output.
            out_path: Destination HDF5 path for processed output.

        Returns:
            Path to the processed HDF5 artifact.
        """
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:

            f_out.attrs[ATTR_PROCESSOR_TYPE] = PROCESSOR_NAME_STANDARD

            fs = int(f_in.attrs.get(META_FS_HZ, -1))
            if fs <= 0:
                raise ValueError(f"Invalid or missing sampling rate: {fs}")

            f_out.attrs[CONF_FS] = fs
            f_out.attrs[CONF_LOWCUT] = self.lowcut
            f_out.attrs[CONF_HIGHCUT] = self.highcut
            f_out.attrs[CONF_FILTER_ORDER] = self.filter_order
            f_out.attrs[CONF_QUALITY_THRESH] = self.quality_threshold
            f_out.attrs[ATTR_ADDITIONAL_SIG] = f_in.attrs.get(ATTR_ADDITIONAL_SIG, [])

            keys = list(f_in.keys())
            total_recordings = len(keys)
            pbar = self.pbar(
                keys,
                desc=f"Processing ({PROCESSOR_NAME_STANDARD})",
                total=len(keys),
                unit="grp",
            )

            unique_subjects = set()
            processed_count = 0
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

                    ppg_clean = signal_filter(
                        ppg_raw,
                        sampling_rate=fs,
                        lowcut=self.lowcut,
                        highcut=self.highcut,
                        method="butterworth",
                        order=self.filter_order,
                    )

                    quality = np.ones_like(ppg_clean, dtype=np.float32)
                    bad_ratio = 0.0
                    if self.quality_threshold < 1.0:
                        # Skip quality computation for flat/degenerate signals.
                        if (
                            np.all(ppg_clean == ppg_clean[0])
                            or np.nanstd(ppg_clean) == 0
                        ):
                            removed_count += 1
                            removed_quality += 1
                            update_postfix()
                            continue

                        try:
                            # peaks_info = nk.ppg_findpeaks(ppg_clean, sampling_rate=fs)
                            quality = ppg_quality(
                                ppg_clean,
                                # peaks=peaks_info["PPG_Peaks"],
                                sampling_rate=fs,
                                method="templatematch",
                            )

                            quality = np.asarray(quality, dtype=np.float32)
                            if quality.size == 0 or np.all(np.isnan(quality)):
                                raise ValueError(
                                    "Quality computation returned empty/NaN."
                                )

                            bad_ratio = 1.0 - float(np.nanmean(quality))

                            if bad_ratio > self.quality_threshold:
                                removed_count += 1
                                removed_quality += 1
                                update_postfix()
                                continue
                        except Exception as e:
                            self.log(
                                f"Quality check failed for {key}: {e}. Marking as low quality.",
                                level="debug",
                            )
                            removed_count += 1
                            removed_quality += 1
                            update_postfix()
                            continue

                    self.save_processed_group(
                        f_out=f_out,
                        key=key,
                        ppg_clean=ppg_clean,
                        spo2_raw=spo2_raw,
                        quality=quality,
                        meta_attrs=dict(grp_in.attrs),
                        bad_ratio=bad_ratio,
                    )

                    subj_id = grp_in.attrs.get(META_SUBJECTID, -1)
                    if subj_id != -1:
                        unique_subjects.add(subj_id)

                    processed_count += 1

                    grp_out = f_out[key]
                    self._copy_additional_signals(grp_in, grp_out)

                except Exception as e:
                    self.handle_exception(e, f"{PROCESSOR_NAME_STANDARD} with {key}")
                    removed_count += 1
                    update_postfix()
                    continue

            f_out.attrs[ATTR_TOTAL_RECS] = int(processed_count)
            f_out.attrs[ATTR_TOTAL_SUBJS] = int(len(unique_subjects))
            f_out.attrs["total_removed_recordings"] = int(removed_count)
            f_out.attrs["total_removed_quality"] = int(removed_quality)

            update_postfix()
        return out_path
