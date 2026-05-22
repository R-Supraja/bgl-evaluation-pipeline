from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.algorithms import zeynali
from src.extractors.base import META_CASEID, META_FS_HZ, META_GLUCOSE
from src.processors.base import KEY_PPG_CLEAN
from src.utils import get_group, load_dataset_array

from .base import ATTR_NUM_WINDOWS, BaseSequenceBuilder, register_builder

BUILDER_NAME = "zeynali_beat_segmentation"
VERSION = 1


def _py(v):
    return v.item() if hasattr(v, "item") else v


@register_builder(BUILDER_NAME)
class ZeynaliSegmentationBuilder(BaseSequenceBuilder):
    """Build sequence artifacts by chunking signals and extracting beat windows."""

    def __init__(self, config: dict):
        defaults = {
            "chunk_sec": 10.0,
            "window_sec": 1.0,
            "adaptive_height_threshold": False,
            "similarity_threshold": 0.85,
            "distance_threshold": 0.8,
            "height_threshold": 20.0,
            "force": False,
        }
        super().__init__(config, defaults=defaults)

        self.chunk_sec = self.config["chunk_sec"]
        self.adaptive_height_threshold = self.config["adaptive_height_threshold"]

        self.algo_params = {
            "window_duration": self.config["window_sec"],
            "similarity_threshold": self.config["similarity_threshold"],
            "distance_threshold": self.config["distance_threshold"],
            "height_threshold": self.config["height_threshold"],
        }

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        """Convert processed groups into beat-window sequence data."""
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
            self._write_provenance(f_out, builder_type=BUILDER_NAME, version=VERSION)
            self._save_metadata(f_out)

            fs_raw = f_in.attrs.get(META_FS_HZ, None)
            if fs_raw is None:
                raise ValueError(
                    f"Missing required attribute '{META_FS_HZ}' in input file: {in_path}"
                )

            fs = int(fs_raw)
            self.log(f"Using fs={fs} Hz from input file attrs", level="debug")

            keys = list(f_in.keys())
            total_beats = 0
            processed = 0
            skipped_empty = 0
            unique_subjects = set()
            removed_count = 0

            pbar = self.pbar(keys, desc=f"Segmenting beats {BUILDER_NAME}", unit="grp")

            def update_postfix():
                pbar.set_postfix(total_beats=total_beats, removed=removed_count)

            for key in pbar:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    removed_count += 1
                    update_postfix()
                    continue

                ppg_signal = load_dataset_array(grp_in, KEY_PPG_CLEAN, verbose=False)
                glucose = grp_in.attrs.get(META_GLUCOSE, None)
                if ppg_signal is None or glucose is None:
                    removed_count += 1
                    update_postfix()
                    continue

                chunk_len = int(self.chunk_sec * fs)
                if chunk_len <= 0:
                    removed_count += 1
                    update_postfix()
                    continue

                num_chunks = len(ppg_signal) // chunk_len
                if num_chunks == 0:
                    removed_count += 1
                    update_postfix()
                    continue

                all_case_beats = []
                for seg_idx in range(num_chunks):
                    start = seg_idx * chunk_len
                    end = start + chunk_len
                    seg_signal = ppg_signal[start:end]

                    current_params = self.algo_params

                    if self.adaptive_height_threshold:
                        local_mean = np.mean(seg_signal)
                        local_std = np.std(seg_signal)

                        if local_std < 1e-6:
                            continue

                        dynamic_threshold = local_mean + 2 * local_std

                        current_params = self.algo_params.copy()
                        current_params["height_threshold"] = float(dynamic_threshold)
                    try:
                        beat_windows, _ = zeynali.run_segmentation(
                            seg_signal, fs=fs, params=current_params
                        )
                        if beat_windows is not None and len(beat_windows) > 0:
                            all_case_beats.append(beat_windows)
                        # else:
                        #     self.log(
                        #         msg=f"Error no beat_windows at {key} seg {seg_idx}",
                        #         level="debug",
                        #     )
                    except Exception as e:
                        self.handle_exception(
                            e, f"Error while segmentation at {key} seg {seg_idx}"
                        )
                        continue

                if all_case_beats:
                    x_data = np.vstack(all_case_beats).astype(np.float32)

                    if x_data.ndim == 2:
                        x_data = np.expand_dims(x_data, axis=-1)

                    y_data = np.full(
                        (x_data.shape[0],), float(_py(glucose)), dtype=np.float32
                    )

                    meta = {str(k): _py(v) for k, v in grp_in.attrs.items()}
                    meta[ATTR_NUM_WINDOWS] = x_data.shape[0]

                    cid = meta.get(META_CASEID)
                    if cid is not None:
                        unique_subjects.add(cid)

                    self.save_sequence_group(
                        f_out=f_out,
                        key=key,
                        x_data=x_data,
                        y_data=y_data,
                        meta_attrs=meta,
                    )

                    total_beats += x_data.shape[0]
                    processed += 1

                    del x_data, y_data, all_case_beats

                else:
                    skipped_empty += 1
                    removed_count += 1
                update_postfix()

                if (processed + skipped_empty) % 100 == 0:
                    f_out.flush()

            f_out.attrs["total_recordings_with_beat"] = int(processed)
            f_out.attrs["total_subjects"] = int(len(unique_subjects))
            f_out.attrs["groups_skipped_empty"] = int(skipped_empty)
            f_out.attrs["total_beats"] = int(total_beats)

            f_out.flush()

            self.log(
                f"{BUILDER_NAME} created {total_beats} beat segments. Removed {removed_count} recordings.",
                level="info",
            )

        return out_path
