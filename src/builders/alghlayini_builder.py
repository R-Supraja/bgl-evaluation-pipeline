from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.builders.base import BaseSequenceBuilder, register_builder
from src.extractors.base import META_FS_HZ, META_GLUCOSE
from src.processors.base import KEY_PPG_CLEAN
from src.utils import get_group, load_dataset_array

from .base import ATTR_NUM_WINDOWS

BUILDER_NAME = "alghlayini_builder"
VERSION = 1


@register_builder(BUILDER_NAME)
class AlghlayiniWindowBuilder(BaseSequenceBuilder):
    """
    Builds continuous sequence segments according to Alghlayini et al. (2023).
    """

    def __init__(self, config: dict):
        defaults = {"window_sec": 60.0, "force": False}
        super().__init__(config, defaults=defaults)
        self.window_sec = self.config["window_sec"]

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
            fs = f_in.attrs.get(META_FS_HZ)
            if not fs or fs <= 0:
                raise ValueError(
                    f"Missing required attribute '{META_FS_HZ}' in input file: {in_path}"
                )

            self._write_provenance(f_out, builder_type=BUILDER_NAME, version=VERSION)
            self._save_metadata(f_out)

            f_out.attrs[META_FS_HZ] = fs
            f_out.attrs["window_sec"] = self.window_sec

            window_samples = int(fs * self.window_sec)

            keys = list(f_in.keys())
            pbar = self.pbar(keys, desc=f"Windowing ({BUILDER_NAME})", unit="grp")

            removed_count = 0
            total_sequences = 0

            for key in pbar:
                grp = get_group(f_in, key, verbose=False)
                if grp is None:
                    removed_count += 1
                    continue

                ppg = load_dataset_array(grp, KEY_PPG_CLEAN, verbose=False)
                glucose = grp.attrs.get(META_GLUCOSE, None)

                if ppg is None or glucose is None:
                    removed_count += 1
                    continue

                if len(ppg) < window_samples:
                    removed_count += 1
                    continue

                num_windows = len(ppg) // window_samples
                if num_windows <= 0:
                    removed_count += 1
                    continue

                grp_x = []

                for i in range(num_windows):
                    start = i * window_samples
                    end = start + window_samples
                    window = ppg[start:end]

                    if np.isnan(window).any():
                        continue

                    grp_x.append(window)

                if not grp_x:
                    removed_count += 1
                    continue

                # Convert to numpy array and shape properly for SequenceBuilder (N, L, 1)
                x_data = np.vstack(grp_x).astype(np.float32)
                if x_data.ndim == 2:
                    x_data = np.expand_dims(x_data, axis=-1)

                # Create array of targets matching the number of windows
                y_data = np.full((x_data.shape[0],), float(glucose), dtype=np.float32)

                meta = {
                    str(k): (v.item() if hasattr(v, "item") else v)
                    for k, v in grp.attrs.items()
                }
                meta[ATTR_NUM_WINDOWS] = x_data.shape[0]

                self.save_sequence_group(
                    f_out=f_out,
                    key=key,
                    x_data=x_data,
                    y_data=y_data,
                    meta_attrs=meta,
                )

                total_sequences += x_data.shape[0]
                pbar.set_postfix(total_seqs=total_sequences, removed=removed_count)

            if total_sequences == 0:
                raise RuntimeError(
                    f"No valid sequences found for {BUILDER_NAME} in {in_path.name}"
                )

            f_out.attrs["total_sequences"] = total_sequences

            self.log(
                f"{BUILDER_NAME} created {total_sequences} segments ({self.window_sec}s). Removed {removed_count} recordings.",
                level="info",
            )

        return out_path
