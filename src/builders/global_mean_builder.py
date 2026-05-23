from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.extractors.base import META_GLUCOSE
from src.utils import get_group

from .base import BaseTabularBuilder, register_builder

BUILDER_NAME = "global_mean_builder"
VERSION = 1


@register_builder(BUILDER_NAME)
class GlobalMeanBuilder(BaseTabularBuilder):
    """Create a minimal tabular artifact for standalone global-mean baseline runs."""

    def __init__(self, config: dict):
        super().__init__(config, defaults={})

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        with h5py.File(in_path, "r") as f_in, h5py.File(out_path, "w") as f_out:
            self._write_provenance(f_out, BUILDER_NAME, VERSION)
            self._save_metadata(f_out)

            keys = list(f_in.keys())
            pbar = self.pbar(keys, desc=f"Building Tabular ({BUILDER_NAME})", unit="grp")

            created = 0
            for key in pbar:
                grp_in = get_group(f_in, key, verbose=False)
                if grp_in is None:
                    continue

                glucose = grp_in.attrs.get(META_GLUCOSE)
                if glucose is None:
                    continue

                # Dummy feature enables tabular loading while keeping this baseline standalone.
                df = pd.DataFrame({"baseline_dummy": [0.0]}, dtype=np.float32)
                self.save_tabular_group(
                    f_out=f_out,
                    key=key,
                    df_features=df,
                    target_val=float(glucose),
                    meta_attrs=dict(grp_in.attrs),
                )
                created += 1

            self.log(
                f"{BUILDER_NAME} created {created} baseline samples (1 per recording).",
                level="info",
            )

        return out_path
