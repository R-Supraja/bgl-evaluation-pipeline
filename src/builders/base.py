from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from src.extractors.base import COMPRESSION_ALGO, META_FS_HZ, META_GLUCOSE
from src.progressbar import ProgressReporter

ATTR_SCHEMA_VERSION = "schema_version"
ATTR_BUILDER_TYPE = "builder_type"
ATTR_ARTIFACT_KIND = "artifact_kind"
STEP_NAME_BUILDER = "builder"

KIND_TABULAR = "tabular"
KIND_SEQUENCE = "sequence"

KEY_X_FEATURES = "x_features"
KEY_X_SEQUENCES = "x_sequences"
KEY_Y_GLUCOSE = "y_glucose"

ATTR_FEATURE_NAMES = "feature_names"
ATTR_NUM_WINDOWS = "num_windows"
ATTR_SHAPE = "shape"
META_WINDOW_SEC = "window_sec"

PREFIX_CONFIG = "config_"

BUILDERS = {}


def register_builder(name):
    def decorator(cls):
        BUILDERS[name] = cls
        return cls

    return decorator


class BaseBuilder(ProgressReporter):
    """Base class for builders creating model-ready artifacts from processed data."""

    def __init__(self, config: dict, defaults: Optional[dict]):
        super().__init__()

        if defaults is not None:
            self.config = self._setup_config(config, defaults)
        else:
            self.config = config

    def _setup_config(self, user_config: dict, defaults: dict) -> dict:
        final_config = defaults.copy()
        unused_params = []

        for key in user_config:
            if key not in defaults:
                unused_params.append(key)
            else:
                final_config[key] = user_config[key]

        if unused_params:
            self.log(
                f"Warning: The following parameters are not used by {self.__class__.__name__}: {unused_params}",
                level="warning",
            )

        return final_config

    def run(
        self, name: str, in_path: Union[str, Path], out_path: Union[str, Path], **kwargs
    ) -> Path:
        """Run builder logic with standardized logging and error handling.

        Args:
            name: Builder name for logs.
            in_path: Input artifact path.
            out_path: Output artifact path.
            **kwargs: Extra runtime arguments accepted by subclasses.

        Returns:
            Path to the produced builder artifact.
        """
        in_path = Path(in_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = self.run_start(
            step=STEP_NAME_BUILDER, name=name, in_path=in_path, out_path=out_path
        )

        try:
            result = self._run_impl(in_path, out_path)
            self.run_end(STEP_NAME_BUILDER, t0=t0)
            return Path(result)

        except Exception as e:
            self.handle_exception(e, context=f"Builder Step [{name}]")
            raise e

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        """Implement builder-specific logic in subclasses.

        Args:
            in_path: Input artifact path.
            out_path: Output artifact path.

        Returns:
            Path to the written artifact.
        """
        raise NotImplementedError

    @staticmethod
    def _write_provenance(f_out: h5py.File, builder_type: str):
        """Write builder provenance attributes to an output HDF5 file.

        Args:
            f_out: Open output HDF5 file handle.
            builder_type: Builder identifier stored in file attributes.
        """
        raise NotImplementedError

    def _save_metadata(self, f_out: h5py.File):
        """Persist config values as output file attributes.

        Args:
            f_out: Open output HDF5 file handle.
        """
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool)):
                f_out.attrs[f"{PREFIX_CONFIG}{k}"] = v


class BaseTabularBuilder(BaseBuilder):
    """Base implementation helpers for tabular builder outputs."""

    @staticmethod
    def _write_provenance(f_out: h5py.File, builder_type: str, version: int):
        """Write provenance fields for tabular artifacts.

        Args:
            f_out: Open output HDF5 file handle.
            builder_type: Builder identifier.
            version: Schema version of the artifact format.
        """
        f_out.attrs[ATTR_SCHEMA_VERSION] = version
        f_out.attrs[ATTR_BUILDER_TYPE] = builder_type
        f_out.attrs[ATTR_ARTIFACT_KIND] = "tabular"

    def save_tabular_group(self, f_out, key, df_features, target_val, meta_attrs):
        """Save one tabular training group with features and glucose targets.

        Args:
            f_out: Open output HDF5 file handle.
            key: Group key in the output file.
            df_features: Feature dataframe (rows are windows).
            target_val: Scalar glucose target value.
            meta_attrs: Metadata attributes copied to the group.
        """
        if df_features is None or df_features.empty:
            return
        if key in f_out:
            return

        grp = f_out.create_group(key)
        x_data = df_features.to_numpy(dtype=np.float32)
        grp.create_dataset(KEY_X_FEATURES, data=x_data, compression=COMPRESSION_ALGO)

        y_data = np.full(len(x_data), float(target_val), dtype=np.float32)
        grp.create_dataset(KEY_Y_GLUCOSE, data=y_data, compression=COMPRESSION_ALGO)

        grp.attrs[ATTR_FEATURE_NAMES] = [n.encode("utf-8") for n in df_features.columns]
        for k, v in dict(meta_attrs).items():
            grp.attrs[k] = v

        grp.attrs[ATTR_NUM_WINDOWS] = len(x_data)
        grp.attrs[META_GLUCOSE] = float(target_val)


class BaseSequenceBuilder(BaseBuilder):
    """Base implementation helpers for sequence builder outputs."""

    @staticmethod
    def _write_provenance(f_out: h5py.File, builder_type: str, version: int):
        """Write provenance fields for sequence artifacts.

        Args:
            f_out: Open output HDF5 file handle.
            builder_type: Builder identifier.
            version: Schema version of the artifact format.
        """
        f_out.attrs[ATTR_SCHEMA_VERSION] = version
        f_out.attrs[ATTR_BUILDER_TYPE] = builder_type
        f_out.attrs[ATTR_ARTIFACT_KIND] = "sequence"

    def save_sequence_group(self, f_out, key, x_data, y_data, meta_attrs):
        """Save one sequence training group with inputs, targets, and metadata.

        Args:
            f_out: Open output HDF5 file handle.
            key: Group key in the output file.
            x_data: Sequence input tensor-like data.
            y_data: Sequence target tensor-like data.
            meta_attrs: Metadata attributes copied to the group.
        """
        if x_data is None or len(x_data) == 0:
            return
        if key in f_out:
            return

        x_arr = np.asarray(x_data, dtype=np.float32)
        y_arr = np.asarray(y_data, dtype=np.float32)

        grp = f_out.create_group(key)
        chunk_samples = min(64, x_arr.shape[0]) if x_arr.shape[0] > 0 else 1

        if x_arr.ndim == 3:
            x_chunks = (chunk_samples, x_arr.shape[1], x_arr.shape[2])
        elif x_arr.ndim == 2:
            x_chunks = (chunk_samples, x_arr.shape[1])
        else:
            x_chunks = True

        grp.create_dataset(
            KEY_X_SEQUENCES, data=x_arr, compression=COMPRESSION_ALGO, chunks=x_chunks
        )

        y_chunks = (chunk_samples,)
        grp.create_dataset(
            KEY_Y_GLUCOSE, data=y_arr, compression=COMPRESSION_ALGO, chunks=y_chunks
        )

        grp.attrs[ATTR_NUM_WINDOWS] = int(x_arr.shape[0])
        grp.attrs[ATTR_SHAPE] = np.array(x_arr.shape, dtype=np.int64)

        if META_FS_HZ in meta_attrs:
            grp.attrs[META_FS_HZ] = int(meta_attrs[META_FS_HZ])
        if META_WINDOW_SEC in meta_attrs:
            grp.attrs[META_WINDOW_SEC] = float(meta_attrs[META_WINDOW_SEC])

        for k, v in dict(meta_attrs).items():
            grp.attrs[k] = v
