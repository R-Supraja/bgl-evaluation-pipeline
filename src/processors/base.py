from pathlib import Path
from typing import Union

import h5py
import numpy as np

from src.extractors.base import (
    ADDITIONAL_SIG_NAME,
    COMPRESSION_ALGO,
    KEY_PPG_RAW,
    KEY_SPO2_RAW,
)
from src.progressbar import ProgressReporter
from src.utils import load_dataset_array

KEY_PPG_CLEAN = "ppg_signal_cleaned"
KEY_QUALITY = "ppg_quality"

ATTR_SCHEMA_VERSION = "schema_version"
ATTR_PROCESSOR_TYPE = "processor_type"
ATTR_BAD_QUALITY_RATIO = "bad_quality_ratio"

COMPRESSION_ALGO = "gzip"
STEP_NAME_PROCESSOR = "processor"

PROCESSORS = {}


def register_processor(name):
    def decorator(cls):
        PROCESSORS[name] = cls
        return cls

    return decorator


class BaseProcessor(ProgressReporter):
    """Base class for processors transforming raw signals into cleaned outputs."""

    def __init__(self, config):
        self.config = config
        super().__init__()

    def run(
        self, name: str, in_path: Union[str, Path], out_path: Union[str, Path], **kwargs
    ) -> Path:
        """Run the processor implementation with standardized logging.

        Args:
            name: processor name for logs.
            in_path: Input artifact path.
            out_path: Output artifact path.
            **kwargs: Extra runtime arguments accepted by subclasses.

        Returns:
            Path to the produced processed artifact.
        """
        in_path = Path(in_path)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = self.run_start(
            step=STEP_NAME_PROCESSOR, name=name, in_path=in_path, out_path=out_path
        )

        try:
            result_path = self._run_impl(in_path, out_path)
            self.run_end(STEP_NAME_PROCESSOR, t0=t0)
            return result_path

        except Exception as e:
            self.handle_exception(e, context=f"Processor Step [{name}]")
            raise e

    def _run_impl(self, in_path: Path, out_path: Path) -> Path:
        """Implement processor-specific logic in subclasses.

        Args:
            in_path: Input artifact path.
            out_path: Output artifact path.

        Returns:
            Path to the written processed artifact.
        """
        raise NotImplementedError

    def _copy_additional_signals(self, grp_in, grp_out):
        """Copy non-core aligned signals from input to output group.

        Args:
            grp_in: Input HDF5 group.
            grp_out: Output HDF5 group.
        """
        for key in grp_in.keys():
            if key in [
                KEY_PPG_RAW,
                KEY_SPO2_RAW,
                KEY_PPG_CLEAN,
                KEY_QUALITY,
            ]:
                continue

            if key.endswith(ADDITIONAL_SIG_NAME):
                try:
                    signal_data = load_dataset_array(grp_in, key, verbose=False)
                    if signal_data is not None and len(signal_data) > 0:
                        grp_out.create_dataset(
                            key,
                            data=signal_data.astype(np.float32),
                            compression=COMPRESSION_ALGO,
                        )
                except Exception as e:
                    self.log(f"Could not copy signal {key}: {e}", level="debug")

    def _normalize_0_1(self, signal: np.ndarray) -> np.ndarray:
        """Normalizes the signal strictly to a range of [0, 1]."""
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        if sig_max - sig_min == 0:
            return signal
        return (signal - sig_min) / (sig_max - sig_min)

    @staticmethod
    def save_processed_group(
        f_out: h5py.File, key, ppg_clean, spo2_raw, quality, meta_attrs, bad_ratio
    ):
        """Persist one processed sample group with cleaned signals and metadata.

        Args:
            f_out: Open output HDF5 file handle.
            key: Target group key.
            ppg_clean: Cleaned PPG signal array.
            spo2_raw: Optional aligned SpO2 signal array.
            quality: Optional per-sample quality array.
            meta_attrs: Metadata attributes to store on the group.
            bad_ratio: Fraction of low-quality samples in the segment.
        """
        if key in f_out:
            return
        if ppg_clean is None or len(ppg_clean) == 0:
            return

        grp_out = f_out.create_group(key)

        ppg_clean = np.asarray(ppg_clean, dtype=np.float32)
        grp_out.create_dataset(
            KEY_PPG_CLEAN, data=ppg_clean, compression=COMPRESSION_ALGO
        )

        if spo2_raw is not None:
            spo2_raw = np.asarray(spo2_raw, dtype=np.float32)
            grp_out.create_dataset(
                KEY_SPO2_RAW, data=spo2_raw, compression=COMPRESSION_ALGO
            )

        if quality is not None:
            quality = np.asarray(quality, dtype=np.float32)
            grp_out.create_dataset(
                KEY_QUALITY, data=quality, compression=COMPRESSION_ALGO
            )

        for k, v in meta_attrs.items():
            grp_out.attrs[k] = v

        grp_out.attrs[ATTR_BAD_QUALITY_RATIO] = float(bad_ratio)
