from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from src.builders.base import (
    ATTR_FEATURE_NAMES,
    KEY_X_FEATURES,
    KEY_X_SEQUENCES,
    KEY_Y_GLUCOSE,
)
from src.extractors.base import META_CASEID, META_GLUCOSE, META_SUBJECTID
from src.utils import get_dataset, get_group, load_dataset_array

from .base import Dataset


class DatasetLoader(ABC):
    """Interface for loading model-ready datasets from persisted artifacts."""

    @abstractmethod
    def load(self, path: Path) -> Dataset:
        pass


class SequenceHDF5Loader(DatasetLoader):
    """Load sequence artifacts (`x_sequences`) into a unified Dataset object."""

    def load(self, path: Path) -> Dataset:
        path = Path(path)

        with h5py.File(path, "r") as f:
            total_n = 0
            sample_shape = None

            for key in f.keys():
                grp = get_group(f, key, verbose=False)
                if grp is None:
                    continue

                ds = get_dataset(grp, KEY_X_SEQUENCES, verbose=False)
                if ds is None:
                    continue

                total_n += ds.shape[0]
                if sample_shape is None:
                    sample_shape = ds.shape[1:]

            if total_n == 0 or sample_shape is None:
                raise RuntimeError(f"No valid {KEY_X_SEQUENCES} found in {path}")

            if len(sample_shape) == 1:
                alloc_shape = (total_n, sample_shape[0], 1)
            else:
                alloc_shape = (total_n, *sample_shape)

            X = np.empty(alloc_shape, dtype=np.float32)
            y = np.empty(total_n, dtype=np.float32)
            subject_ids = np.empty(total_n, dtype=np.int32)
            case_ids = np.empty(total_n, dtype=np.int32)

            current_idx = 0
            for key in f.keys():
                grp = get_group(f, key, verbose=False)
                if grp is None:
                    continue

                x_data = load_dataset_array(grp, KEY_X_SEQUENCES, verbose=False)
                if x_data is None:
                    continue

                x_data = np.asarray(x_data, dtype=np.float32)
                if x_data.ndim == 2:
                    x_data = x_data[..., np.newaxis]

                n = int(x_data.shape[0])
                glucose = float(grp.attrs.get(META_GLUCOSE, 0.0))

                caseid_raw = grp.attrs.get(META_CASEID, -1)
                case_id = (
                    int(caseid_raw.item())
                    if hasattr(caseid_raw, "item")
                    else int(caseid_raw)
                )

                subjid_raw = grp.attrs.get(META_SUBJECTID, -1)
                subject = (
                    int(subjid_raw.item())
                    if hasattr(subjid_raw, "item")
                    else int(subjid_raw)
                )

                slc = slice(current_idx, current_idx + n)

                X[slc] = x_data
                y[slc] = glucose
                subject_ids[slc] = subject
                case_ids[slc] = case_id

                current_idx += n

        return Dataset(
            X=X,
            y=y,
            subject_ids=subject_ids,
            case_ids=case_ids,
            metadata={"source": path.name, "type": "sequence"},
        )


class TabularHDF5Loader(DatasetLoader):
    """Load tabular artifacts (`x_features`, `y_glucose`) into Dataset."""

    def load(self, path: Path) -> Dataset:
        path = Path(path)
        all_X, all_y, all_subjects, all_cases = [], [], [], []
        available_features = []
        indices_to_keep = None
        feature_names = []

        with h5py.File(path, "r") as f:
            for key in list(f.keys()):
                grp = get_group(f, key, verbose=False)
                if grp is None:
                    continue
                raw_names = grp.attrs.get(ATTR_FEATURE_NAMES, None)
                if raw_names is not None:
                    feature_names = [
                        (
                            n.decode("utf-8")
                            if isinstance(n, (bytes, bytearray))
                            else str(n)
                        )
                        for n in raw_names
                    ]
                    break

            # 2. Daten laden
            for key in f.keys():
                grp = get_group(f, key, verbose=False)
                if grp is None:
                    continue

                x_data = load_dataset_array(grp, KEY_X_FEATURES, verbose=False)
                y_data = load_dataset_array(grp, KEY_Y_GLUCOSE, verbose=False)
                if x_data is None or y_data is None:
                    continue

                if indices_to_keep is not None:
                    x_data = x_data[:, indices_to_keep]

                caseid_raw = grp.attrs.get(META_CASEID, -1)
                case_id = (
                    int(caseid_raw.item())
                    if hasattr(caseid_raw, "item")
                    else int(caseid_raw)
                )

                subjid_raw = grp.attrs.get(META_SUBJECTID, -1)
                subjid = (
                    int(subjid_raw.item())
                    if hasattr(subjid_raw, "item")
                    else int(subjid_raw)
                )

                n = int(x_data.shape[0])
                all_X.append(x_data)
                all_y.append(y_data)
                all_cases.extend([case_id] * n)
                all_subjects.extend([subjid] * n)

        if not all_X:
            raise RuntimeError(f"No valid x_features found in {path}")

        return Dataset(
            X=np.concatenate(all_X, axis=0),
            y=np.concatenate(all_y, axis=0),
            subject_ids=np.asarray(all_subjects),
            case_ids=np.asarray(all_cases),
            metadata={
                "source": path.name,
                "type": "tabular",
                "feature_names": feature_names,
            },
        )
