from dataclasses import dataclass, field

import numpy as np


@dataclass
class DataSplit:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    case_ids: np.ndarray

    def __len__(self):
        return len(self.y)


@dataclass
class FoldData:
    """Groups train, validation, and test partitions for one fold.

    Attributes:
        train: Training subset.
        val: Validation subset.
        test: Test subset.
        name: Human-readable fold name.
        feature_names: Optional feature names shared across all splits.
    """

    train: DataSplit
    val: DataSplit
    test: DataSplit
    name: str
    feature_names: list[str] = field(default_factory=list)


class Dataset:
    def __init__(self, X, y, subject_ids, case_ids, metadata=None):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.subject_ids = np.asarray(subject_ids)
        self.case_ids = np.asarray(case_ids)
        self.metadata = metadata or {}
        self._validate()

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_subjects(self) -> int:
        return len(np.unique(self.subject_ids))

    @property
    def input_shape(self):
        return self.X.shape[1:]

    def _validate(self):
        """Checks that all sample-aligned arrays share the same length.

        Raises:
            ValueError: If feature, target, or identifier arrays differ in
                length.
        """
        n = len(self.y)
        if not (len(self.X) == len(self.subject_ids) == len(self.case_ids) == n):
            raise ValueError("Dataset dimension mismatch.")
