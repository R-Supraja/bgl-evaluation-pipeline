from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .base import Dataset


class DatasetFilter(ABC):
    """Abstract base class (Interface) for all dataset filters."""

    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        """Applies the filter and returns a new, filtered Dataset."""
        pass


class TargetRangeFilter(DatasetFilter):
    """Filters data points based on the y-values (e.g., glucose target ranges)."""

    def __init__(self, target_range: Optional[dict]):
        self.target_range = target_range

    @staticmethod
    def _target_mask(y: np.ndarray, target_range: Optional[dict]) -> np.ndarray:
        """Helper method generating the boolean mask."""
        if not target_range:
            return np.ones(len(y), dtype=bool)

        mask = np.ones(len(y), dtype=bool)

        gt = target_range.get("gt")
        if gt is not None:
            mask &= y > float(gt)

        gte = target_range.get("gte")
        if gte is not None:
            mask &= y >= float(gte)

        lt = target_range.get("lt")
        if lt is not None:
            mask &= y < float(lt)

        lte = target_range.get("lte")
        if lte is not None:
            mask &= y <= float(lte)

        return mask

    def apply(self, dataset: Dataset) -> Dataset:
        if not self.target_range:
            return dataset

        mask = self._target_mask(dataset.y, self.target_range)

        if not np.any(mask):
            raise RuntimeError(
                f"No samples remain after applying target_range={self.target_range}"
            )

        new_meta = dataset.metadata.copy()
        new_meta.update(
            {
                "applied_filter": self.target_range,
                "samples_before_filter": len(dataset.y),
                "samples_after_filter": int(np.sum(mask)),
            }
        )

        return Dataset(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
            metadata=new_meta,
        )


class FeatureExclusionFilter(DatasetFilter):
    """Drop configured feature columns while preserving labels and metadata."""

    def __init__(self, exclude_features: list[str]):
        self.exclude_features = [f.lower() for f in exclude_features]

    def apply(self, dataset: Dataset) -> Dataset:
        feature_names = dataset.metadata.get("feature_names")
        if not self.exclude_features or not feature_names:
            return dataset

        indices_to_keep = [
            i
            for i, name in enumerate(feature_names)
            if name.lower() not in self.exclude_features
        ]

        new_meta = dataset.metadata.copy()
        new_meta["feature_names"] = [feature_names[i] for i in indices_to_keep]
        new_meta["excluded_features"] = self.exclude_features

        return Dataset(
            X=dataset.X[:, indices_to_keep],
            y=dataset.y,
            subject_ids=dataset.subject_ids,
            case_ids=dataset.case_ids,
            metadata=new_meta,
        )
