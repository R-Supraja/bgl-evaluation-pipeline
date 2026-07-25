from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from src.progressbar import ProgressReporter

from .base import Dataset, DataSplit, FoldData


class DataSplitter(ABC):
    """Base interface for converting a Dataset into train/val/test fold maps."""

    @abstractmethod
    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        pass

    def _make_split(self, dataset: Dataset, subjects: np.ndarray) -> DataSplit:
        mask = np.isin(dataset.subject_ids, subjects)
        return DataSplit(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
        )


class CrossValidationSplitter(DataSplitter):
    """Subject-wise K-fold CV splitter with per-fold validation carve-out."""

    def __init__(self, n_splits: int = 5, val_ratio: float = 0.15):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")

        rng = np.random.default_rng(seed)
        subjects = np.unique(dataset.subject_ids)
        # sorted unique subjectids
        rng.shuffle(subjects)

        chunks = np.array_split(subjects, self.n_splits)
        folds = {}
        features = dataset.metadata.get("feature_names", [])

        for i in range(self.n_splits):
            test_subjects = chunks[i]
            train_val_subjects = np.concatenate(
                [c for j, c in enumerate(chunks) if j != i]
            )

            if self.val_ratio > 0.0:
                n_val = max(1, int(len(train_val_subjects) * self.val_ratio))
            else:
                n_val = 0

            val_subjects = train_val_subjects[:n_val]
            train_subjects = train_val_subjects[n_val:]

            folds[f"fold_{i}"] = FoldData(
                train=self._make_split(dataset, train_subjects),
                val=self._make_split(dataset, val_subjects),
                test=self._make_split(dataset, test_subjects),
                name=f"fold_{i}",
                feature_names=features,
            )
        return folds


class SingleSplitter(DataSplitter):
    """Single subject-wise train/val/test split."""

    def __init__(self, test_ratio: float = 0.2, val_ratio: float = 0.15):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")

        rng = np.random.default_rng(seed)
        subjects = np.unique(dataset.subject_ids)
        rng.shuffle(subjects)

        if self.test_ratio > 0.0:
            n_test = max(1, int(len(subjects) * self.test_ratio))
        else:
            n_test = 0

        test_subjects = subjects[:n_test]
        train_val = subjects[n_test:]

        if self.val_ratio > 0.0:
            n_val = max(1, int(len(train_val) * self.val_ratio))
        else:
            n_val = 0

        val_subjects = train_val[:n_val]
        train_subjects = train_val[n_val:]

        features = dataset.metadata.get("feature_names", [])
        return {
            "single_run": FoldData(
                train=self._make_split(dataset, train_subjects),
                val=self._make_split(dataset, val_subjects),
                test=self._make_split(dataset, test_subjects),
                name="single_run",
                feature_names=features,
            )
        }


class CaseWiseCrossValidationSplitter(DataSplitter):
    """Case-wise K-fold CV splitter to avoid case leakage across folds."""
    def __init__(self, n_splits: int = 5, val_ratio: float = 0.15):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def _make_split(self, dataset: Dataset, cases: np.ndarray) -> DataSplit:
        mask = np.isin(dataset.case_ids, cases)
        return DataSplit(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
        )

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")

        rng = np.random.default_rng(seed)

        cases = np.unique(dataset.case_ids)
        rng.shuffle(cases)

        chunks = np.array_split(cases, self.n_splits)
        folds = {}
        features = dataset.metadata.get("feature_names", [])

        for i in range(self.n_splits):
            test_cases = chunks[i]
            train_val_cases = np.concatenate(
                [c for j, c in enumerate(chunks) if j != i]
            )

            if self.val_ratio > 0.0:
                n_val = max(1, int(len(train_val_cases) * self.val_ratio))
            else:
                n_val = 0

            val_cases = train_val_cases[:n_val]
            train_cases = train_val_cases[n_val:]

            folds[f"fold_{i}"] = FoldData(
                train=self._make_split(dataset, train_cases),
                val=self._make_split(dataset, val_cases),
                test=self._make_split(dataset, test_cases),
                name=f"fold_{i}",
                feature_names=features,
            )
        return folds


class CaseWiseSingleSplitter(DataSplitter):
    """Single case-wise train/val/test split."""
    def __init__(self, test_ratio: float = 0.2, val_ratio: float = 0.15):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def _make_split(self, dataset: Dataset, cases: np.ndarray) -> DataSplit:
        mask = np.isin(dataset.case_ids, cases)
        return DataSplit(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
        )

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")

        rng = np.random.default_rng(seed)

        cases = np.unique(dataset.case_ids)
        rng.shuffle(cases)

        if self.test_ratio > 0.0:
            n_test = max(1, int(len(cases) * self.test_ratio))
        else:
            n_test = 0

        test_cases = cases[:n_test]
        train_val = cases[n_test:]

        if self.val_ratio > 0.0:
            n_val = max(1, int(len(train_val) * self.val_ratio))
        else:
            n_val = 0

        val_cases = train_val[:n_val]
        train_cases = train_val[n_val:]

        features = dataset.metadata.get("feature_names", [])
        return {
            "single_run": FoldData(
                train=self._make_split(dataset, train_cases),
                val=self._make_split(dataset, val_cases),
                test=self._make_split(dataset, test_cases),
                name="single_run",
                feature_names=features,
            )
        }


class RandomSegmentSplitter(DataSplitter):
    """
    Splits the dataset purely by individual segments (rows).
    Ignores whether segments belong to the same case or subject.
    """

    def __init__(self, test_ratio: float = 0.2, val_ratio: float = 0.15):
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def _make_split_by_indices(
        self, dataset: Dataset, indices: np.ndarray
    ) -> DataSplit:
        """Helper to create a DataSplit based on row indices."""
        return DataSplit(
            X=dataset.X[indices],
            y=dataset.y[indices],
            subject_ids=dataset.subject_ids[indices],
            case_ids=dataset.case_ids[indices],
        )

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")

        rng = np.random.default_rng(seed)

        all_indices = np.arange(dataset.n_samples)
        rng.shuffle(all_indices)

        n_total = len(all_indices)
        n_test = int(n_total * self.test_ratio)
        n_val = int((n_total - n_test) * self.val_ratio)

        test_idx = all_indices[:n_test]
        val_idx = all_indices[n_test : n_test + n_val]
        train_idx = all_indices[n_test + n_val :]

        features = dataset.metadata.get("feature_names", [])

        return {
            "single_run": FoldData(
                train=self._make_split_by_indices(dataset, train_idx),
                val=self._make_split_by_indices(dataset, val_idx),
                test=self._make_split_by_indices(dataset, test_idx),
                name="single_run",
                feature_names=features,
            )
        }


class RandomSegmentCVSplitter(DataSplitter):
    """
    K-Fold Cross-Validation purely on the segment (row) level.
    Ignores whether segments belong to the same case or subject.
    """

    def __init__(self, n_splits: int = 5, val_ratio: float = 0.15):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def _make_split_by_indices(
        self, dataset: Dataset, indices: np.ndarray
    ) -> DataSplit:
        """Helper to create a DataSplit based on row indices."""
        return DataSplit(
            X=dataset.X[indices],
            y=dataset.y[indices],
            subject_ids=dataset.subject_ids[indices],
            case_ids=dataset.case_ids[indices],
        )

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        unique_subjects = np.unique(dataset.subject_ids)

        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")
        self.logger.log(f"  - Total segments: {dataset.n_samples}", level="info")

        rng = np.random.default_rng(seed)

        all_indices = np.arange(dataset.n_samples)
        rng.shuffle(all_indices)

        chunks = np.array_split(all_indices, self.n_splits)
        folds = {}
        features = dataset.metadata.get("feature_names", [])

        for i in range(self.n_splits):
            test_idx = chunks[i]
            train_val_idx = np.concatenate([c for j, c in enumerate(chunks) if j != i])

            if self.val_ratio > 0.0:
                n_val = max(1, int(len(train_val_idx) * self.val_ratio))
            else:
                n_val = 0

            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]

            folds[f"fold_{i}"] = FoldData(
                train=self._make_split_by_indices(dataset, train_idx),
                val=self._make_split_by_indices(dataset, val_idx),
                test=self._make_split_by_indices(dataset, test_idx),
                name=f"fold_{i}",
                feature_names=features,
            )
        return folds


class LatestThreeSessionSplitter(DataSplitter):
    """Select participants with 3+ sessions, take latest 3, create 3 folds.

    Fold i uses session i (0-indexed, sorted ascending) as test,
    the other 2 sessions as train. No validation split.
    """

    def __init__(self, n_splits: int = 3, val_ratio: float = 0.0):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    @staticmethod
    def _detect_roles(dataset: Dataset) -> tuple[str, str]:
        n_subjects = len(np.unique(dataset.subject_ids))
        n_cases = len(np.unique(dataset.case_ids))

        subj_to_cases = {}
        for subj in np.unique(dataset.subject_ids):
            subj_to_cases[subj] = len(np.unique(dataset.case_ids[dataset.subject_ids == subj]))

        case_to_subjs = {}
        for cid in np.unique(dataset.case_ids):
            case_to_subjs[cid] = len(np.unique(dataset.subject_ids[dataset.case_ids == cid]))

        mean_cases_per_subj = float(np.mean(list(subj_to_cases.values())))
        mean_subjs_per_case = float(np.mean(list(case_to_subjs.values())))

        # The participant field has FEWER unique values and a higher mean sessions per participant
        if n_subjects < n_cases or (n_subjects == n_cases and mean_cases_per_subj > mean_subjs_per_case):
            return "subject", "case"
        else:
            return "case", "subject"

    def _make_split(self, dataset: Dataset, participant_ids: np.ndarray, role: str) -> DataSplit:
        if role == "subject":
            mask = np.isin(dataset.subject_ids, participant_ids)
        else:
            mask = np.isin(dataset.case_ids, participant_ids)
        return DataSplit(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
        )

    def _session_values(self, dataset: Dataset, participant_id, role: str) -> np.ndarray:
        if role == "subject":
            mask = dataset.subject_ids == participant_id
            return np.unique(dataset.case_ids[mask])
        else:
            mask = dataset.case_ids == participant_id
            return np.unique(dataset.subject_ids[mask])

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        role, _ = self._detect_roles(dataset)
        participant_col = dataset.subject_ids if role == "subject" else dataset.case_ids
        unique_participants = np.unique(participant_col)

        self.logger.log(f"Splitter Stats (LatestThreeSession):", level="info")
        self.logger.log(f"  - Role: participant={role}, session={'case' if role == 'subject' else 'subject'}", level="info")
        self.logger.log(f"  - Total raw participants: {len(unique_participants)}", level="info")

        eligible_participants = []
        participant_sessions: list[tuple] = []

        for pid in unique_participants:
            sessions = self._session_values(dataset, pid, role)
            if len(sessions) >= 3:
                eligible_participants.append(pid)
                sorted_sessions = np.sort(sessions)
                participant_sessions.append((pid, sorted_sessions[-3:]))

        self.logger.log(f"  - Participants with 3+ sessions: {len(eligible_participants)}", level="info")

        if not eligible_participants:
            raise ValueError("No participants have at least 3 sessions; cannot run 3-fold CV")

        features = dataset.metadata.get("feature_names", [])
        folds = {}

        for fold_idx in range(self.n_splits):
            fold_test_mask = np.zeros(dataset.n_samples, dtype=bool)
            fold_train_mask = np.zeros(dataset.n_samples, dtype=bool)

            for pid, sessions in participant_sessions:
                test_session = sessions[fold_idx]
                train_sessions = [s for i, s in enumerate(sessions) if i != fold_idx]

                if role == "subject":
                    # participant=subject_id, session=case_id
                    pid_mask = dataset.subject_ids == pid
                    for sid in sessions:
                        s_mask = pid_mask & (dataset.case_ids == sid)
                        if sid == test_session:
                            fold_test_mask |= s_mask
                        else:
                            fold_train_mask |= s_mask
                else:
                    # participant=case_id, session=subject_id
                    pid_mask = dataset.case_ids == pid
                    for sid in sessions:
                        s_mask = pid_mask & (dataset.subject_ids == sid)
                        if sid == test_session:
                            fold_test_mask |= s_mask
                        else:
                            fold_train_mask |= s_mask

            features = dataset.metadata.get("feature_names", [])

            fold_name = f"fold_{fold_idx}"
            folds[fold_name] = FoldData(
                train=DataSplit(
                    X=dataset.X[fold_train_mask],
                    y=dataset.y[fold_train_mask],
                    subject_ids=dataset.subject_ids[fold_train_mask],
                    case_ids=dataset.case_ids[fold_train_mask],
                ),
                val=DataSplit(
                    X=np.empty((0, dataset.X.shape[1]), dtype=dataset.X.dtype),
                    y=np.empty(0, dtype=dataset.y.dtype),
                    subject_ids=np.empty(0, dtype=dataset.subject_ids.dtype),
                    case_ids=np.empty(0, dtype=dataset.case_ids.dtype),
                ),
                test=DataSplit(
                    X=dataset.X[fold_test_mask],
                    y=dataset.y[fold_test_mask],
                    subject_ids=dataset.subject_ids[fold_test_mask],
                    case_ids=dataset.case_ids[fold_test_mask],
                ),
                name=fold_name,
                feature_names=features,
            )

        return folds


class HybridMultiSessionSplitter(DataSplitter):
    """Multi-session subject CV with per-subject personalization training case."""

    def __init__(self, n_splits: int = 5, val_ratio: float = 0.15):
        self.n_splits = n_splits
        self.val_ratio = val_ratio
        self.logger = ProgressReporter()

    def _make_split(self, dataset: Dataset, cases: np.ndarray) -> DataSplit:
        mask = np.isin(dataset.case_ids, cases)
        return DataSplit(
            X=dataset.X[mask],
            y=dataset.y[mask],
            subject_ids=dataset.subject_ids[mask],
            case_ids=dataset.case_ids[mask],
        )

    def split(self, dataset: Dataset, seed: Optional[int] = 42) -> Dict[str, FoldData]:
        rng = np.random.default_rng(seed)
        features = dataset.metadata.get("feature_names", [])
        self.logger.log(f"CASE ID:::: {dataset.metadata.items()}", level="info")
        # Map subjects to their corresponding cases
        unique_subjects = np.unique(dataset.subject_ids)
        
        subj_to_cases = {}

        raw_max_sessions = 0
        raw_total_multi_cases = 0
        raw_single_sess_count = 0
        raw_multi_sess_count = 0

        for subj in unique_subjects:
            mask = dataset.subject_ids == subj
            cases = np.unique(dataset.case_ids[mask])

            n_cases = len(cases)
            if n_cases > raw_max_sessions:
                raw_max_sessions = n_cases

            if n_cases == 1:
                raw_single_sess_count += 1
            else:
                raw_multi_sess_count += 1
                raw_total_multi_cases += n_cases

            if len(cases) > 2:
                rng.shuffle(cases)
                cases = cases[:2]

            subj_to_cases[subj] = cases

        # Partition subjects by session count
        single_sess_subjs = [s for s, c in subj_to_cases.items() if len(c) == 1]
        multi_sess_subjs = np.array([s for s, c in subj_to_cases.items() if len(c) > 1])

        # Logging dataset statistics
        session_counts = [len(c) for c in subj_to_cases.values()]
        total_multi_cases = sum(len(subj_to_cases[s]) for s in multi_sess_subjs)
        self.logger.log(f"Splitter Stats:", level="info")
        self.logger.log(f"  - Total subjects: {len(unique_subjects)}", level="info")
        self.logger.log(
            f"  - Single-session subjects: {raw_single_sess_count}", level="info"
        )
        self.logger.log(
            f"  - Multi-session subjects: {raw_multi_sess_count}", level="info"
        )
        self.logger.log(
            f"  - Max sessions per subject: {raw_max_sessions} (Original) -> 2 (After Trim)",
            level="info",
        )
        self.logger.log(
            f"  - Total cases in multi-session group: {raw_total_multi_cases} (Original) -> {total_multi_cases} (Used)",
            level="info",
        )
        self.logger.log(
            f"  -> Removed {raw_total_multi_cases - total_multi_cases} excess cases for fair comparison.",
            level="info",
        )
        # Always include single-session cases in training pool
        base_train_cases = []
        for s in single_sess_subjs:
            for c in subj_to_cases[s]:
                base_train_cases.append(c)

        rng.shuffle(multi_sess_subjs)
        chunks = np.array_split(multi_sess_subjs, self.n_splits)

        folds = {}
        for i in range(self.n_splits):
            test_subjs = chunks[i]
            train_subjs = np.concatenate([c for j, c in enumerate(chunks) if j != i])

            fold_test_cases = []
            personalization_train_cases = []
            fold_train_pool = list(base_train_cases)

            # Process test subjects: 1 case to test, 1 case strictly to train
            for s in test_subjs:
                cases = list(subj_to_cases[s])
                rng.shuffle(cases)
                fold_test_cases.append(cases[0])
                personalization_train_cases.append(cases[1])

            # Add all cases from other multi-session subjects to train pool
            for s in train_subjs:
                fold_train_pool.extend(subj_to_cases[s])

            rng.shuffle(fold_train_pool)
            n_val = (
                max(1, int(len(fold_train_pool) * self.val_ratio))
                if self.val_ratio > 0
                else 0
            )

            val_cases = fold_train_pool[:n_val]
            train_cases = fold_train_pool[n_val:] + personalization_train_cases

            folds[f"fold_{i}"] = FoldData(
                train=self._make_split(dataset, np.array(train_cases)),
                val=self._make_split(dataset, np.array(val_cases)),
                test=self._make_split(dataset, np.array(fold_test_cases)),
                name=f"fold_{i}",
                feature_names=features,
            )

        return folds
