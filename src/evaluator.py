from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.ceg import compute_clarke_zone_accuracies

METRIC_KEYS = [
    "RMSE",
    "MAE",
    "MARD",
    "R2",
    "Zone A",
    "Zone B",
    "Zone C",
    "Zone D",
    "Zone E",
]


@dataclass
class ModelMetrics:
    """Holds regression and Clarke Error Grid zone metrics for a single evaluation."""

    RMSE: float = np.nan
    MAE: float = np.nan
    MARD: float = np.nan
    R2: float = np.nan
    Zone_A: float = np.nan
    Zone_B: float = np.nan
    Zone_C: float = np.nan
    Zone_D: float = np.nan
    Zone_E: float = np.nan

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "ModelMetrics":
        """Creates a ModelMetrics instance from a dictionary, defaulting missing keys to NaN."""
        return cls(
            RMSE=d.get("RMSE", np.nan),
            MAE=d.get("MAE", np.nan),
            MARD=d.get("MARD", np.nan),
            R2=d.get("R2", np.nan),
            Zone_A=d.get("Zone A", np.nan),
            Zone_B=d.get("Zone B", np.nan),
            Zone_C=d.get("Zone C", np.nan),
            Zone_D=d.get("Zone D", np.nan),
            Zone_E=d.get("Zone E", np.nan),
        )

    def to_dict(self) -> Dict[str, float]:
        """Converts the metrics back to a flat dictionary."""
        return {
            "RMSE": self.RMSE,
            "MAE": self.MAE,
            "MARD": self.MARD,
            "R2": self.R2,
            "Zone A": self.Zone_A,
            "Zone B": self.Zone_B,
            "Zone C": self.Zone_C,
            "Zone D": self.Zone_D,
            "Zone E": self.Zone_E,
        }


@dataclass
class Result:
    """Aggregates all evaluation data for one benchmark run across folds and cases."""

    ceg_type: str = field(default="clarke")
    summary: dict = field(default_factory=dict)

    fold_metrics: Dict[str, ModelMetrics] = field(default_factory=dict)
    case_metrics: Dict[str, Dict[int, ModelMetrics]] = field(default_factory=dict)

    config: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    predictions: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    training_curves: dict = field(default_factory=dict)
    inference_times_ms: dict = field(default_factory=dict)
    layer_outputs: dict = field(default_factory=dict)

    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    schema_version: int = 1

    def add_fold_summary(
        self,
        name: str,
        metrics: dict,  # TODO: change to ModelMetric
        y_true: np.ndarray,
        y_pred: np.ndarray,
        case_ids: np.ndarray,
    ):
        """Stores per-fold metrics and the associated predictions."""
        mm = ModelMetrics.from_dict(metrics)
        self.fold_metrics[name] = mm

        self.predictions[name] = {
            "y_true": y_true.astype(np.float32),
            "y_pred": y_pred.astype(np.float32),
            "case_ids": case_ids.astype(np.int32),
        }
        return mm

    def add_case_summary(self, fold_context: str, case_id: int, metrics: dict):
        """Records per-case metrics under the given fold."""
        if fold_context not in self.case_metrics:
            self.case_metrics[fold_context] = {}

        self.case_metrics[fold_context][case_id] = ModelMetrics.from_dict(metrics)

    @classmethod
    def load(cls, path: Path) -> Result:
        """Deserializes a Result from a joblib/pkl file."""
        data = joblib.load(path)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path, compress: int = 3):
        """Serializes this Result to a compressed joblib/pkl file."""
        joblib.dump(asdict(self), path, compress=compress)

    @property
    def rmse(self) -> float:
        """Shortcut to the mean RMSE from the summary statistics."""
        return self.summary.get("mean", {}).get("RMSE", np.nan)

    @property
    def mode(self) -> str:
        """Returns the run mode (e.g. 'eval' or 'train') from the stored config."""
        m = str(self.config.get("mode", "eval")).lower()
        return m


class Evaluator:
    """
    Stores a run report as a single joblib/pkl file (single source of truth).

    Report structure (final_data):
      - schema_version
      - summary: {"mean": {...}, "std": {...}}
      - folds:   [ {"Fold": ..., "RMSE": ..., "Zone A": ...}, ... ]
      - predictions: {fold_name: {"y_true": np.ndarray, "y_pred": np.ndarray}} (optional)
      - config, model_params, provenance, timestamp

    Notes:
      - Clarke Error Grid (CEG) is computed via compute_clarke_zone_accuracies(pred, ref).
      - Predictions are stored as float32 by default to reduce file size.
      - CEG clipping is optional. If ceg_clip is None, no clipping is applied.
    """

    def __init__(
        self,
        artifact_dir: Path,
    ):
        """Initializes the evaluator and ensures the artifact directory exists."""
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.result = Result()

    @staticmethod
    def _align(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts inputs to 1D float arrays, aligns lengths, and removes non-finite values.
        """
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        n = min(len(yt), len(yp))
        yt = yt[:n]
        yp = yp[:n]
        m = np.isfinite(yt) & np.isfinite(yp)
        return yt[m], yp[m]

    @staticmethod
    def _calculate_mard(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Relative Difference (%).
        """
        mask = y_true != 0
        if np.sum(mask) == 0:
            return 0.0
        diff = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
        return float(np.mean(diff) * 100.0)

    def _calculate_metrics_dict(
        self, yt: np.ndarray, yp: np.ndarray
    ) -> Dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        mard = self._calculate_mard(yt, yp)

        metrics = {"RMSE": rmse, "MAE": mae, "MARD": mard, "R2": r2}
        ceg = compute_clarke_zone_accuracies(yp, yt)
        metrics.update({f"Zone {z}": float(ceg.get(f"Zone {z}", 0.0)) for z in "ABCDE"})
        return metrics

    def log_results(
        self,
        name: str,
        y_true,
        y_pred,
        case_ids,
        *,
        history=None,
        val_rmse=None,
        seed=None,
    ) -> ModelMetrics:
        """Evaluates one fold: computes metrics, stores predictions, and checkpoints the result."""
        yt, yp = self._align(y_true, y_pred)
        cids = np.asarray(case_ids).ravel()[: len(yt)]

        if len(yt) == 0:
            return ModelMetrics()

        fold_metrics = self._calculate_metrics_dict(yt, yp)

        unique_cids = np.unique(cids)
        case_metrics = {}
        for cid in unique_cids:
            mask = cids == cid
            case_metrics[int(cid)] = self._calculate_metrics_dict(yt[mask], yp[mask])

        payload = {
            "name": name,
            "metrics": fold_metrics,
            "case_metrics": case_metrics,
            "predictions": {
                "y_true": yt.astype(np.float32),
                "y_pred": yp.astype(np.float32),
                "case_ids": cids.astype(np.int32),
            },
            "history": history or {},
            "val_rmse": (float(val_rmse) if val_rmse is not None else None),
            "seed": (int(seed) if seed is not None else None),
        }

        checkpoint_path = self.artifact_dir / f"results_checkpoint_{name}.pkl"
        tmp = checkpoint_path.with_suffix(".pkl.tmp")
        joblib.dump(payload, tmp, compress=3)
        tmp.replace(checkpoint_path)

        return ModelMetrics.from_dict(fold_metrics)

    def save_report(
        self,
        filename: str = "cv_results.pkl",
        *,
        config: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        include_predictions: bool = True,
        best_model_size_kb: Optional[float] = None,
        compress: Optional[Union[int, bool]] = 3,
        expected_folds_count: Optional[int] = None,
    ) -> Path:
        """Aggregates fold metrics into a summary and persists the full Result to disk."""

        files = sorted(self.artifact_dir.glob("results_checkpoint_*.pkl"))

        self.result.fold_metrics = {}
        self.result.case_metrics = {}
        self.result.predictions = {}

        found = set()
        for fp in files:
            try:
                payload = joblib.load(fp)
            except Exception as e:
                raise RuntimeError(
                    f"[report] Corrupt per-fold checkpoint {fp}: {e}. "
                    f"Refusing to write an incomplete summary."
                )

            name = payload["name"]
            found.add(name)

            preds = payload["predictions"]

            self.result.add_fold_summary(
                name,
                payload["metrics"],
                preds["y_true"],
                preds["y_pred"],
                preds["case_ids"],
            )

            for cid, cm in payload["case_metrics"].items():
                self.result.add_case_summary(name, int(cid), cm)

            if payload.get("history"):
                self.result.training_curves[name] = payload["history"]

        if not include_predictions:
            self.result.predictions = {}

        if expected_folds_count is not None:
            if len(found) != expected_folds_count:
                raise RuntimeError(
                    f"[report] Final summary INCOMPLETE: found {len(found)} fold(s) "
                    f"{sorted(found)}, expected {expected_folds_count}. Refusing to "
                    f"write a CV mean over the wrong number of folds. Check the "
                    f"results_checkpoint_*.pkl files in {self.artifact_dir}."
                )

        rows = []
        for fold, mm in self.result.fold_metrics.items():
            rows.append({"Fold": fold, **mm.to_dict()})

        df = pd.DataFrame(rows)

        self.result.summary = {
            "mean": df.mean(numeric_only=True).to_dict(),
            "std": df.std(numeric_only=True).to_dict(),
            "model_size_kb": best_model_size_kb,
        }
        if config is not None:
            self.result.config = config

        if model_params is not None:
            self.result.model_params = model_params

        if provenance is not None:
            self.result.provenance = provenance

        path = self.artifact_dir / filename
        self.result.save(path)
        return path
