from __future__ import annotations

import gc
import json
import random
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf

from src.evaluator import Evaluator
from src.models.core.model import Model
from src.models.dataset.base import Dataset, FoldData
from src.models.dataset.filter import TargetRangeFilter
from src.models.dataset.splitter import (
    CaseWiseCrossValidationSplitter,
    CaseWiseSingleSplitter,
    CrossValidationSplitter,
    HybridMultiSessionSplitter,
    LatestThreeSessionSplitter,
    RandomSegmentCVSplitter,
    RandomSegmentSplitter,
    SingleSplitter,
)
from src.models.training.base_trainer import Trainer, TrainingConfig
from src.progressbar import ProgressReporter


@dataclass
class RunResult:
    """Simple container to hold the results of a runner execution."""

    best_fold: str
    best_val_rmse: float


class BaseModelRunner(ProgressReporter, ABC):
    """Shared layer for model training and evaluation runs.

    Concrete runners implement model, dataset, and trainer,
    while this base class handles the pipeline:
    dataset preparation, deterministic seeding, execution,
    artifact finalization, and report generation.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        raw_params = config.get("params")
        self.params = raw_params.copy() if isinstance(raw_params, dict) else {}

    @abstractmethod
    def _make_model(self) -> Model: ...

    @abstractmethod
    def _make_dataset(self, data_path: Path) -> Dataset: ...

    @abstractmethod
    def _make_trainer(self, model: Model, train_cfg: TrainingConfig) -> Trainer: ...

    def _make_evaluator(self) -> Any:
        """Creates the evaluator writing into the configured artifact directory."""
        from src.evaluator import Evaluator

        out = Path(self.config.get("artifact_dir", "experiments"))
        out.mkdir(parents=True, exist_ok=True)
        return Evaluator(out)

    def _build_training_config(self, seed: Optional[int]) -> TrainingConfig:
        """Extracts and builds the training configuration."""
        warm_dir = self.config.get("warm_start_dir")
        warm_model_path = None

        if warm_dir:
            if Path(warm_dir).exists():
                warm_model_path = warm_dir
            else:
                raise FileNotFoundError(f"Warm start directory not found: {warm_dir}")

        return TrainingConfig(
            epochs=self.config.get("epochs", self.params.get("epochs", 50)),
            batch_size=self.config.get("batch_size", self.params.get("batch_size", 32)),
            learning_rate=self.config.get(
                "learning_rate", self.params.get("learning_rate", 1e-3)
            ),
            early_stopping_patience=self.config.get(
                "early_stopping_patience",
                self.params.get("early_stopping_patience", 10),
            ),
            fit_verbose=self.config.get("fit_verbose", 0),
            seed=seed,
            init_from_dir=warm_model_path,
        )

    def _ckpt_path(self, fold_idx: int, kind: str) -> Path:
        """Path of a fold marker"""
        d = Path(self.config["artifact_dir"]) / "folds" / "_checkpoints"
        return d / f"fold_{fold_idx}_{kind}.json"

    def _write_ckpt(self, path: Path, payload: dict) -> None:
        """Writes a checkpoint marker"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")

        def _json_safe(o):
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)

        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, default=_json_safe)
        tmp.replace(path)

    def _verify_ckpt_seed(self, info: dict, fold_idx: int) -> None:
        """Guards against a silent resume with a different seed"""
        stored = info.get("seed")
        current = int(self.params.get("seed", -1))
        if stored is not None and stored != current:
            raise RuntimeError(
                f"[checkpoint] Seed mismatch at fold {fold_idx}: checkpoint has "
                f"seed={stored}, current run has seed={current}."
            )
        if current < 0:
            self.log(
                f"[checkpoint] WARNING: resuming fold {fold_idx} in random mode "
                f"(seed=-1) - fold splits are not reproducible, resume "
                f"correctness is not guaranteed.",
                level="info",
            )

    def _run_cv(
        self, folds: Dict[str, FoldData], trainer: Trainer, evaluator: Evaluator
    ) -> RunResult:
        """Executes cross-validation on all dataset folds."""
        best_val_rmse = float("inf")
        best_fold = ""

        force = bool(self.config.get("force_retrain", False))

        fold_items = list(folds.items())
        it = range(len(fold_items))
        if hasattr(self, "pbar") and self.pbar:
            it = self.pbar(it, desc="CV Folds")

        for fold_idx in it:
            fold_name, fold = fold_items[fold_idx]

            trained_ckpt = self._ckpt_path(fold_idx, "trained")
            eval_pkl = self.config["artifact_dir"]
            eval_pkl = (
                Path(self.config["artifact_dir"])
                / f"results_checkpoint_{fold_name}.pkl"
            )

            # Case 1: fold fully done -> skip entirely
            if not force and eval_pkl.exists():
                self.log(
                    f"[checkpoint] Fold {fold_idx} already evaluated - skipping.",
                    level="info",
                )

                import joblib

                payload = joblib.load(eval_pkl)
                self._verify_ckpt_seed(payload, fold_idx)
                vr = payload.get("val_rmse")
                if vr is not None and vr < best_val_rmse:
                    best_val_rmse = vr
                    best_fold = fold_name
                continue

            fold_path = Path(self.config["artifact_dir"]) / "folds" / f"fold_{fold_idx}"
            probe = self._make_model()
            model_artifact_ok = probe.load(fold_path)

            # Case 2: Training: fresh OR restored from checkpoint
            if not force and trained_ckpt.exists() and model_artifact_ok:
                info = json.loads(trained_ckpt.read_text(encoding="utf-8"))
                self._verify_ckpt_seed(info, fold_idx)
                self.log(
                    f"[checkpoint] Fold {fold_idx} already trained - "
                    f"resuming at test prediction.",
                    level="info",
                )
                val_rmse = info["val_rmse"]
                history = info.get("history", {})
            else:
                if not force and trained_ckpt.exists() and not model_artifact_ok:
                    self.log(
                        f"[checkpoint] Fold {fold_idx}: model is missing - retraining.",
                        level="info",
                    )
                    trained_ckpt.unlink()
                del probe

                _, train_result = trainer.train_fold(fold, fold_idx)
                val_rmse = train_result.val_rmse
                history = train_result.history

                self._write_ckpt(
                    trained_ckpt,
                    {
                        "val_rmse": float(val_rmse),
                        "history": history,
                        "model_size_kb": getattr(train_result, "model_size_kb", 0.0),
                        "seed": int(self.params.get("seed", -1)),
                    },
                )

            evaluator.result.training_curves[fold_name] = history

            if fold.test.X.shape[0] != 0:

                # TODO: add to return value for train_fold..
                fold_path = (
                    Path(self.config["artifact_dir"]) / "folds" / f"fold_{fold_idx}"
                )

                self.log(
                    f"Reloading model from {fold_path} for test prediction...",
                    level="info",
                )

                eval_model = self._make_model()

                if not eval_model.load(fold_path):
                    raise FileNotFoundError(
                        f"Could not reload fold model from {fold_path}.keras"
                    )
                self.log("Reload successful.", level="info")

                self.log(
                    f"Start predicting the test set ({fold.test.X.shape[0]} samples).",
                    level="info",
                )
                y_pred = eval_model.predict(fold.test.X)
                self.log("Prediction done. Logging results.", level="info")

                evaluator.log_results(
                    name=fold_name,
                    y_true=fold.test.y,
                    y_pred=y_pred,
                    case_ids=fold.test.case_ids,
                    history=history,
                    val_rmse=val_rmse,
                    seed=int(self.params.get("seed", -1)),
                )

                self.log(f"[checkpoint] Fold {fold_idx} completed.", level="info")

                del eval_model
                gc.collect()

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_fold = fold_name

        return RunResult(best_fold=best_fold, best_val_rmse=best_val_rmse)

    def _run_single(
        self, folds: Dict[str, FoldData], trainer: Trainer, evaluator: Any
    ) -> RunResult:
        """Executes training on a single dataset split."""
        fold_name, fold = next(iter(folds.items()))

        fold_idx = 0
        fold_name = f"fold_{fold_idx}"

        if hasattr(self, "pbar") and self.pbar:
            self.pbar(range(1), desc="Single Run")

        # Train
        self.log("Starting training for fold_0...", level="info")
        trained_model, train_result = trainer.train_fold(fold, fold_idx=0)
        evaluator.result.training_curves[fold_name] = train_result.history
        self.log(
            f"--- Training Finished (RMSE: {train_result.val_rmse:.4f}) ---",
            level="info",
        )

        # Evaluate
        if len(fold.test.y) > 0:
            y_pred = trained_model.predict(fold.test.X)
            self.log(
                f"Prediction completed. Logging results to evaluator...", level="info"
            )
            evaluator.log_results(
                name=fold_name,
                y_true=fold.test.y,
                y_pred=y_pred,
                case_ids=fold.test.case_ids,
            )
            self.log("Evaluation report generated.", level="info")
        else:
            if hasattr(self, "log"):
                self.log(
                    "Skipping evaluation: Test set is empty.",
                    level="info",
                )

        return RunResult(best_fold=fold_name, best_val_rmse=train_result.val_rmse)

    def _run_eval(self, dataset: Dataset, model: Model, evaluator: Any) -> RunResult:
        """Executes inference on the dataset using an already loaded model."""
        y_pred = model.predict(dataset.X)

        evaluator.log_results(
            name="eval",
            y_true=dataset.y,
            y_pred=y_pred,
            case_ids=dataset.case_ids,
        )

        return RunResult(best_fold="eval", best_val_rmse=float("nan"))

    def run(
        self,
        *,
        mode: str,
        name: str,
        data_path: Path,
        model_path: Optional[Path] = None,
        results_name: str = "results.pkl",
        split_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Executes a full model workflow for training or evaluation."""

        split_dict: Dict[str, Any] = split_kwargs if split_kwargs is not None else {}

        t0 = self.run_start(step="model run", name=name, note=f"Mode: {mode}")

        try:
            evaluator = self._make_evaluator()
            model = self._make_model()

            # -------- Seed Handling --------
            seed_val = int(self.params.get("seed", -1))
            seed = seed_val if seed_val >= 0 else None

            if seed is not None:
                self.log(f"Seed: {seed}", level="debug")
                # Ensure complete reproducibility ONLY if a valid seed is provided
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)
            else:
                self.log("Seed: None (Random Mode)", level="debug")
            # -------- Dataset --------
            dataset: Dataset = self._make_dataset(data_path)
            self.log(f"Dataset metadata: {dataset.metadata}", level="debug")

            target_range = self.config.get("target_range")
            self.log(f"range: {target_range}")
            if target_range:
                range_filter = TargetRangeFilter(target_range)
                pre_filter_samples = dataset.n_samples
                dataset = range_filter.apply(dataset)
                self.log(
                    f"Filtered dataset down from {pre_filter_samples} to {dataset.n_samples} samples.",
                    level="info",
                )
            # -------------------------------

            SPLITTER_REGISTRY = {
                "train_cv": (CrossValidationSplitter, self._run_cv),
                "train_single": (SingleSplitter, self._run_single),
                "train_cv_casewise": (CaseWiseCrossValidationSplitter, self._run_cv),
                "train_single_casewise": (CaseWiseSingleSplitter, self._run_single),
                "train_single_random": (RandomSegmentSplitter, self._run_single),
                "train_cv_random": (RandomSegmentCVSplitter, self._run_cv),
                "train_cv_multi_session": (HybridMultiSessionSplitter, self._run_cv),
                "train_cv_three_latest": (LatestThreeSessionSplitter, self._run_cv),
            }

            if mode == "eval":
                if model_path is None:
                    raise ValueError("eval requires model_path")
                if not model.load(model_path):
                    raise ValueError(f"Could not load model: {model_path}")
                self.log(f"eval model path: {model_path}", level="debug")
                result = self._run_eval(dataset, model, evaluator)

            elif mode in SPLITTER_REGISTRY:
                SplitterClass, run_method = SPLITTER_REGISTRY[mode]

                kwargs = split_dict.copy()
                if "cv" in mode:
                    kwargs.setdefault("n_splits", 5)
                    kwargs.setdefault("val_ratio", 0.15)
                else:
                    kwargs.setdefault("test_ratio", 0.15)
                    kwargs.setdefault("val_ratio", 0.15)

                splitter = SplitterClass(**kwargs)
                folds = splitter.split(dataset, seed=seed)

                trainer = self._make_trainer(model, self._build_training_config(seed))
                result = run_method(folds, trainer, evaluator)

            else:
                available = ", ".join(["eval"] + list(SPLITTER_REGISTRY.keys()))
                raise ValueError(f"Unknown mode '{mode}'. Available modes: {available}")

            # -------- Save and Finalize Artifacts --------
            final_size_kb = 0.0

            # Initial placeholder save
            evaluator.save_report(
                filename=results_name,
                config=self.config,
                model_params=self.params,
                provenance=self.config.get("provenance", {}),
                best_model_size_kb=0.0,
            )

            if mode != "eval":
                artifact_dir = Path(self.config["artifact_dir"])
                final_dir = artifact_dir / "final"
                final_dir.mkdir(parents=True, exist_ok=True)

                best_fold_name = result.best_fold

                # Save JSON summary of the best fold
                final_info = {
                    "best_fold": best_fold_name,
                    "best_val_rmse": result.best_val_rmse,
                    "seed": (seed if seed is not None else -1),
                    "params": self.params,
                    "feature_names": dataset.metadata.get("feature_names", []),
                }
                with open(final_dir / "summary.json", "w") as f:
                    json.dump(final_info, f, indent=4)

                # Move best model files to the final directory
                fold_dir = artifact_dir / "folds"
                for src_file in fold_dir.glob(f"{best_fold_name}*"):
                    dest_file = final_dir / src_file.name.replace(
                        best_fold_name, "best_model"
                    )
                    shutil.copy2(src_file, dest_file)

                    if "int8.tflite" in src_file.name:
                        # Prefer INT8 size when available (deployment target).
                        final_size_kb = dest_file.stat().st_size / 1024
                    elif "float.tflite" in src_file.name:
                        # Fallback to float TFLite if no INT8 artifact exists.
                        final_size_kb = dest_file.stat().st_size / 1024
                    elif src_file.suffix in [".joblib"]:
                        final_size_kb = model.get_model_size_kb()
                    elif final_size_kb == 0.0 and src_file.suffix in [
                        ".keras",
                        ".json",
                        ".joblib",
                    ]:
                        final_size_kb = dest_file.stat().st_size / 1024

            self.params["used_feature_names"] = dataset.metadata.get(
                "feature_names", []
            )

            evaluator.save_report(
                filename=results_name,
                config=self.config,
                model_params=self.params,
                provenance=self.config.get("provenance", {}),
                best_model_size_kb=final_size_kb,
            )

            self.run_end("model run", t0=t0)
            return result.best_val_rmse

        except Exception as e:
            # self.handle_exception(e, context=f"Model Run [{name}]")
            raise
