from __future__ import annotations

import hashlib
import inspect
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from src.builders import get_builder
from src.extractors import get_extractor
from src.models.factory import get_model
from src.models.runners.base_runner import BaseModelRunner
from src.processors import get_processor
from src.progressbar import ProgressReporter

TRANSIENT_CONFIG_KEYS = (
    "progress",
    "log_level",
    "force",
    "force_overwrite",
    "show_fold_metrics",
    "fit_verbose",
)

MODEL_EXTENSIONS = {
    "keras": [".h5", ".keras"],
    "sklearn": [".json", ".pkl"],
}

KERAS_MODEL_TYPES = ["tiny_cnn", "resnet34", "tcn"]
RESULTS_FILE_NAME = "results.pkl"
SUCCESS_FILE_ENDING = ".success"


def stable_hash(obj: Any) -> str:
    """Create a stable short hash for a JSON-serializable object."""
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(s).hexdigest()[:10]


def ensure_parent(p: Path) -> None:
    """Ensure the parent directory of a path exists."""
    p.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def as_list(x: Any) -> List[Any]:
    """Normalize a scalar or sequence into a list."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def get_model_filename(model_type: str) -> str:
    """Get the model filename based on model type."""
    if model_type in KERAS_MODEL_TYPES:
        return "final/best_model.keras"
    if model_type in ("pls", "random_forest"):
        return "final/best_model.joblib"
    return "final/best_model.json"


def get_summary_mean_std(
    results_pkl: Path,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Load summary mean and std metric dictionaries from a results file."""
    d = joblib.load(results_pkl)
    summary = d.get("summary", {}) or {}

    mean = summary.get("mean", {}) if isinstance(summary, dict) else {}
    std = summary.get("std", {}) if isinstance(summary, dict) else {}

    return {k: float(v) for k, v in (mean or {}).items()}, {
        k: float(v) for k, v in (std or {}).items()
    }


def get_model_size(results_pkl: Path) -> float:
    """Load model size in KB from a results file."""
    d = joblib.load(results_pkl)
    summary = d.get("summary", {}) or {}
    return summary.get("model_size_kb", 0.0)


def call_run_best_effort(obj: Any, **kwargs: Any) -> Any:
    """Call obj.run with only supported keyword arguments."""
    if not hasattr(obj, "run"):
        raise AttributeError(f"{obj.__class__.__name__} has no run() method")

    fn = obj.run
    try:
        sig = inspect.signature(fn)
        has_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        final_kwargs = (
            kwargs
            if has_var_kwargs
            else {k: v for k, v in kwargs.items() if k in sig.parameters}
        )
    except (ValueError, TypeError):
        final_kwargs = kwargs

    return fn(**final_kwargs) if final_kwargs else fn()


@dataclass(frozen=True)
class PipelineConfig:
    """Container for one concrete dataset-processor-builder-model pipeline."""

    dataset_name: str
    processor_name: str
    builder_name: str
    model_name: str

    dataset_cfg: Dict[str, Any]
    processor_cfg: Dict[str, Any]
    builder_cfg: Dict[str, Any]
    model_cfg: Dict[str, Any]


@dataclass(frozen=True)
class Task:
    """Execution unit describing one pipeline run request."""

    config: PipelineConfig
    mode: str = "train_cv"
    force_train: bool = False
    optimization: Optional[str] = None

    base_model_config: Optional[PipelineConfig] = None
    eval_model_path: Optional[str] = None
    base_model_mode: str = "None"
    split_ratio: Optional[Dict[str, float]] = (
        None  # For train_single: {"test_ratio": 0.15, "val_ratio": 0.15}
    )


@dataclass
class BenchmarkResult:
    """Compact result payload returned after a task finishes."""

    exp_name: str
    run_dir: Path
    dataset_name: str
    mode: str
    summary_mean: Dict[str, float]
    summary_std: Dict[str, float]
    model_size: float


class ConfigBuilder:
    """Helper class to build configurations cleanly."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def get_component_config(self, section: str, name: str) -> Dict[str, Any]:
        """Get configuration for a named component."""
        if name not in self.cfg.get(section, {}):
            raise KeyError(f"'{name}' not found in section '{section}'")
        return self.cfg[section][name]

    def build_pipeline_config(
        self, dataset: str, processor: str, builder: str, model: str
    ) -> PipelineConfig:
        """Build a pipeline configuration from component names."""
        return PipelineConfig(
            dataset_name=dataset,
            processor_name=processor,
            builder_name=builder,
            model_name=model,
            dataset_cfg=self.get_component_config("datasets", dataset),
            processor_cfg=self.get_component_config("processors", processor),
            builder_cfg=self.get_component_config("builders", builder),
            model_cfg=self.get_component_config("models", model),
        )


class JobValidator:
    """Validates a benchmark job configuration."""

    @staticmethod
    def validate_job(job: Dict[str, Any], mode: str) -> None:
        """Validate that a job has all required fields for its mode."""
        is_train = mode.startswith("train_")

        if not (is_train or mode == "eval"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be 'eval' or start with 'train_'"
            )

        # Required fields check
        if is_train:
            required = ["datasets", "processors", "builders", "models"]
            if "single" in mode:
                required.append("split_ratio")
        else:  # eval
            required = ["datasets", "processors", "builders", "eval_model_config"]

        for field in required:
            if field not in job:
                raise ValueError(
                    f"Job in mode '{mode}' missing required field: '{field}'"
                )

        # Mode-specific validation
        if mode == "eval":
            if "eval_model_config" not in job:
                raise ValueError(
                    "Mode 'eval' requires 'eval_model_config' to specify the source model"
                )
            if as_list(job.get("models")) and len(as_list(job.get("models"))) > 1:
                raise ValueError("Multiple models for eval is not allowed")

        if is_train and "single" in mode:
            JobValidator._validate_split_ratio(job.get("split_ratio"))

    @staticmethod
    def _validate_split_ratio(split_ratio: Any) -> None:
        """Validate split_ratio configuration for train_single mode."""
        if not isinstance(split_ratio, dict):
            raise ValueError("split_ratio must be a dictionary")

        required_keys = {"test_ratio", "val_ratio"}
        provided_keys = set(split_ratio.keys())

        if not required_keys.issubset(provided_keys):
            missing = required_keys - provided_keys
            raise ValueError(
                f"split_ratio missing keys: {missing}. Required: {required_keys}"
            )

        for key, value in split_ratio.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"split_ratio['{key}'] must be a number, got {type(value)}"
                )
            if not (0 <= value < 1):
                raise ValueError(
                    f"split_ratio['{key}'] must be between 0 and 1, got {value}"
                )

        total_ratio = split_ratio["test_ratio"] + split_ratio["val_ratio"]
        if total_ratio >= 1:
            raise ValueError(
                f"Sum of test_ratio and val_ratio must be < 1, got {total_ratio}"
            )

    @staticmethod
    def validate_quantization(model_cfg: Dict[str, Any], model_name: str) -> None:
        """Validate quantization configuration."""
        quant = model_cfg.get("config", {}).get("params", {}).get("quantization")
        base_model = model_cfg.get("config", {}).get("base_model")

        if quant and not base_model:
            raise ValueError(
                f"Model '{model_name}' sets quantization='{quant}' but has no config.base_model"
            )


class TaskFactory:
    """Factory for creating Task objects from job specifications."""

    def __init__(self, builder: ConfigBuilder):
        self.builder = builder

    def create_eval_tasks(
        self,
        job: Dict[str, Any],
    ) -> List[Task]:
        """Create evaluation tasks from job specification."""
        tasks = []

        datasets = as_list(job.get("datasets"))
        processors = as_list(job.get("processors"))
        builders = as_list(job.get("builders") or job.get("builder"))
        force_train = bool(job.get("force_train", False))
        eval_model_path = job.get("model_path")

        eval_config_dict = job.get("eval_model_config", {})
        if not eval_config_dict:
            raise ValueError(
                "Mode 'eval' requires 'eval_model_config' to specify the source model"
            )

        eval_models = as_list(
            eval_config_dict.get("models") or eval_config_dict.get("model")
        )
        if not eval_models:
            raise ValueError(
                "eval_model_config must contain at least one 'model' or 'models'"
            )

        base_mode = eval_config_dict.get("mode")
        if not base_mode:
            raise ValueError("base_mode is not selected!")
        base_split_ratio = eval_config_dict.get("split_ratio", None)

        eval_opt_list = as_list(eval_config_dict.get("optimizations") or [None])

        for eval_model in eval_models:
            for opt_name in eval_opt_list:
                clean_opt_name = str(opt_name).strip().lower() if opt_name else None
                actual_opt_value = None if clean_opt_name == "none" else opt_name

                base_config = self.builder.build_pipeline_config(
                    dataset=eval_config_dict["dataset"],
                    processor=eval_config_dict["processor"],
                    builder=eval_config_dict["builder"],
                    model=eval_model,
                )

                for ds, pr, b in product(datasets, processors, builders):
                    task = Task(
                        config=self.builder.build_pipeline_config(
                            ds, pr, b, eval_model
                        ),
                        mode="eval",
                        force_train=force_train,
                        base_model_config=base_config,
                        eval_model_path=eval_model_path,
                        base_model_mode=base_mode,
                        split_ratio=base_split_ratio,
                        optimization=actual_opt_value,
                    )
                    tasks.append(task)

        return tasks

    def create_training_tasks(
        self,
        job: Dict[str, Any],
    ) -> List[Task]:
        """Create training tasks from job specification."""
        tasks = []

        mode = job.get("mode")
        if not mode:
            raise ValueError("No mode was seleceted!")

        datasets = as_list(job.get("datasets"))
        processors = as_list(job.get("processors"))
        builders = as_list(job.get("builders") or job.get("builder"))
        models = as_list(job.get("models", []))

        split_ratio = job.get("split_ratio") if "single" in mode else None
        opt_list = as_list(job.get("optimizations") or [None])

        for ds, pr, b, m in product(datasets, processors, builders, models):
            base_pipeline_config = self.builder.build_pipeline_config(ds, pr, b, m)

            model_cfg = self.builder.get_component_config("models", m)
            JobValidator.validate_quantization(model_cfg, m)

            for opt_name in opt_list:
                current_model_cfg = deepcopy(base_pipeline_config.model_cfg)
                base_config = None

                clean_opt_name = str(opt_name).strip().lower()
                actual_opt_value = None if clean_opt_name == "none" else opt_name

                if actual_opt_value:
                    opt_params = self.builder.cfg.get("optimizations", {}).get(
                        actual_opt_value, {}
                    )

                    if "config" not in current_model_cfg:
                        current_model_cfg["config"] = {}
                    if "params" not in current_model_cfg["config"]:
                        current_model_cfg["config"]["params"] = {}

                    current_model_cfg["config"]["params"].update(opt_params)

                    base_config = base_pipeline_config

                task = Task(
                    config=PipelineConfig(
                        ds,
                        pr,
                        b,
                        m,
                        base_pipeline_config.dataset_cfg,
                        base_pipeline_config.processor_cfg,
                        base_pipeline_config.builder_cfg,
                        current_model_cfg,
                    ),
                    mode=mode,
                    optimization=actual_opt_value,
                    force_train=bool(job.get("force_train", False)),
                    base_model_config=base_config,
                    eval_model_path=job.get("model_path"),
                    base_model_mode=mode,
                    split_ratio=split_ratio,
                )
                tasks.append(task)

        return tasks


class Benchmark:
    """Normalize benchmark plans and execute the resulting task list."""

    @staticmethod
    def _normalize_tasks(cfg: Dict[str, Any]) -> List[Task]:
        """Convert benchmark configuration into normalized task list."""
        plan = cfg.get("plan")

        if not plan:
            raise ValueError("Benchmark without a 'plan'")

        builder = ConfigBuilder(cfg)
        factory = TaskFactory(builder)
        all_tasks = []

        for job in plan:
            mode = job.get("mode", "train_cv")
            JobValidator.validate_job(job, mode)

            if mode == "eval":
                tasks = factory.create_eval_tasks(job)
            else:
                tasks = factory.create_training_tasks(job)

            all_tasks.extend(tasks)

        return Benchmark._sort_tasks(all_tasks)

    @staticmethod
    def _sort_tasks(tasks: List[Task]) -> List[Task]:
        """Sort tasks by priority: train before eval, base models first."""
        return sorted(
            tasks,
            key=lambda t: (
                0 if t.mode == "train_cv" or t.mode == "train_single" else 1,
                1 if t.base_model_config is not None else 0,
                t.config.dataset_name,
                t.config.model_name,
            ),
        )

    @staticmethod
    def execute(cfg: Dict[str, Any]) -> pd.DataFrame:
        """Execute benchmark from configuration."""
        root = Path(cfg["artifacts_root"])
        root.mkdir(parents=True, exist_ok=True)

        ProgressReporter.set_logfile(root / "benchmark_run.log")
        logger = ProgressReporter()

        logger.log("--- Pipeline Started ---")

        tasks = Benchmark._normalize_tasks(cfg=cfg)
        results = Benchmark._execute_tasks(logger, root, tasks)

        return Benchmark._build_results_dataframe(logger, root, len(tasks), results)

    @staticmethod
    def _execute_tasks(logger, root: Path, tasks: List[Task]) -> List[BenchmarkResult]:
        """Execute all tasks and collect results."""
        results = []

        for i, task in enumerate(tasks, 1):
            Benchmark._log_task_info(logger, i, len(tasks), task)

            res = run_task(root, task)
            if res:
                results.append(res)
                logger.log(f"Task {i}/{len(tasks)} completed", level="info")
            else:
                logger.log(f"Task {i}/{len(tasks)} failed or skipped", level="warning")

        return results

    @staticmethod
    def _log_task_info(logger, current: int, total: int, task: Task) -> None:
        """Log information about the current task."""
        logger.log(f"─" * 80, level="info")
        logger.log(f"   Task {current}/{total}", level="info")
        logger.log(f"   Dataset: {task.config.dataset_name}", level="info")
        logger.log(f"   Processor: {task.config.processor_name}", level="info")
        logger.log(f"   Builder: {task.config.builder_name}", level="info")
        logger.log(f"   Model: {task.config.model_name}", level="info")

        if task.optimization:
            logger.log(f"   Optimization: {task.optimization}", level="info")
        logger.log(f"   Mode: {task.mode}", level="info")
        logger.log(f"─" * 80, level="info")

    @staticmethod
    def _build_results_dataframe(
        logger, root: Path, total_task: int, results: List[BenchmarkResult]
    ) -> pd.DataFrame:
        """Build and save results dataframe."""
        logger.log("", level="info")
        logger.log("=" * 80, level="info")
        logger.log("PIPELINE COMPLETED", level="info")
        logger.log(f"Successful tasks: {len(results)}/{total_task}", level="info")
        logger.log("=" * 80, level="info")

        if not results:
            logger.log("No results to report", level="warning")
            return pd.DataFrame()

        rows = []
        for r in results:
            row = {
                "Model": r.exp_name,
                "Dataset": r.dataset_name,
                "Mode": r.mode,
            }

            for metric, mean_value in r.summary_mean.items():
                row[metric] = mean_value
                if metric in r.summary_std:
                    row[f"{metric}_std"] = r.summary_std[metric]

            for metric, std_value in r.summary_std.items():
                row.setdefault(f"{metric}_std", std_value)

            row["model_size (KB)"] = r.model_size
            rows.append(row)

        df = pd.DataFrame(rows)

        if "Dataset" in df.columns and "Model" in df.columns:
            df = df.sort_values(by=["Dataset", "Model"])

        results_path = root / "results.csv"
        df.to_csv(results_path, index=False)
        logger.log(f"Results saved to: {results_path}", level="info")

        return df


class ArtifactManager:
    """Central class for managing caches, hashes, and file paths."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.logger = ProgressReporter()

    def _clean_step(self, step: Optional[Dict[str, Any]]) -> dict:
        """Filters out transient settings that shouldn't alter the hash (e.g., log_level)."""
        if not step:
            return {}
        cfg = step.get("config", {})
        return {
            "type": step.get("type"),
            "config": {k: v for k, v in cfg.items() if k not in TRANSIENT_CONFIG_KEYS},
        }

    def _get_snapshot(self, config: PipelineConfig, level: str) -> Dict[str, Any]:
        """Builds the configuration snapshot for a specific pipeline stage."""
        levels = {"raw": 1, "processed": 2, "built": 3, "model": 4}
        target = levels.get(level, 0)

        snap = {}
        if target >= 1:
            snap["datasets"] = {
                config.dataset_name: self._clean_step(config.dataset_cfg)
            }
        if target >= 2:
            snap["processors"] = {
                config.processor_name: self._clean_step(config.processor_cfg)
            }
        if target >= 3:
            snap["builders"] = {
                config.builder_name: self._clean_step(config.builder_cfg)
            }
        if target >= 4:
            snap["models"] = {config.model_name: self._clean_step(config.model_cfg)}
        return snap

    def get_dir(self, task: Task, level: str) -> Tuple[Path, Dict[str, Any]]:
        """Returns the exact cache directory and the hash payload for a given task."""
        config = task.config
        snapshot = self._get_snapshot(config, level)

        # 1. Model directory (requires mode and split_ratio for the hash)
        if level == "model":
            hash_input = {
                "config": snapshot,
                "runtime": {
                    "mode": task.mode,
                    "split_ratio": getattr(task, "split_ratio", None),
                    "src": (
                        task.base_model_config.model_name
                        if task.mode == "eval" and task.base_model_config
                        else None
                    ),
                },
                "trained_on_config": (
                    self._clean_step(config.dataset_cfg)
                    if task.mode == "eval"
                    else None
                ),
            }
            h = stable_hash(hash_input)
            mode_str = (
                f"eval_from_{config.model_name}" if "eval" in task.mode else task.mode
            )

            optimization_name = getattr(task, "optimization", "none")
            if optimization_name is None or str(optimization_name).lower() == "none":
                builder_model_folder = f"{config.builder_name}__{config.model_name}"
            else:
                builder_model_folder = (
                    f"{config.builder_name}_{optimization_name}_{config.model_name}"
                )

            p = (
                self.root
                / "runs"
                / config.dataset_name
                / config.processor_name
                / builder_model_folder
                / mode_str
                / h
            )
            return p, hash_input

        # 2. Data Pipeline directories (raw, processed, built)
        h = stable_hash(snapshot)
        if level == "raw":
            p = self.root / level / config.dataset_name / h
        elif level == "processed":
            p = self.root / level / config.dataset_name / config.processor_name / h
        elif level == "built":
            p = (
                self.root
                / level
                / config.dataset_name
                / config.processor_name
                / config.builder_name
                / h
            )
        else:
            raise ValueError(f"Unknown level: {level}")

        return p, snapshot

    def ensure_pipeline(self, task: Task) -> Optional[Path]:
        """Automatically builds the data pipeline and returns the path to dataset.h5."""
        built_dir, _ = self.get_dir(task, "built")
        built_path = built_dir / "dataset.h5"
        built_success = built_dir / SUCCESS_FILE_ENDING

        force = bool(
            task.config.builder_cfg.get("force")
            or task.config.builder_cfg.get("force_overwrite")
        )
        if built_success.exists() and not force:
            self.logger.log(
                f"[BUILT] Cached: {built_path.relative_to(self.root)}", level="info"
            )
            return built_path

        # Define all stages of the pipeline
        stages = [
            (
                "raw",
                task.config.dataset_name,
                "raw.h5",
                get_extractor,
                task.config.dataset_cfg,
            ),
            (
                "processed",
                task.config.processor_name,
                "processed.h5",
                get_processor,
                task.config.processor_cfg,
            ),
            (
                "built",
                task.config.builder_name,
                "dataset.h5",
                get_builder,
                task.config.builder_cfg,
            ),
        ]

        current_input = None
        for lvl, name, filename, factory, cfg in stages:
            target_dir, snap = self.get_dir(task, lvl)
            target_path = target_dir / filename
            success_marker = target_dir / SUCCESS_FILE_ENDING

            step_force = bool(cfg.get("force") or cfg.get("force_overwrite"))
            if step_force or not success_marker.exists():

                if target_path.exists():
                    self.logger.log(
                        f"[{lvl.upper()}] Found incomplete file from previous crash, removing: {target_path.name}",
                        level="warning",
                    )
                    target_path.unlink()

                ensure_parent(target_path)
                write_json(target_dir / "config_used.json", snap)

                kwargs = {"name": name, "out_path": target_path}
                if current_input:
                    kwargs["in_path"] = current_input

                call_run_best_effort(factory(cfg), **kwargs)

                success_marker.touch()

                self.logger.log(
                    f"[{lvl.upper()}] Created: {target_path.name}", level="info"
                )
            else:
                self.logger.log(
                    f"[{lvl.upper()}] Cached: {target_path.relative_to(self.root)}",
                    level="info",
                )

            current_input = target_path

        return current_input


# ============================================================================
# TASK EXECUTION
# ============================================================================


def _get_base_model_dir(
    manager: ArtifactManager, task: Task, default_mode: str
) -> Optional[Path]:
    """Helper: Creates a dummy task for the base model to retrieve its directory."""
    if not task.base_model_config:
        return None
    base_task = Task(
        config=task.base_model_config,
        mode=getattr(task, "base_model_mode", default_mode),
        split_ratio=getattr(task, "split_ratio", None),
    )
    base_dir, _ = manager.get_dir(base_task, "model")
    return base_dir.resolve()


def run_task_train(
    manager: ArtifactManager,
    task: Task,
    builder_path: Path,
    run_dir: Path,
    *,
    force: bool = False,
) -> None:
    logger = ProgressReporter()
    final_params = task.config.model_cfg.get("config", {}).get("params", {}).copy()

    warm_model_run_dir = _get_base_model_dir(manager, task, task.mode)

    if warm_model_run_dir and task.base_model_config:
        if not warm_model_run_dir.exists():
            raise ValueError(f"Base model not found: {warm_model_run_dir}")
        else:
            final_params.update(
                task.base_model_config.model_cfg.get("config", {}).get("params", {})
            )

    model_config = {
        "type": task.config.model_cfg["type"],
        "config": {
            **task.config.model_cfg.get("config", {}),
            "params": final_params,
            "artifact_dir": str(run_dir),
            "warm_start_dir": str(warm_model_run_dir) if warm_model_run_dir else None,
            "force_retrain": force,
        },
    }

    model_runner: BaseModelRunner = get_model(model_config)

    model_runner.run(
        mode=task.mode,
        name=task.config.model_name,
        data_path=builder_path,
        split_kwargs=getattr(task, "split_ratio", None),
    )


def run_task_eval(
    manager: ArtifactManager, task: Task, builder_path: Path, run_dir: Path
) -> None:
    logger = ProgressReporter()
    model_path = None

    if task.eval_model_path:
        model_path = Path(task.eval_model_path).resolve()
        logger.log(f"[EVAL] Using manual path: {model_path}", level="info")
    elif task.base_model_config:
        model_run_dir = _get_base_model_dir(manager, task, "train_cv")

        if model_run_dir:
            model_file_name = get_model_filename(
                task.base_model_config.model_cfg["type"]
            )
            model_path = (model_run_dir / model_file_name).resolve()
        else:
            logger.log(
                "[EVAL] Base model directory could not be resolved.", level="warning"
            )
            return
    else:
        logger.log("[EVAL] Requires model_path or eval_model_config", level="warning")
        return

    if model_path and model_path.exists() and task.base_model_config:
        model_config = {
            "type": task.config.model_cfg["type"],
            "config": {
                **task.config.model_cfg.get("config", {}),
                "artifact_dir": str(run_dir),
            },
        }
        model_runner: BaseModelRunner = get_model(model_config)
        model_runner.run(
            mode="eval",
            name=task.base_model_config.model_name,
            data_path=builder_path,
            model_path=model_path,
        )
    else:
        logger.log(f"[EVAL] Base model not found at {model_path}", level="warning")


def run_task(root: Path, task: Task) -> Optional[BenchmarkResult]:
    logger = ProgressReporter()
    manager = ArtifactManager(root)

    try:
        # Secure the entire pipeline at once
        builder_path = manager.ensure_pipeline(task)

        if builder_path is None:
            logger.log(
                f"Pipeline creation failed for task {task.config.model_name}. Aborting.",
                level="error",
            )
            return None

        # Retrieve the specific directory and hash payload for the model
        run_dir, model_hash_payload = manager.get_dir(task, "model")

        ensure_parent(run_dir / "placeholder")
        write_json(run_dir / "config_used.json", model_hash_payload)

        results_file = run_dir / RESULTS_FILE_NAME
        success_file = run_dir / SUCCESS_FILE_ENDING

        force = task.force_train or bool(task.config.model_cfg.get("force_overwrite"))

        if force or not success_file.exists():

            if results_file.exists() and not success_file.exists():
                logger.log(
                    f"[MODEL] Found results.pkl without .success in "
                    f"{run_dir.name} - previous run likely crashed. "
                    f"Resuming from checkpoints (or retraining if force).",
                    level="warning",
                )

            if task.mode.startswith("train_"):
                run_task_train(manager, task, builder_path, run_dir, force=force)
            elif task.mode == "eval":
                run_task_eval(manager, task, builder_path, run_dir)
            else:
                logger.log(f"Unknown task mode: {task.mode}", level="error")
                return None

            if results_file.exists():
                success_file.touch()

        else:
            rel_path = results_file.relative_to(manager.root)
            logger.log(
                f"[{task.mode.upper()}] Cached: {rel_path} (skipping execution)",
                level="info",
            )
        if results_file.exists():
            mean, std = get_summary_mean_std(results_file)
            return BenchmarkResult(
                exp_name=task.config.model_name,
                dataset_name=task.config.dataset_name,
                mode=task.mode,
                run_dir=run_dir,
                summary_mean=mean,
                summary_std=std,
                model_size=get_model_size(results_pkl=results_file),
            )

        return None

    except Exception as e:
        logger.handle_exception(
            e,
            context=f"Task {task.config.model_name} on {task.config.dataset_name} ({task.mode})",
            show_traceback=True,
        )
        return None
