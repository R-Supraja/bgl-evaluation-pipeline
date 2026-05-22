import gc
import json
from pathlib import Path

import numpy as np

try:
    import tf_keras as keras
    from tf_keras import Model, backend, callbacks, optimizers
except Exception:  # Fallback for environments without tf_keras
    import tensorflow as tf

    keras = tf.keras
    from tensorflow.keras import backend
    from tensorflow.keras import Model

from src.models.core.keras_base import KerasBaseModel
from src.models.core.model import Model
from src.models.dataset.base import FoldData
from src.models.training.base_trainer import Trainer, TrainingConfig, TrainingResult
from src.progressbar import ProgressReporter


class KerasOptimizationTrainer(Trainer):
    def __init__(self, model, config: TrainingConfig, strategy):
        super().__init__(model, config, tuner=None)
        self.strategy = strategy

    def train_fold(self, fold_data: FoldData, fold_idx: int):
        if not isinstance(self.model, KerasBaseModel):
            raise TypeError(f"Trainer requires {KerasBaseModel.__name__}.")

        original_base_params = self.model.params.copy()

        # 1. Reset state
        backend.clear_session()
        gc.collect()
        if self.config.seed is not None:
            keras.utils.set_random_seed(self.config.seed * 1000 + fold_idx)

        # 2. Warm Start: Load Base Model and Base Params
        if not self.config.init_from_dir:
            raise ValueError(
                f"{self.__class__.__name__} requires an init_from_dir path!"
            )

        base_path = Path(self.config.init_from_dir)
        keras_model_file = base_path / "folds" / f"fold_{fold_idx}.keras"
        params_file = base_path / "folds" / f"fold_{fold_idx}_params.json"

        if not self.model.load(keras_model_file):
            raise FileNotFoundError(f"Base model not found: {keras_model_file}")

        # Load the tuned params from the base run so they are carried over
        current_fold_params = original_base_params.copy()
        if params_file.exists():
            with open(params_file, "r", encoding="utf-8") as f:
                base_run_params = json.load(f)
                if "batch_size" in base_run_params:
                    current_fold_params["batch_size"] = base_run_params["batch_size"]

        self.model.params = current_fold_params
        keras_model = self.model.get_native_model()

        # 3. Define Fine-Tuning Execution Function
        batch_size = self.model.params.get("batch_size", self.config.batch_size)

        def fine_tune_fn(model_to_train, epochs, callbacks=None):
            return model_to_train.fit(
                fold_data.train.X,
                fold_data.train.y,
                validation_data=(fold_data.val.X, fold_data.val.y),
                epochs=epochs,
                batch_size=batch_size,
                verbose=self.config.fit_verbose,
                callbacks=callbacks or [],
            )

        # 4. Apply Optimization Strategy
        logger = ProgressReporter()
        logger.log(f"Fold {fold_idx}: Applying optimization strategy...", level="info")
        num_samples = len(fold_data.train.X)
        optimized_keras_model = self.strategy.optimize(
            model=keras_model,
            train_fn=fine_tune_fn,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        self.model._model = optimized_keras_model

        # 5. Evaluate Val RMSE
        y_val_pred = self.model.predict(fold_data.val.X)
        val_rmse = float(
            np.sqrt(np.mean((fold_data.val.y.flatten() - y_val_pred) ** 2))
        )

        # 6. Save Artifacts
        artifact_root = Path(self.model.config["artifact_dir"])
        fold_path = artifact_root / "folds" / f"fold_{fold_idx}"
        fold_path.parent.mkdir(parents=True, exist_ok=True)

        # A. Save Keras model
        self.model.save(fold_path)

        # B. Save model params as JSON
        params_path = fold_path.with_name(f"{fold_path.name}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(current_fold_params, f, indent=4, sort_keys=True)

        # C. Export INT8 TFLite
        model_size_kb = 0.0
        if not bool(self.model.config.get("skip_tflite_export", False)):
            int8_tflite_path = fold_path.with_name(fold_path.name + "_int8")

            # Using validation data as representative data for INT8 calibration
            model_size_kb = self.model.export_to_tflite(
                int8_tflite_path,
                representative_data=fold_data.val.X[:100],
            )

        # 7. Cleanup & Restore
        backend.clear_session()
        gc.collect()
        self.model.params = original_base_params

        return self.model, TrainingResult(
            val_rmse=val_rmse,
            history={},
            model_size_kb=model_size_kb,
        )
