from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import tf_keras as keras
    from tf_keras import Model, backend, callbacks
except Exception:  # Fallback for environments without tf_keras
    import tensorflow as tf

    keras = tf.keras
    from tensorflow.keras import backend, callbacks
    from tensorflow.keras import Model

from src.models.core.keras_base import KerasBaseModel
from src.models.dataset.base import FoldData
from src.models.training.base_trainer import Trainer, TrainingConfig, TrainingResult
from src.models.tuning.tuner_base import BaseTuner
from src.progressbar import ProgressReporter


class KerasTrainer(Trainer):
    def __init__(
        self,
        model,
        config: TrainingConfig,
        tuner: Optional[BaseTuner] = None,
    ):
        super().__init__(model, config, tuner)

    def train_fold(
        self,
        fold_data: FoldData,
        fold_idx: int,
    ):
        """Trains one fold and exports fold artifacts.

        Args:
            fold_data: Train/validation/test split for this fold.
            fold_idx: Fold index used for deterministic seeding and naming.

        Returns:
            Tuple of trained model wrapper and fold ``TrainingResult``.

        Raises:
            TypeError: If the bound model is not a ``KerasBaseModel``.
            FileNotFoundError: If warm-start loading is configured but the
                expected artifact cannot be found.
        """
        if not isinstance(self.model, KerasBaseModel):
            raise TypeError("KerasTrainer requires KerasBaseModel.")

        original_base_params = self.model.params.copy()
        current_fold_params = original_base_params.copy()

        # Reset graph/session state between folds.
        backend.clear_session()
        gc.collect()
        if self.config.seed is not None:
            keras.utils.set_random_seed(self.config.seed * 1000 + fold_idx)

        # ---------- Hyperparameter Tuning ----------
        if self.tuner is not None and not self.config.init_from_dir:
            best_params = self.tuner.tune(fold_data, original_base_params)

            current_fold_params = {**original_base_params, **best_params}
            self.model.params = current_fold_params

            logger = ProgressReporter()
            logger.log(f"Fold {fold_idx} best DL params: {best_params}", level="info")

        # ---------- Build & Compile ----------
        self.model._model = None
        input_shape = fold_data.train.X.shape[1:]
        self.model.build(input_shape)
        keras_model: Model = self.model.get_native_model()

        lr = self.model.params.get("learning_rate", self.config.learning_rate)
        batch_size = self.model.params.get("batch_size", self.config.batch_size)

        optimizer = keras.optimizers.Adam(learning_rate=lr)

        keras_model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        # ---------- Fit ----------
        history: Any = keras_model.fit(
            fold_data.train.X,
            fold_data.train.y,
            validation_data=(fold_data.val.X, fold_data.val.y),
            epochs=self.config.epochs,
            batch_size=batch_size,
            verbose=self.config.fit_verbose,  # type: ignore
            callbacks=[
                callbacks.EarlyStopping(
                    monitor="val_rmse",
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True,
                ),
            ],
        )

        # ---------- Val RMSE ----------
        y_val_pred = self.model.predict(fold_data.val.X)
        val_rmse = float(
            np.sqrt(np.mean((fold_data.val.y.flatten() - y_val_pred) ** 2))
        )

        artifact_root = Path(self.model.config["artifact_dir"])
        fold_path = artifact_root / "folds" / f"fold_{fold_idx}"
        fold_path.parent.mkdir(parents=True, exist_ok=True)

        logger = ProgressReporter()

        logger.log(f"Saving Keras model to {fold_path}...", level="info")
        # A. save model
        self.model.save(fold_path)

        # B. Save model params as JSON
        params_path = fold_path.with_name(f"{fold_path.name}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(current_fold_params, f, indent=4, sort_keys=True)

        model_size_kb = 0.0
        if not bool(self.model.config.get("skip_tflite_export", False)):
            float_tflite_path = fold_path.with_name(fold_path.name + "_float")
            logger.log("Exporting model to TFLite format...", level="info")

            model_size_kb = self.model.export_to_tflite(
                float_tflite_path,
                representative_data=None,
            )
            logger.log(
                f"TFLite export finished. Size: {model_size_kb:.2f} KB", level="info"
            )

        logger.log(
            "Cleaning up TensorFlow backend session and triggering garbage collection...",
            level="debug",
        )
        # Cleanup
        backend.clear_session()
        gc.collect()
        history_data = history.history if hasattr(history, "history") else history
        # restore input base params
        self.model.params = original_base_params

        logger.log(
            f"Fold processing complete. Final Validation RMSE: {val_rmse:.6f}",
            level="info",
        )
        return self.model, TrainingResult(
            val_rmse=val_rmse,
            history=history_data,
            model_size_kb=model_size_kb,
        )
