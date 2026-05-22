from typing import Any, Dict

import optuna
import tensorflow as tf

try:
    import tf_keras as keras
    from tf_keras import backend, callbacks
except Exception:
    import tensorflow as tf

    keras = tf.keras
    from tensorflow.keras import backend, callbacks

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.models.dataset.base import FoldData
from src.models.tuning.tuner_base import BaseTuner


class KerasTuner(BaseTuner):
    """Optuna tuner for Keras-based sequence models."""

    def __init__(self, config: Dict[str, Any], model):
        super().__init__(config)
        self.model = model
        self.tune_epochs = config.get("tune_epochs", 20)

    def tune(self, fold_data: FoldData, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize learning rate and batch size against fold validation RMSE."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        train_X, train_y = fold_data.train.X, fold_data.train.y
        val_X, val_y = fold_data.val.X, fold_data.val.y
        input_shape = train_X.shape[1:]

        def objective(trial: optuna.Trial) -> float:
            try:
                # 1. Suggest hyperparameters
                params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-2, log=True
                    ),
                    "batch_size": trial.suggest_categorical(
                        "batch_size", [32, 64, 128, 256]
                    ),
                }
                full_params = {**base_params, **params}

                # 2. Build model
                self.model.params = full_params
                keras_model = self.model.build(input_shape=input_shape)

                # 3. Compile
                keras_model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=full_params["learning_rate"]
                    ),
                    loss="mse",
                    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
                )

                # 4. Train with callbacks
                history = keras_model.fit(
                    train_X,
                    train_y,
                    validation_data=(val_X, val_y),
                    epochs=self.tune_epochs,
                    batch_size=full_params["batch_size"],
                    callbacks=[
                        keras.callbacks.EarlyStopping(
                            monitor="val_rmse",
                            patience=3,
                            restore_best_weights=True,
                        ),
                        optuna.integration.TFKerasPruningCallback(
                            trial, monitor="val_rmse"
                        ),
                    ],
                    verbose=0,
                )

                # 5. Return best metric
                return min(history.history["val_rmse"])

            except optuna.TrialPruned:
                raise
            except Exception:
                return float("inf")
            finally:
                backend.clear_session()

        # Create study with pruner
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed, multivariate=True),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params
