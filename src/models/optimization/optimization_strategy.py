import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, cast

import tensorflow as tf

try:
    import tf_keras as keras
    from tf_keras import Model as KerasModel
    from tf_keras import layers, metrics
    from tf_keras import models as models_lib
    from tf_keras import optimizers
except Exception:
    import tensorflow as tf

    keras = tf.keras
    from tensorflow.keras import layers, optimizers, metrics
    from tensorflow.keras import models as models_lib
    from tensorflow.keras import Model as KerasModel

from src.models.tf_architectures import Conv1DQuantizeConfig, NoOpQuantizeConfig
from src.progressbar import ProgressReporter

# Layers that should be explicitly ignored during QAT annotation to prevent crashes
NOOP_LAYERS = (
    layers.MaxPooling1D,
    layers.AveragePooling1D,
    layers.GlobalAveragePooling1D,
    layers.GlobalMaxPooling1D,
    layers.BatchNormalization,
    layers.Dropout,
    layers.Flatten,
    layers.Reshape,
    layers.Add,
    layers.ReLU,
    layers.Activation,
    layers.InputLayer,
)


class OptimizationStrategy(ABC):
    """Abstract base class for post-training optimization strategies."""

    @abstractmethod
    def optimize(
        self, model: KerasModel, train_fn: Callable, num_samples: int, batch_size: int
    ) -> KerasModel:
        """
        Executes the architecture transformation and fine-tuning.

        Args:
            model: The base Keras model to optimize.
            train_fn: A callback function provided by the trainer to handle
                      the actual model.fit() execution. Signature should be:
                      train_fn(model_to_train, epochs, callbacks)
        """
        pass


class PruningStrategy(OptimizationStrategy):
    """Applies Magnitude-based Pruning to the model."""

    def __init__(
        self,
        pruning_epochs: int,
        target_sparsity: float,
        lr: float,
    ):
        self.pruning_epochs = pruning_epochs
        self.target_sparsity = target_sparsity
        self.lr = lr

    def optimize(
        self, model: KerasModel, train_fn: Callable, num_samples: int, batch_size: int
    ) -> KerasModel:
        import math

        import tensorflow_model_optimization as tfmot

        # NOTE: In Keras, pruning updates happen after every batch (step), not epoch.
        # To calculate this curve from 0% to 'target_sparsity', the algorithm
        # MUST know the TOTAL number of training iterations ('end_step').
        # We find this by dividing the total samples by the batch size.

        steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
        end_step = steps_per_epoch * self.pruning_epochs

        schedule = tfmot.sparsity.keras.PolynomialDecay(
            0.0, self.target_sparsity, 0, end_step, frequency=100
        )

        # 1. Wrap model with pruning layers
        pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            model, pruning_schedule=cast(Any, schedule)
        )

        if isinstance(pruned_model, list):
            raise TypeError(
                "Expected 'model' to be a Keras Model instance, but got a list instead."
            )

        pruned_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr),
            loss="mse",
            metrics=[metrics.RootMeanSquaredError(name="rmse")],
        )

        # 2. Delegate fine-tuning to the trainer, injecting the required callback
        train_fn(
            pruned_model,
            epochs=self.pruning_epochs,
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        )

        # 3. Strip wrappers before returning
        return tfmot.sparsity.keras.strip_pruning(pruned_model)


class QATStrategy(OptimizationStrategy):
    """Applies Quantization-Aware Training (Fake Quantization) to the model."""

    def __init__(self, qat_epochs: int, lr: float):
        self.qat_epochs = qat_epochs
        self.lr = lr

    def _annotate(self, model: KerasModel) -> KerasModel:
        """Annotates layers with specific quantization configs."""
        import tensorflow_model_optimization as tfmot

        def fn(layer):
            if isinstance(layer, layers.Conv1D):
                return tfmot.quantization.keras.quantize_annotate_layer(
                    layer, quantize_config=Conv1DQuantizeConfig()
                )
            elif isinstance(layer, (layers.Dense, layers.Conv2D)):
                return tfmot.quantization.keras.quantize_annotate_layer(layer)
            elif isinstance(layer, NOOP_LAYERS):
                return tfmot.quantization.keras.quantize_annotate_layer(
                    layer, quantize_config=NoOpQuantizeConfig()
                )
            return layer

        return models_lib.clone_model(model, clone_function=fn)

    def optimize(
        self, model: KerasModel, train_fn: Callable, num_samples: int, batch_size: int
    ) -> KerasModel:
        import tensorflow_model_optimization as tfmot

        # 1. Annotate layers
        annotated_model = self._annotate(model)

        # 2. Apply Quantization Scope
        with tfmot.quantization.keras.quantize_scope(
            {
                "Conv1DQuantizeConfig": Conv1DQuantizeConfig,
                "NoOpQuantizeConfig": NoOpQuantizeConfig,
            }
        ):
            qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

        # Compile with reduced learning rate for QAT fine-tuning
        qat_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.lr * 0.1),
            loss="mse",
            metrics=[metrics.RootMeanSquaredError(name="rmse")],
        )

        # 3. Delegate QAT fine-tuning to the trainer
        train_fn(qat_model, epochs=self.qat_epochs, callbacks=[])

        return qat_model


class PQATStrategy(OptimizationStrategy):
    """
    Composite Strategy: Applies Pruning, strips it, rebuilds the graph cleanly,
    and then applies QAT.
    """

    def __init__(
        self,
        pruning_epochs: int,
        qat_epochs: int,
        target_sparsity: float,
        lr: float,
    ):
        # Instantiate the two individual strategies
        self.pruning_strategy = PruningStrategy(pruning_epochs, target_sparsity, lr)
        self.qat_strategy = QATStrategy(qat_epochs, lr)

    def optimize(
        self, model: KerasModel, train_fn: Callable, num_samples: int, batch_size: int
    ) -> KerasModel:
        # Phase 1: Pruning
        pruned_stripped_model = self.pruning_strategy.optimize(
            model, train_fn, num_samples, batch_size
        )

        # Phase 1.5: Clean the Graph
        # Rebuild a clean graph from stripped config to avoid wrapper
        # leftovers before QAT application.
        model_config = pruned_stripped_model.get_config()
        clean_model = KerasModel.from_config(model_config)
        clean_model.set_weights(pruned_stripped_model.get_weights())

        # Phase 2: QAT
        final_pqat_model = self.qat_strategy.optimize(
            clean_model, train_fn, num_samples, batch_size
        )

        return final_pqat_model


class OptimizerFactory:
    """Factory to instantiate the correct optimization strategy based on configuration parameters."""

    @staticmethod
    def create(params: Dict[str, Any]) -> OptimizationStrategy:
        """Build optimization strategy from `params['quantization']` mode."""
        mode = params.get("quantization")
        lr = params.get("learning_rate", 1e-3)
        pruning_epochs = params.get("pruning_epochs", 10)
        qat_epochs = params.get("qat_epochs", 10)
        target_sparsity = params.get("target_sparsity", 0.5)

        logger = ProgressReporter()

        if mode == "qat":
            # Check for parameters that QAT doesn't use
            unused = [k for k in ["pruning_epochs", "target_sparsity"] if k in params]
            if unused:
                logger.log(
                    f"Unused parameters for 'qat' mode: {', '.join(unused)}",
                    level="warning",
                )
            return QATStrategy(qat_epochs=qat_epochs, lr=lr)

        elif mode == "pruning":
            # Check for parameters that Pruning doesn't use
            unused = [k for k in ["qat_epochs"] if k in params]
            if unused:
                logger.log(
                    f"Unused parameters for 'pruning' mode: {', '.join(unused)}",
                    level="warning",
                )
            return PruningStrategy(
                pruning_epochs=pruning_epochs,
                target_sparsity=target_sparsity,
                lr=lr,
            )

        elif mode == "pqat":
            # PQAT uses all parameters, so no warnings are needed here
            return PQATStrategy(
                pruning_epochs=pruning_epochs,
                qat_epochs=qat_epochs,
                target_sparsity=target_sparsity,
                lr=lr,
            )

        raise ValueError(f"Unknown optimization mode: {mode}")
