from pathlib import Path

from src.models.core.model import Model
from src.models.core.tiny_cnn_model import TinyCNNModel
from src.models.dataset.loader import SequenceHDF5Loader
from src.models.optimization.optimization_strategy import OptimizerFactory
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.keras_optimization_trainer import KerasOptimizationTrainer
from src.models.training.keras_trainer import KerasTrainer
from src.models.training.base_trainer import Trainer, TrainingConfig
from src.models.tuning.keras_tuner import KerasTuner


class TinyCNNModelRunner(BaseModelRunner):
    """Runner for TinyCNN sequence regression with optional tuning/optimization."""

    def _make_model(self):
        return TinyCNNModel(self.config)

    def _make_dataset(self, data_path: Path):
        """Load sequence-form training data from HDF5 artifact."""
        loader = SequenceHDF5Loader()
        return loader.load(data_path)

    def _make_trainer(self, model: Model, train_cfg: TrainingConfig) -> Trainer:
        """Route to quantization path or standard Keras path with optional tuner."""
        quantization = self.params.get("quantization")

        if quantization:
            strategy = OptimizerFactory.create(self.params)

            return KerasOptimizationTrainer(model, train_cfg, strategy=strategy)

        else:
            tuner = None
            if self.params.get("use_tuner", True):
                tuner = KerasTuner(config=self.params, model=model)

        return KerasTrainer(model, train_cfg, tuner=tuner)
