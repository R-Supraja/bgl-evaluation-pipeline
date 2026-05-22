from pathlib import Path

from src.models.optimization.optimization_strategy import OptimizerFactory
from src.models.core.model import Model
from src.models.core.tcn_model import TCNModel
from src.models.dataset.loader import SequenceHDF5Loader
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.keras_optimization_trainer import KerasOptimizationTrainer
from src.models.training.keras_trainer import KerasTrainer
from src.models.training.base_trainer import Trainer, TrainingConfig


class TCNModelRunner(BaseModelRunner):
    """Runner for TCN sequence regression with optional optimization strategy."""

    def _make_model(self) -> Model:
        return TCNModel(self.config)

    def _make_dataset(self, data_path: Path):
        """Load sequence data for temporal convolutional model training."""
        loader = SequenceHDF5Loader()
        return loader.load(data_path)

    def _make_trainer(self, model: Model, train_cfg: TrainingConfig) -> Trainer:
        """Use optimization-aware trainer when quantization is requested."""
        quantization = self.params.get("quantization")

        if quantization:
            strategy = OptimizerFactory.create(self.params)

            return KerasOptimizationTrainer(model, train_cfg, strategy=strategy)

        return KerasTrainer(model, train_cfg)
