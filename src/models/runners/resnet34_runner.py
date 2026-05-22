from pathlib import Path

from src.models.core.model import Model
from src.models.core.resnet34_model import ResNet34Model
from src.models.dataset.loader import SequenceHDF5Loader
from src.models.optimization.optimization_strategy import OptimizerFactory
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.keras_optimization_trainer import KerasOptimizationTrainer
from src.models.training.keras_trainer import KerasTrainer
from src.models.training.base_trainer import Trainer, TrainingConfig
from src.models.tuning.keras_tuner import KerasTuner


class ResNet34ModelRunner(BaseModelRunner):
    """Runner for ResNet34 sequence modeling with optional optimization/tuning."""

    def _make_model(self) -> Model:
        return ResNet34Model(self.config)

    def _make_dataset(self, data_path: Path):
        """Load beat/window sequence tensors from HDF5 artifact."""
        loader = SequenceHDF5Loader()
        return loader.load(data_path)

    def _make_trainer(self, model: Model, train_cfg: TrainingConfig) -> Trainer:
        """Choose optimization-aware or standard Keras trainer based on params."""
        quantization = self.params.get("quantization")

        if quantization:
            strategy = OptimizerFactory.create(self.params)

            return KerasOptimizationTrainer(model, train_cfg, strategy=strategy)

        else:
            tuner = None
            if self.params.get("use_tuner", True):
                tuner = KerasTuner(config=self.params, model=model)

        return KerasTrainer(model, train_cfg, tuner=tuner)
