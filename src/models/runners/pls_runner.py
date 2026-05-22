from pathlib import Path

from src.models.core.pls_model import PLSModel
from src.models.dataset.filter import FeatureExclusionFilter
from src.models.dataset.loader import TabularHDF5Loader
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.pls_trainer import PLSTrainer
from src.models.training.base_trainer import TrainingConfig


class PLSModelRunner(BaseModelRunner):
    """Runner for PLS regression on tabular feature artifacts."""

    def _make_model(self):
        return PLSModel(self.config)

    def _make_dataset(self, data_path: Path):
        """Load tabular features and optionally drop configured columns."""
        exclude = self.config.get("exclude_features", [])

        loader = TabularHDF5Loader()
        ds = loader.load(data_path)

        exclude = self.config.get("exclude_features")
        if exclude:
            ds = FeatureExclusionFilter(exclude).apply(ds)

        return ds

    def _make_trainer(self, model, train_cfg: TrainingConfig):
        """Use the PLS-specific trainer implementation."""
        return PLSTrainer(model, train_cfg)
