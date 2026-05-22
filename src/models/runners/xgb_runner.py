from pathlib import Path

from src.models.core.xgb_model import XGBModel
from src.models.dataset.filter import FeatureExclusionFilter
from src.models.dataset.loader import TabularHDF5Loader
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.base_trainer import TrainingConfig
from src.models.training.xgb_trainer import XGBoostTrainer
from src.models.tuning.xgb_tuner import XGBoostOptunaTuner


class XGBModelRunner(BaseModelRunner):
    """Runner for XGBoost-based regression on engineered tabular features."""

    def _make_model(self):
        return XGBModel(self.config)

    def _make_dataset(self, data_path: Path):
        """Load tabular artifacts and apply optional feature exclusion."""
        exclude = self.config.get("exclude_features", [])

        loader = TabularHDF5Loader()
        ds = loader.load(data_path)

        exclude = self.config.get("exclude_features")
        if exclude:
            ds = FeatureExclusionFilter(exclude).apply(ds)

        return ds

    def _make_trainer(self, model, train_cfg: TrainingConfig):
        """Attach Optuna tuner when enabled, then build XGBoost trainer."""
        tuner = None
        if self.config.get("use_tuner", True):
            tuner = XGBoostOptunaTuner(config=self.params)

        return XGBoostTrainer(model, train_cfg, tuner=tuner)
