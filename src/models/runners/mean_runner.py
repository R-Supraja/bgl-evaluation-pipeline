from pathlib import Path

import h5py

from src.models.core.mean_model import MeanModel
from src.models.dataset.loader import SequenceHDF5Loader, TabularHDF5Loader
from src.models.runners.base_runner import BaseModelRunner
from src.models.training.mean_trainer import MeanTrainer
from src.models.training.base_trainer import TrainingConfig
from src.utils import get_group


class MeanModelRunner(BaseModelRunner):
    """Runner for the constant-mean baseline across tabular or sequence artifacts."""

    def _make_model(self):
        return MeanModel(self.config)

    def _make_dataset(self, data_path: Path):
        """Infer artifact type from HDF5 keys and load with the matching loader."""
        with h5py.File(data_path, "r") as f:
            for key in f.keys():
                grp = get_group(f, key, verbose=False)
                if grp is None:
                    continue
                if "x_features" in grp:
                    return TabularHDF5Loader().load(data_path)
                if "x_sequences" in grp:
                    return SequenceHDF5Loader().load(data_path)

        raise RuntimeError(f"Could not determine dataset type for {data_path}")

    def _make_trainer(self, model, train_cfg: TrainingConfig):
        """Use the lightweight trainer that estimates and stores global mean."""
        return MeanTrainer(model, train_cfg)
