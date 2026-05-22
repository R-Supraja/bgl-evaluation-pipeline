from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

from src.models.dataset.base import FoldData
from src.models.training.base_trainer import Trainer, TrainingResult


class MeanTrainer(Trainer):
    """Trainer for the constant-mean baseline model."""

    def train_fold(self, fold_data: FoldData, fold_idx: int):
        train_mean = float(np.mean(fold_data.train.y))
        self.model.mean_value = train_mean
        self.model._model = {"mean_value": train_mean}

        y_val_pred = self.model.predict(fold_data.val.X)
        val_rmse = float(np.sqrt(mean_squared_error(fold_data.val.y, y_val_pred)))

        artifact_root = Path(self.model.config["artifact_dir"])
        fold_path = artifact_root / "folds" / f"fold_{fold_idx}"
        self.model.save(fold_path)

        model_size_kb = self.model.get_model_size_kb()

        return self.model, TrainingResult(
            val_rmse=val_rmse,
            history=None,
            model_size_kb=model_size_kb,
        )
