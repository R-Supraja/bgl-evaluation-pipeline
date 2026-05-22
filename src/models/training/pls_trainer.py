from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

from src.models.dataset.base import FoldData
from src.models.training.base_trainer import Trainer, TrainingConfig, TrainingResult


class PLSTrainer(Trainer):
    """Trainer for Partial Least Squares regression models."""

    def train_fold(self, fold_data: FoldData, fold_idx: int):
        """Fits one PLS model on the provided fold and stores artifacts.

        Args:
            fold_data: Train/validation/test split for this fold.
            fold_idx: Fold index used in artifact naming.

        Returns:
            Tuple of trained model wrapper and fold ``TrainingResult``.
        """
        n_components = self.model.params.get("n_components", 2)
        scale = bool(self.model.params.get("scale", True))

        pls = PLSRegression(n_components=n_components, scale=scale)
        pls.fit(fold_data.train.X, fold_data.train.y)

        self.model._model = pls

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
