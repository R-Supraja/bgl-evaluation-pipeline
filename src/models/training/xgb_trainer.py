from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.dataset.base import FoldData
from src.models.training.base_trainer import Trainer, TrainingConfig, TrainingResult
from src.progressbar import ProgressReporter


class XGBoostTrainer(Trainer):
    """Trainer for XGBoost-based regression models."""

    def _fit_booster(
        self,
        *,
        train_X: np.ndarray,
        train_y: np.ndarray,
        val_X: np.ndarray,
        val_y: np.ndarray,
    ) -> tuple[xgb.Booster, dict]:
        cfg: TrainingConfig = self.config

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dval = xgb.DMatrix(val_X, label=val_y)

        params = {
            "objective": "reg:squarederror",
            "learning_rate": cfg.learning_rate,
            **self.model.params,
        }
        evals_result = {}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=cfg.epochs,
            evals=[(dval, "val")],
            evals_result=evals_result,
            early_stopping_rounds=cfg.early_stopping_patience,
            verbose_eval=cfg.fit_verbose > 0,
        )
        return booster, evals_result

    @staticmethod
    def _select_top_feature_indices(
        importance_scores: dict[str, float],
        *,
        n_features: int,
        top_k: int,
    ) -> list[int]:
        ranked = [
            (idx, float(importance_scores.get(f"f{idx}", 0.0)))
            for idx in range(n_features)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        chosen = [idx for idx, _ in ranked[: max(1, min(top_k, n_features))]]
        return sorted(chosen)

    def train_fold(
        self,
        fold_data: FoldData,
        fold_idx: int,
    ):
        """Trains one XGBoost fold, and saves artifacts.

        Args:
            fold_data: Train/validation/test split for this fold.
            fold_idx: Fold index used in artifact naming.

        Returns:
            Tuple of trained model wrapper and fold ``TrainingResult``.
        """

        original_base_params = self.model.params.copy()
        current_fold_params = original_base_params.copy()

        top_k = int(self.model.config.get("feature_selection_top_k", 0) or 0)
        importance_type = self.model.config.get(
            "feature_selection_importance_type", "gain"
        )
        feature_names = list(fold_data.feature_names)

        train_X = fold_data.train.X
        val_X = fold_data.val.X

        if top_k > 0 and feature_names:
            selector_bst, _ = self._fit_booster(
                train_X=train_X,
                train_y=fold_data.train.y,
                val_X=val_X,
                val_y=fold_data.val.y,
            )
            selected_indices = self._select_top_feature_indices(
                selector_bst.get_score(importance_type=importance_type),
                n_features=len(feature_names),
                top_k=top_k,
            )
            selected_feature_names = [feature_names[idx] for idx in selected_indices]
            self.model.set_selected_features(selected_indices, selected_feature_names)
            train_X = train_X[:, selected_indices]
            val_X = val_X[:, selected_indices]
        else:
            self.model.set_selected_features()

        if self.tuner:
            # Pass the original, clean parameters to the tuner
            best_params = self.tuner.tune(fold_data, original_base_params)

            # Merge original params with the new best params
            current_fold_params = {**original_base_params, **best_params}

            # Temporarily overwrite the params
            self.model.params = current_fold_params

            logger = ProgressReporter()
            logger.log(f"Fold {fold_idx} best params: {best_params}", level="info")

        bst, evals_result = self._fit_booster(
            train_X=train_X,
            train_y=fold_data.train.y,
            val_X=val_X,
            val_y=fold_data.val.y,
        )

        self.model._model = bst
        # restore original base params
        self.model.params = original_base_params

        val_rmse = float(evals_result["val"]["rmse"][-1])

        artifact_root = Path(self.model.config["artifact_dir"])
        fold_path = artifact_root / "folds" / f"fold_{fold_idx}"
        fold_path.parent.mkdir(parents=True, exist_ok=True)

        # A. save model
        self.model.save(fold_path)

        # B. save model params
        params_path = fold_path.with_name(f"{fold_path.name}_params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(current_fold_params, f, indent=4, sort_keys=True)

        importance_scores = bst.get_score(importance_type=importance_type)
        active_feature_names = self.model.selected_feature_names or feature_names

        data = []
        for i, name in enumerate(active_feature_names):
            data.append(
                {"feature": name, "importance": importance_scores.get(f"f{i}", 0.0)}
            )
        if data:
            pd.DataFrame(data).to_csv(
                fold_path.with_name(f"{fold_path.name}_importance.csv"),
                index=False,
            )

        model_size_kb = self.model.get_model_size_kb()

        return self.model, TrainingResult(
            val_rmse=val_rmse,
            history=evals_result,
            model_size_kb=model_size_kb,
        )
