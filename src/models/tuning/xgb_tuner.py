from typing import Any, Dict

import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from src.models.dataset.base import FoldData
from src.models.tuning.tuner_base import BaseTuner


class XGBoostOptunaTuner(BaseTuner):
    """Optuna tuner for XGBoost tabular regression models."""

    def tune(self, fold_data: FoldData, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search tree/regularization hyperparameters minimizing validation RMSE."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        dtrain = xgb.DMatrix(fold_data.train.X, label=fold_data.train.y)
        dval = xgb.DMatrix(fold_data.val.X, label=fold_data.val.y)

        def objective(trial: optuna.Trial) -> float:
            # 1. Define the search space
            params = {
                "objective": "reg:squarederror",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0, 5),
            }
            # Merge with fixed parameters
            full_params = {**base_params, **params}

            # 2. Train a quick model for evaluation
            evals_result = {}
            xgb.train(
                full_params,
                dtrain,
                num_boost_round=100,
                evals=[(dval, "val")],
                evals_result=evals_result,
                early_stopping_rounds=10,
                verbose_eval=False,
                callbacks=(
                    [
                        optuna.integration.XGBoostPruningCallback(
                            trial, observation_key=("val-rmse")
                        )
                    ]
                    if optuna.__version__ >= "2.0"
                    else []
                ),
            )

            # 3. Minimize the validation RMSE
            return float(evals_result["val"]["rmse"][-1])

        # Create and run the Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed, multivariate=True),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        )
        study.optimize(objective, n_trials=self.n_trials)

        return study.best_params
