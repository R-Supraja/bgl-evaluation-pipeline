from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import xgboost as xgb

from src.models.core.model import Model

SAVE_FILE_ENDING = ".json"


class XGBModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config)
        self.selected_feature_indices: list[int] = []
        self.selected_feature_names: list[str] = []

    def build(self, input_shape=None) -> Any:
        """
        XGBoost does not need build().
        Trainer will create and fit the estimator.
        """
        return self._model

    def set_selected_features(
        self,
        indices: Optional[list[int]] = None,
        names: Optional[list[str]] = None,
    ) -> None:
        self.selected_feature_indices = list(indices or [])
        self.selected_feature_names = list(names or [])

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        if not self.selected_feature_indices:
            return X
        return X[:, self.selected_feature_indices]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBModel not trained or loaded.")

        X_transformed = self._transform_features(X)

        expected_features = self._model.num_features()
        actual_features = X_transformed.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"Dimension mismatch: Model expects {expected_features} features, "
                f"but the data has {actual_features} features."
            )

        dmat = xgb.DMatrix(X_transformed)
        return self._model.predict(dmat)

    def get_native_model(self) -> Any:
        return self._model

    def save(self, path: Union[str, Path]) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save XGBModel: model is None.")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # use JSON because it's stable and portable
        if not hasattr(self._model, "_estimator_type"):
            setattr(self._model, "_estimator_type", "regressor")

        p = p.with_suffix(SAVE_FILE_ENDING)
        self._model.save_model(str(p))
        meta_path = p.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "selected_feature_indices": self.selected_feature_indices,
                    "selected_feature_names": self.selected_feature_names,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        self._last_artifact_path = p

    def load(self, path: Union[str, Path]) -> bool:
        p = Path(path).with_suffix(SAVE_FILE_ENDING)
        if not p.exists():
            return False

        model = xgb.Booster()
        # if not hasattr(model, "_estimator_type"):
        #     setattr(model, "_estimator_type", "regressor")

        model.load_model(str(p))
        self._model = model
        meta_path = p.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.selected_feature_indices = list(
                meta.get("selected_feature_indices", []) or []
            )
            self.selected_feature_names = list(
                meta.get("selected_feature_names", []) or []
            )
        else:
            self.set_selected_features()
        self._last_artifact_path = p
        return True

    def get_model_size_kb(self) -> float:
        if self._last_artifact_path is None:
            return 0.0
        if not self._last_artifact_path.exists():
            return 0.0
        return self._last_artifact_path.stat().st_size / 1024

    def export_final_artifact(
        self,
        out_path: Union[str, Path],
        representative_data=None,
    ) -> float:
        self.save(out_path)
        return self.get_model_size_kb()
