from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np

from src.models.core.model import Model


class PLSModel(Model):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config)

    def build(self, input_shape=None) -> Any:
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("PLSModel not trained or loaded.")
        return self._model.predict(X).reshape(-1)

    def get_native_model(self) -> Any:
        return self._model

    def save(self, path: Union[str, Path]) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save PLSModel: model is None.")

        p = Path(path).with_suffix(".joblib")
        p.parent.mkdir(parents=True, exist_ok=True)
        for attr in ["x_scores_", "y_scores_", "x_rotations_", "y_rotations_"]:
            if hasattr(self._model, attr):
                setattr(self._model, attr, None)

        joblib.dump(self._model, p)
        self._last_artifact_path = p

    def load(self, path: Union[str, Path]) -> bool:
        p = Path(path).with_suffix(".joblib")
        if not p.exists():
            return False

        self._model = joblib.load(p)
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
