from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from src.models.core.model import Model

SAVE_FILE_ENDING = ".json"


class MeanModel(Model):
    """Constant regressor that always predicts the training-target mean."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config=config)
        self.mean_value: float = float(config.get("mean_value", 0.0))

    def build(self, input_shape=None) -> Any:
        return self._model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MeanModel not trained or loaded.")
        return np.full(len(X), self.mean_value, dtype=np.float32)

    def get_native_model(self) -> Any:
        return self._model

    def save(self, path: Union[str, Path]) -> None:
        if self._model is None:
            raise RuntimeError("Cannot save MeanModel: model is None.")

        p = Path(path).with_suffix(SAVE_FILE_ENDING)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "mean",
            "mean_value": self.mean_value,
        }
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._last_artifact_path = p

    def load(self, path: Union[str, Path]) -> bool:
        p = Path(path).with_suffix(SAVE_FILE_ENDING)
        if not p.exists():
            return False

        payload = json.loads(p.read_text(encoding="utf-8"))
        self.mean_value = float(payload["mean_value"])
        self._model = payload
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
