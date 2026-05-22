from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np


class Model(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.params = config.get("params", {})
        self._model: Any = None

        self._last_artifact_path: Optional[Path] = None

    @abstractmethod
    def build(self, input_shape: Tuple[int, ...]) -> Any: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def get_native_model(self) -> Any: ...

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None: ...

    @abstractmethod
    def load(self, path: Union[str, Path]) -> bool: ...

    @abstractmethod
    def get_model_size_kb(self) -> float: ...

    @abstractmethod
    def export_final_artifact(
        self,
        out_path: Union[str, Path],
        representative_data=None,
    ) -> float: ...

    @property
    def is_built(self) -> bool:
        return self._model is not None
