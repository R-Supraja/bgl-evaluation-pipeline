from __future__ import annotations

from typing import Tuple

from src.models.core.keras_base import KerasBaseModel
from src.models.tf_architectures import create_tiny_1d


class TinyCNNModel(KerasBaseModel):

    def build(self, input_shape: Tuple[int, ...]):
        self._model = create_tiny_1d(input_shape)
        return self._model
