from __future__ import annotations

from typing import Tuple

from src.models.core.keras_base import KerasBaseModel
from src.models.tf_architectures import create_resnet34


class ResNet34Model(KerasBaseModel):

    def build(self, input_shape: Tuple[int, ...]):
        self._model = create_resnet34(input_shape)
        return self._model
