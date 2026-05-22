from __future__ import annotations

from typing import Tuple

try:
    import tf_keras as keras
    from tf_keras import Model, layers
except Exception:  # Fallback for environments without tf_keras
    import tensorflow as tf
    keras = tf.keras
    from tensorflow.keras import Model, layers

from src.models.core.keras_base import KerasBaseModel


def _tcn_residual_block(
    x,
    filters: int,
    dilation: int,
    dropout_rate: float,
    name: str,
):
    shortcut = x

    x = layers.Conv1D(
        filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop1")(x)

    x = layers.Conv1D(
        filters,
        kernel_size=3,
        padding="same",
        dilation_rate=dilation,
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=f"{name}_skip",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out_relu")(x)
    return x


def create_tcn_regressor(input_shape: Tuple[int, ...]) -> Model:
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(64, kernel_size=1, padding="same", use_bias=False, name="stem")(
        inputs
    )
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = _tcn_residual_block(x, filters=64, dilation=1, dropout_rate=0.15, name="tcn1")
    x = _tcn_residual_block(x, filters=64, dilation=2, dropout_rate=0.15, name="tcn2")
    x = _tcn_residual_block(x, filters=128, dilation=4, dropout_rate=0.2, name="tcn3")
    x = _tcn_residual_block(x, filters=128, dilation=8, dropout_rate=0.2, name="tcn4")

    avg = layers.GlobalAveragePooling1D(name="gap")(x)
    mx = layers.GlobalMaxPooling1D(name="gmp")(x)
    x = layers.Concatenate(name="pool_concat")([avg, mx])

    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.3, name="fc1_drop")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="TCN_Regressor")


class TCNModel(KerasBaseModel):
    def build(self, input_shape: Tuple[int, ...]):
        self._model = create_tcn_regressor(input_shape)
        return self._model
