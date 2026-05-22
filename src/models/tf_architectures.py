import tensorflow as tf
def _optional_tfmot():
    try:
        import tensorflow_model_optimization as tfmot  # type: ignore
        return tfmot
    except Exception:
        return None
try:
    import tf_keras as keras
    from tf_keras import Model, layers
except Exception:  # Fallback for environments without tf_keras
    import tensorflow as tf
    keras = tf.keras
    from tensorflow.keras import Model, layers

_TFMOT = _optional_tfmot()
_QuantizeBase = (
    _TFMOT.quantization.keras.QuantizeConfig if _TFMOT is not None else object
)


class Conv1DQuantizeConfig(_QuantizeBase):
    """Quantization Config for the Conv1D Layer"""

    def get_weights_and_quantizers(self, layer):
        if _TFMOT is None:
            raise ImportError(
                "tensorflow_model_optimization is required for quantization."
            )
        return [
            (
                layer.kernel,
                _TFMOT.quantization.keras.quantizers.LastValueQuantizer(
                    num_bits=8, symmetric=True, narrow_range=False, per_axis=False
                ),
            )
        ]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        if _TFMOT is None:
            raise ImportError(
                "tensorflow_model_optimization is required for quantization."
            )
        return [
            _TFMOT.quantization.keras.quantizers.MovingAverageQuantizer(
                num_bits=8, per_axis=False, symmetric=False, narrow_range=False
            )
        ]

    def get_config(self):
        return {}


class NoOpQuantizeConfig(_QuantizeBase):
    """No-op quantization config for structural/non-parametric layers.

    This config is intentionally empty: it provides no weight, activation, or
    output quantizers. It is used to annotate layers that should stay part of
    the quantization transformation flow (for graph consistency) but do not
    need custom fake-quant behavior in this project.
    """

    def get_weights_and_quantizers(self, layer):
        return []

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        pass

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


def residual_block(x, filters, stride=1, downsample=False, name="block"):
    shortcut = x

    if downsample:
        shortcut = layers.Conv1D(
            filters, 1, strides=stride, use_bias=False, name=f"{name}_shortcut_conv"
        )(x)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    # 1. Conv
    x = layers.Conv1D(
        filters, 3, strides=stride, padding="same", use_bias=False, name=f"{name}_conv1"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    # 2. Conv
    x = layers.Conv1D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(
        x
    )
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Add & ReLU
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_out_relu")(x)
    return x


def create_resnet34(input_shape) -> Model:
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(
        64, 3, strides=1, padding="same", use_bias=False, name="stem_conv"
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Stage 1 (64 filters, 3 blocks)
    x = residual_block(x, 64, name="s1_b1")
    x = residual_block(x, 64, name="s1_b2")
    x = residual_block(x, 64, name="s1_b3")

    # Stage 2 (128 filters, 4 blocks)
    x = residual_block(x, 128, stride=2, downsample=True, name="s2_b1")
    x = residual_block(x, 128, name="s2_b2")
    x = residual_block(x, 128, name="s2_b3")
    x = residual_block(x, 128, name="s2_b4")

    # Stage 3 (256 filters, 6 blocks)
    x = residual_block(x, 256, stride=2, downsample=True, name="s3_b1")
    for i in range(2, 7):
        x = residual_block(x, 256, name=f"s3_b{i}")

    # Stage 4 (512 filters, 3 blocks)
    x = residual_block(x, 512, stride=2, downsample=True, name="s4_b1")
    x = residual_block(x, 512, name="s4_b2")
    x = residual_block(x, 512, name="s4_b3")

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dense(128, activation="relu", name="fc2")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="ResNet34_1D")


def create_tiny_1d(input_shape) -> Model:
    """
    Tiny 1D CNN for BGL regression (fixed architecture):
    - Conv1D -> ReLU -> MaxPool
    - Conv1D -> ReLU -> MaxPool
    - Flatten -> Dense(ReLU) -> Dropout(0.2) -> Dense(linear)
    Designed to be small for microcontroller deployment.
    """
    inputs = layers.Input(shape=input_shape, name="input")

    # Feature extraction stage
    x = layers.Conv1D(filters=60, kernel_size=3, padding="same", name="conv1")(inputs)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling1D(pool_size=3, name="pool1")(x)

    x = layers.Conv1D(filters=30, kernel_size=3, padding="same", name="conv2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling1D(pool_size=3, name="pool2")(x)

    # Regression stage
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(12, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(1, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="Tiny1D_CNN")


def get_tf_architecture(name, input_shape) -> Model:
    if name == "resnet34":
        return create_resnet34(input_shape)
    if name == "tiny_cnn":
        return create_tiny_1d(input_shape)
    raise ValueError(f"Unknown architecture: {name}")
