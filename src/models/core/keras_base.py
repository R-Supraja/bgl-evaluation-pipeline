from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf
def _optional_tfmot():
    try:
        import tensorflow_model_optimization as tfmot  # type: ignore
        return tfmot
    except Exception:
        return None
try:
    import tf_keras as keras
except Exception:  # Fallback for environments without tf_keras
    import tensorflow as tf
    keras = tf.keras

from src.models.core.model import Model
from src.models.tf_architectures import Conv1DQuantizeConfig, NoOpQuantizeConfig


class KerasBaseModel(Model):

    def build(self, input_shape):
        """Builds the concrete Keras model for the given input shape.

        Args:
            input_shape: Shape tuple.

        Raises:
            NotImplementedError: Always, because subclasses must provide the
                actual architecture.
        """
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Runs predictions and returns a prediction vector.

        Args:
            X: Input samples in the tensor shape expected by the model.

        Returns:
            Prediction array with one scalar output per sample.

        Raises:
            RuntimeError: If no model has been built or loaded yet.
        """
        if self._model is None:
            raise RuntimeError("Model not built/loaded.")
        y = self._model.predict(X, verbose=1)
        return y.reshape(-1)

    def get_native_model(self):
        return self._model

    @staticmethod
    def custom_objects() -> Dict[str, Any]:
        return {
            "Conv1DQuantizeConfig": Conv1DQuantizeConfig,
            "NoOpQuantizeConfig": NoOpQuantizeConfig,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Saves the model as a native Keras artifact.

        Before writing, pruning wrappers are stripped so the saved artifact is
        closer to the actual inference graph used for downstream export.

        Args:
            path: Target path without or with extension. The method always
                writes a ``.keras`` file.

        Raises:
            RuntimeError: If the model has not been built yet.
        """
        if self._model is None:
            raise RuntimeError("Cannot save: model not built.")

        p = Path(path).with_suffix(".keras")
        p.parent.mkdir(parents=True, exist_ok=True)

        model_to_save = self._strip_for_export(self._model)
        model_to_save.save(p)

        self._last_saved_path = p

    def load(self, path: Union[str, Path]) -> bool:
        """Loads a saved Keras model from disk.

        The loader first tries to restore the model inside quantization and
        pruning scopes so serialized optimization wrappers can be resolved. If
        that fails, it falls back to a plain Keras load.

        Args:
            path: Source path without or with extension. The method always
                reads from a ``.keras`` file.

        Returns:
            ``True`` if loading succeeded, otherwise ``False``.
        """
        p = Path(path).with_suffix(".keras")
        if not p.exists():
            return False

        tfmot = _optional_tfmot()
        if tfmot is not None:
            try:
                with tfmot.quantization.keras.quantize_scope(self.custom_objects()):
                    with tfmot.sparsity.keras.prune_scope():
                        self._model = keras.models.load_model(
                            p, compile=False, custom_objects=self.custom_objects()
                        )
            except Exception as e:
                print(f"EXCEPTION: {e}")
                try:
                    self._model = keras.models.load_model(p, compile=False)
                except Exception:
                    return False
        else:
            try:
                self._model = keras.models.load_model(p, compile=False)
            except Exception:
                return False

        self._last_saved_path = p
        return True

    def _strip_for_export(self, model: keras.Model) -> keras.Model:
        tfmot = _optional_tfmot()
        if tfmot is None:
            return model
        try:
            model = tfmot.sparsity.keras.strip_pruning(model)
            return model
        except Exception:
            return model

    def export_to_tflite(
        self,
        path: Union[str, Path],
        representative_data: Optional[np.ndarray] = None,
    ) -> float:
        """Exports the current model as a TFLite artifact.

        Args:
            path: Output path without or with extension. The method always
                writes a ``.tflite`` file.
            representative_data: Optional calibration samples used for INT8
                quantization.

        Returns:
            Artifact size in kilobytes. Returns ``0.0`` if conversion fails.

        Raises:
            RuntimeError: If the model has not been built or loaded.
        """

        if self._model is None:
            raise RuntimeError("Model not built or loaded.")

        tflite_bytes = self._convert_to_tflite_bytes(self._model, representative_data)
        if tflite_bytes is None:
            return 0.0

        p = Path(path).with_suffix(".tflite")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            f.write(tflite_bytes)

        self._last_artifact_path = p
        return p.stat().st_size / 1024

    def _convert_to_tflite_bytes(
        self,
        model: keras.Model,
        representative_data: Optional[np.ndarray],
    ) -> Optional[bytes]:
        """Converts a Keras model into raw TFLite bytes.

        If representative data is available, full INT8 quantization is
        configured. Quantized models are also treated as optimized export
        targets even when no representative dataset is supplied.

        Args:
            model: Source Keras model.
            representative_data: Optional calibration samples for converter
                representative dataset generation.

        Returns:
            Serialized TFLite model bytes, or ``None`` if conversion fails.
        """

        try:
            is_quantized = any(
                "quantize" in l.name.lower() or "QuantizeWrapper" in str(type(l))
                for l in model.layers
            )
            tfmot = _optional_tfmot()
            model_to_convert = (
                tfmot.sparsity.keras.strip_pruning(model) if tfmot is not None else model
            )
            converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)

            if (
                representative_data is not None and len(representative_data) > 0
            ) or is_quantized:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

                if representative_data is not None and len(representative_data) > 0:
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8

                    def rep_gen():
                        for i in range(len(representative_data)):
                            yield [representative_data[i : i + 1].astype(np.float32)]

                    converter.representative_dataset = rep_gen
                else:
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]

            return converter.convert()

        except Exception as e:
            print(f"[WARN] TFLite conversion failed: {e}")
            return None

    def get_model_size_kb(self) -> float:
        """Returns the size of the last exported artifact in kilobytes.

        Returns:
            File size in kilobytes, or ``0.0`` if no artifact is known.
        """
        if self._last_artifact_path is None:
            return 0.0
        if not self._last_artifact_path.exists():
            return 0.0
        return self._last_artifact_path.stat().st_size / 1024

    def export_final_artifact(
        self, out_path: Union[str, Path], representative_data=None
    ) -> float:
        """Exports both float and INT8 TFLite variants for final benchmarking.

        The float artifact is always produced as a reference export. The method
        returns the size of the INT8 artifact because that is the deployment
        target used for final comparisons.

        Args:
            out_path: Base output path used to derive ``_float`` and ``_int8``
                artifact names.
            representative_data: Optional calibration samples for the INT8
                export.

        Returns:
            Size of the generated INT8 artifact in kilobytes.
        """
        # Export a float baseline artifact for direct comparison.
        float_path = Path(out_path).with_name(Path(out_path).name + "_float")
        self.export_to_tflite(float_path, representative_data=None)

        # Export the quantized artifact that is used as final deployment target.
        int8_path = Path(out_path).with_name(Path(out_path).name + "_int8")

        size_kb = self.export_to_tflite(
            int8_path,
            representative_data=representative_data,
        )

        return size_kb
