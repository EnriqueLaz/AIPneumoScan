"""Explainability utilities for model predictions."""

from __future__ import annotations

from typing import Optional

import numpy as np


def grad_cam(
    model: object,
    image: np.ndarray,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for the provided image."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - require tf
        raise RuntimeError("TensorFlow must be installed to compute Grad-CAM.") from exc

    layer_name = layer_name or _find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(np.expand_dims(image, axis=0), tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8
    return heatmap.numpy()


def _find_last_conv_layer(model: object) -> str:
    """Best-effort search for the last convolutional layer."""
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            return layer.name
    raise ValueError("Could not automatically determine a convolutional layer.")
