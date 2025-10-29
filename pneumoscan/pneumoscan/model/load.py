"""Model construction and loading utilities for PneumoScan."""

from __future__ import annotations

import logging
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

LOGGER = logging.getLogger(__name__)


def build_model(img_size: tuple[int, int] = (224, 224), dropout_rate: float = 0.3) -> tf.keras.Model:
    """Assemble the EfficientNet-based classifier."""
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_shape=(img_size[0], img_size[1], 3),
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="EfficientNet_PneumoScan")
    return model


def load_model(weights_path: str) -> tf.keras.Model:
    """Instantiate the classifier and load its weights."""
    model = build_model()
    weights_file = Path(weights_path)

    weights_loaded = False

    if not weights_file.exists():
        LOGGER.warning("Weights file %s not found. Using randomly initialised weights.", weights_path)
    else:
        try:
            model.load_weights(weights_path)
            weights_loaded = True
        except Exception as exc:  # pragma: no cover - relies on external file availability
            LOGGER.warning("Unable to load weights from %s: %s", weights_path, exc)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    setattr(model, "_pneumoscan_weights_loaded", weights_loaded)
    return model
