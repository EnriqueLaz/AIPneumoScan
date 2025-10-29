"""Image preprocessing utilities."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Prepare a chest X-ray image for model prediction."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return arr
