"""Utility helpers for reading and validating imaging data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_image(path: str | Path, target_size: Optional[tuple[int, int]] = None) -> np.ndarray:
    """Return an image as a float32 NumPy array normalized to [0, 1]."""
    image_path = Path(path)
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported file format: {image_path.suffix}")

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        if target_size:
            image = image.resize(target_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def load_class_map(path: str | Path) -> dict[int, str]:
    """Load class labels from a JSON mapping file."""
    import json

    with Path(path).open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {int(k): v for k, v in data.items()}
