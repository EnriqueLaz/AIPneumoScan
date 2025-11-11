"""Generate model probabilities and ground-truth labels for PneumoScan."""

from __future__ import annotations

import os
import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

# Force CPU usage to avoid GPU/Metal issues in sandboxed environments.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from pneumoscan.data.preprocess import preprocess_image
from pneumoscan.model.load import load_model

LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1,
}


def iter_image_paths(dataset_dir: Path, *, limit: int | None = None) -> Iterable[Path]:
    """Yield image paths from the dataset respecting the label subfolders."""
    supported_exts = (".jpeg", ".jpg", ".png")
    count = 0

    for label_name in sorted(LABEL_MAP):
        label_dir = dataset_dir / label_name
        if not label_dir.exists():
            continue
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in supported_exts:
                continue
            yield path
            count += 1
            if limit is not None and count >= limit:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for PneumoScan dataset.")
    parser.add_argument(
        "--dataset",
        default="data/chest_xray/val",
        type=Path,
        help="Path to dataset split containing 'NORMAL' and 'PNEUMONIA' subfolders.",
    )
    parser.add_argument(
        "--weights",
        default="models/efficientnet_pneumonia.weights.h5",
        type=Path,
        help="Path to the trained weights file.",
    )
    parser.add_argument(
        "--output",
        default="metrics/pneumoscan_probs.npz",
        type=Path,
        help="Output path for the saved numpy archive.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

    model = load_model(str(args.weights))
    weights_loaded = bool(getattr(model, "_pneumoscan_weights_loaded", False))
    if not weights_loaded:
        raise RuntimeError(
            "Model weights were not loaded. Ensure the weights file exists and is compatible."
        )

    paths = list(iter_image_paths(dataset_dir, limit=args.limit))
    if not paths:
        raise RuntimeError("No images found. Check dataset path and supported extensions.")

    y_true = np.zeros(len(paths), dtype=np.int32)
    y_prob = np.zeros(len(paths), dtype=np.float32)

    for idx, path in enumerate(paths):
        label_name = path.parent.name
        y_true[idx] = LABEL_MAP[label_name]
        image_bytes = path.read_bytes()
        batch = preprocess_image(image_bytes)
        preds = model.predict(batch, verbose=0)
        y_prob[idx] = float(preds[0][0])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, y_true=y_true, y_prob=y_prob, metadata={"dataset": str(dataset_dir)}, paths=[str(p) for p in paths])

    metrics = {
        "samples": len(paths),
        "dataset": str(dataset_dir),
        "output_file": str(args.output),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
