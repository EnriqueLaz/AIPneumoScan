"""Prediction helpers wrapping the FastAPI response."""

from __future__ import annotations

from fastapi.responses import JSONResponse

from pneumoscan.data.preprocess import preprocess_image

# Threshold selected via Youden J on data/chest_xray/val (2025-11-05 run).
DETECTION_THRESHOLD = 0.9914915


def predict_pneumonia(model, image_bytes: bytes, filename: str) -> JSONResponse:
    """Run inference on the provided image and return a JSON response."""
    batch = preprocess_image(image_bytes)
    preds = model.predict(batch)
    prob = float(preds[0][0])

    detected = prob > DETECTION_THRESHOLD
    friendly_label = "Neumonía detectada" if detected else "Pulmones normales"

    return JSONResponse(
        {
            "filename": filename,
            "label": friendly_label,
            "confidence": prob,
            "detected": detected,
            "threshold": DETECTION_THRESHOLD,
        }
    )
