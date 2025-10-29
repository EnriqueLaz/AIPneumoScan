"""Prediction helpers wrapping the FastAPI response."""

from __future__ import annotations

from fastapi.responses import JSONResponse

from pneumoscan.data.preprocess import preprocess_image

RULE_IN_THRESHOLD = 0.9
ACTIVE_MONITOR_THRESHOLD = 0.7


def predict_pneumonia(model, image_bytes: bytes, filename: str) -> JSONResponse:
    """Run inference on the provided image and return a JSON response."""
    batch = preprocess_image(image_bytes)
    preds = model.predict(batch)
    prob = float(preds[0][0])

    if prob >= RULE_IN_THRESHOLD:
        risk_level = "red"
        friendly_label = "Muy alta sospecha de neumonía"
        recommendation = "Prioriza revisión experta/immediata."
        prediction_label = "PNEUMONIA"
    elif prob >= ACTIVE_MONITOR_THRESHOLD:
        risk_level = "amber"
        friendly_label = "Zona ámbar: no descartar neumonía"
        recommendation = "Requiere revisión manual y/o estudios complementarios."
        prediction_label = "REVIEW"
    else:
        risk_level = "green"
        friendly_label = "Sin sospecha de neumonía"
        recommendation = "Continúa con vigilancia clínica habitual."
        prediction_label = "NORMAL"

    details = (
        f"{recommendation} Umbrales: rojo ≥ {RULE_IN_THRESHOLD:.2f}, "
        f"ámbar ≥ {ACTIVE_MONITOR_THRESHOLD:.2f}. Archivo: {filename}"
    )

    return JSONResponse(
        {
            "filename": filename,
            "label": friendly_label,
            "confidence": prob,
            "details": details,
            "prediction_label": prediction_label,
            "pneumonia_probability": prob,
            "risk_level": risk_level,
            "thresholds": {
                "rule_in": RULE_IN_THRESHOLD,
                "active_monitor": ACTIVE_MONITOR_THRESHOLD,
            },
        }
    )
