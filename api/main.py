"""FastAPI application exposing the PneumoScan EfficientNet model."""

from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from pneumoscan.model.load import load_model
from pneumoscan.model.predict import predict_pneumonia

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# ============================================================
# Configuración
# ============================================================
MODEL_PATH = "models/efficientnet_pneumonia.weights.h5"

app = FastAPI(
    title="AI PneumoScan API",
    description="API para detección de neumonía con EfficientNetB0.",
    version="1.0.0",
)

# ============================================================
# Carga del modelo
# ============================================================
LOGGER.info("🔹 Cargando modelo...")
model = load_model(MODEL_PATH)
MODEL_LOADED = bool(getattr(model, "_pneumoscan_weights_loaded", False))
if MODEL_LOADED:
    LOGGER.info("✅ Modelo cargado correctamente.")
else:
    LOGGER.warning("⚠️ Modelo cargado sin pesos entrenados. Verifica el archivo de pesos.")


# ============================================================
# Endpoints
# ============================================================
@app.get("/health")
def health_check() -> dict[str, object]:
    return {"status": "ok", "model_loaded": MODEL_LOADED, "model": "EfficientNet PneumoScan"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    try:
        contents = await file.read()
        return predict_pneumonia(model, contents, file.filename)
    except Exception as exc:  # pragma: no cover - defensive error handling
        LOGGER.exception("Error al procesar la predicción.")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


# ============================================================
# Main (local)
# ============================================================
if __name__ == "__main__":  # pragma: no cover - sólo ejecución manual
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
