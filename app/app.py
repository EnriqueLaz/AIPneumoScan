"""Streamlit front-end for PneumoScan with configurable inference backends."""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from PIL import Image, ImageStat

from pneumoscan.data.preprocess import preprocess_image
from pneumoscan.model.load import load_model
from pneumoscan.model.predict import (
    ACTIVE_MONITOR_THRESHOLD as MODEL_AMBER_THRESHOLD,
    RULE_IN_THRESHOLD as MODEL_RULE_IN_THRESHOLD,
)

st.set_page_config(page_title="AI PneumoScan", page_icon="🫁", layout="wide")

DEFAULT_API_ENDPOINT = os.getenv("PNEUMOSCAN_API_ENDPOINT", "http://localhost:8000/predict")
MODEL_WEIGHTS_PATH = os.getenv(
    "PNEUMOSCAN_MODEL_PATH",
    "models/efficientnet_pneumonia.weights.h5",
)


@dataclass
class APISettings:
    endpoint: str
    timeout: float
    headers: Dict[str, str]
    extra_fields: Dict[str, str]
    verify_ssl: bool


@dataclass
class PredictionResult:
    label: str
    confidence: float
    details: str


@st.cache_resource(show_spinner="Inicializando modelo de IA...")
def get_local_model(weights_path: str):
    """Carga y mantiene en caché el modelo TensorFlow para inferencia local."""
    model = load_model(weights_path)
    weights_ready = bool(getattr(model, "_pneumoscan_weights_loaded", False))
    return model, weights_ready


st.markdown(
    """
    <style>
    :root {
        --accent-color: #0f4c75;
        --accent-color-secondary: #4ba3c3;
        --accent-light: rgba(75, 163, 195, 0.15);
        --surface: #ffffff;
        --surface-alt: #f3f6fb;
        --text-main: #1d2a34;
        --text-muted: #5c6a75;
    }
    body {
        background: linear-gradient(180deg, #eff6ff 0%, #f8fbff 45%, #ffffff 100%);
    }
    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(145deg, rgba(15, 76, 117, 0.94), rgba(75, 163, 195, 0.82));
        border-radius: 20px;
        padding: 2.5rem 3rem;
        color: #f7fbff;
        box-shadow: 0 26px 60px rgba(16, 42, 67, 0.25);
        margin-bottom: 2.4rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .hero h1 {
        font-size: 2.48rem;
        margin-bottom: 0.6rem;
        letter-spacing: 0.01em;
    }
    .hero p {
        font-size: 1.08rem;
        opacity: 0.92;
        max-width: 780px;
        line-height: 1.65;
    }
    .hero small {
        display: inline-block;
        margin-top: 1.4rem;
        font-size: 0.86rem;
        opacity: 0.78;
    }
    .badge-group {
        margin-top: 1.8rem;
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255, 255, 255, 0.22);
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-size: 0.94rem;
        letter-spacing: 0.015em;
        backdrop-filter: blur(10px);
    }
    .card {
        background: var(--surface);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        box-shadow: 0 12px 28px rgba(18, 38, 60, 0.08);
        border: 1px solid rgba(15, 76, 117, 0.08);
    }
    .placeholder-card {
        border: 1px dashed rgba(15, 76, 117, 0.28);
        border-radius: 14px;
        padding: 2.2rem;
        background: linear-gradient(135deg, rgba(75, 163, 195, 0.08), rgba(15, 76, 117, 0.05));
        color: var(--text-muted);
        text-align: center;
        font-size: 0.96rem;
    }
    .placeholder-card strong {
        color: var(--text-main);
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 1rem;
        margin: 1.8rem 0 2.2rem;
    }
    .status-card {
        display: flex;
        gap: 0.85rem;
        align-items: flex-start;
        background: var(--surface);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 14px 30px rgba(18, 38, 60, 0.1);
        border: 1px solid rgba(15, 76, 117, 0.08);
    }
    .status-icon {
        width: 44px;
        height: 44px;
        border-radius: 14px;
        background: var(--accent-light);
        color: var(--accent-color);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.6);
    }
    .status-title {
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.2rem;
    }
    .status-text {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .workflow-card {
        background: var(--surface);
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 22px 45px rgba(15, 76, 117, 0.12);
        border: 1px solid rgba(15, 76, 117, 0.08);
    }
    .workflow-heading {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 1.2rem;
    }
    .timeline {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .timeline-step {
        display: flex;
        gap: 1rem;
        align-items: flex-start;
    }
    .timeline-index {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: var(--accent-color);
        color: #f4fbff;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
        box-shadow: 0 10px 24px rgba(15, 76, 117, 0.28);
    }
    .timeline-body {
        flex: 1;
        background: var(--surface-alt);
        border-radius: 14px;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(15, 76, 117, 0.08);
    }
    .timeline-body strong {
        color: var(--text-main);
        display: block;
        margin-bottom: 0.35rem;
    }
    .timeline-body span {
        color: var(--text-muted);
        font-size: 0.92rem;
        line-height: 1.55;
        display: block;
    }
    .checklist-card {
        background: var(--surface);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        border: 1px solid rgba(15, 76, 117, 0.12);
        box-shadow: 0 18px 38px rgba(18, 38, 60, 0.1);
    }
    .checklist-card h4 {
        margin-bottom: 0.7rem;
        color: var(--text-main);
    }
    .checklist-card ul {
        margin: 0;
        padding-left: 1.1rem;
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.55;
    }
    .help-banner {
        background: linear-gradient(120deg, rgba(75, 163, 195, 0.12), rgba(15, 76, 117, 0.18));
        border-radius: 18px;
        padding: 1rem 1.4rem;
        color: var(--text-main);
        border: 1px solid rgba(15, 76, 117, 0.22);
        margin-top: 1.2rem;
    }
    .help-banner strong {
        display: block;
        margin-bottom: 0.35rem;
    }
    .result-card {
        background: var(--surface);
        border-left: 6px solid var(--accent-color-secondary);
        border-radius: 18px;
        padding: 1.45rem 1.7rem;
        box-shadow: 0 24px 45px rgba(15, 76, 117, 0.22);
        margin-top: 1.3rem;
    }
    .result-label {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-main);
        margin-bottom: 0.3rem;
    }
    .metric-grid {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }
    .metric-chip {
        background: rgba(75, 163, 195, 0.15);
        border-radius: 999px;
        padding: 0.55rem 1.05rem;
        color: var(--accent-color);
        font-weight: 600;
        letter-spacing: 0.015em;
        font-size: 0.92rem;
    }
    .result-insight {
        background: rgba(15, 76, 117, 0.07);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-top: 0.85rem;
        color: var(--accent-color);
        font-weight: 500;
    }
    .result-insight span {
        font-weight: 600;
    }
    .stTabs [role="tablist"] {
        gap: 0.8rem;
    }
    .stTabs [role="tab"] {
        border-radius: 999px;
        background: var(--surface);
        padding: 0.5rem 1.1rem;
        border: 1px solid rgba(15, 76, 117, 0.15);
        color: var(--text-muted);
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(15, 76, 117, 0.95), rgba(75, 163, 195, 0.9)) !important;
        color: #f4fbff !important;
        border: none !important;
    }
    .stAlert {
        border-radius: 14px;
        border: 1px solid rgba(15, 76, 117, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>🫁 AI PneumoScan</h1>
        <p>
            Herramienta digital para apoyar la lectura de radiografías de tórax
            y estimar la probabilidad de neumonía. Diseñada para equipos médicos,
            con una experiencia clara para pacientes y cuidadores que participan
            en la toma de decisiones.
        </p>
        <div class="badge-group">
            <span class="badge">Compatible con flujos hospitalarios</span>
            <span class="badge">Notas explicativas para pacientes</span>
            <span class="badge">Predicción asistida al instante</span>
        </div>
        <small>Uso clínico asistido · No sustituye la valoración médica presencial.</small>
    </section>
    """,
    unsafe_allow_html=True,
)

# Additional layout and workflow sections (unchanged from provided design)
st.markdown(
    """
    <div class="status-grid">
        <div class="status-card">
            <div class="status-icon">🩻</div>
            <div>
                <div class="status-title">Estudios compatibles</div>
                <p class="status-text">Radiografías PNG/JPG/JPEG, preferentemente proyección PA; admitir AP/Lateral.</p>
            </div>
        </div>
        <div class="status-card">
            <div class="status-icon">🔒</div>
            <div>
                <div class="status-title">Privacidad ante todo</div>
                <p class="status-text">Anonimiza la imagen y verifica el consentimiento del paciente antes de subirla.</p>
            </div>
        </div>
        <div class="status-card">
            <div class="status-icon">⚙️</div>
            <div>
                <div class="status-title">Análisis asistido</div>
                <p class="status-text">Carga la radiografía y obtén una probabilidad estimada de neumonía en segundos.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

workflow_col, checklist_col = st.columns((1.9, 1.1), gap="large")
with workflow_col:
    st.markdown(
        """
        <div class="workflow-card">
            <div class="workflow-heading">🧭 Flujo guiado</div>
            <div class="timeline">
                <div class="timeline-step">
                    <span class="timeline-index">1</span>
                    <div class="timeline-body">
                        <strong>Sube la radiografía</strong>
                        <span>Asegúrate de que esté en formato PNG/JPG, con buena exposición y sin metadatos sensibles visibles.</span>
                    </div>
                </div>
                <div class="timeline-step">
                    <span class="timeline-index">2</span>
                    <div class="timeline-body">
                        <strong>Revisa la configuración</strong>
                        <span>Confirma que la imagen se previsualiza correctamente y que los datos del paciente están listos.</span>
                    </div>
                </div>
                <div class="timeline-step">
                    <span class="timeline-index">3</span>
                    <div class="timeline-body">
                        <strong>Analiza y comunica</strong>
                        <span>Revisa la confianza, agrega tu interpretación clínica y conversa el resultado con el equipo tratante.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with checklist_col:
    st.markdown(
        """
        <div class="checklist-card">
            <h4>Checklist rápido</h4>
            <ul>
                <li>Confirmaste identidad y anonimización del estudio.</li>
                <li>Dispones de la historia clínica y signos vitales recientes.</li>
                <li>La imagen se visualiza clara y sin artefactos que dificulten la interpretación.</li>
                <li>El equipo médico sabe que la herramienta es apoyo, no diagnóstico final.</li>
            </ul>
        </div>
        <div class="help-banner">
            <strong>¿Necesitas recordatorio?</strong>
            Sube la radiografía y toca <em>Analizar imagen</em>. La app devolverá una probabilidad estimada para apoyar tu decisión clínica.
        </div>
        """,
        unsafe_allow_html=True,
    )

tabs = st.tabs(["Panel principal", "Formato API", "Recomendaciones"])

with tabs[1]:
    st.markdown(
        """
        **Petición esperada** (`multipart/form-data`):

        ```
        POST /predict
        file=<contenido_binario_de_la_imagen>
        ```

        **Respuesta sugerida (JSON):**

        ```json
        {
          "label": "Probable neumonía",
          "confidence": 0.87,
          "details": "Modelo DenseNet v1.3 - 08/2023"
        }
        ```

        **Encabezados opcionales (JSON):**

        ```json
        {
          "Authorization": "Bearer <TOKEN>",
          "X-Hospital-ID": "central-01"
        }
        ```

        **Campos extra en el formulario (JSON):**

        ```json
        {
          "patient_id": "12345",
          "study_type": "rx_torax"
        }
        ```
        """
    )

with tabs[2]:
    st.markdown(
        """
        **Para profesionales de la salud**

        - Valide que la radiografía corresponda al paciente correcto.
        - Utilice la probabilidad como apoyo, no como diagnóstico definitivo.
        - Documente en la historia clínica si la decisión final difiere del algoritmo.

        **Para pacientes y cuidadores**

        - Consulte a su médico antes de sacar conclusiones o iniciar tratamientos.
        - Comparta síntomas recientes y antecedentes respiratorios para una lectura integral.
        - Solicite siempre que un profesional explique el resultado en palabras sencillas.
        """
    )


st.sidebar.header("Configuración")
st.sidebar.markdown(
    "Seleccione la fuente de inferencia. Puede conectar un servicio institucional o "
    "utilizar la demostración local para capacitación."
)
inference_mode = st.sidebar.radio(
    "Modo de inferencia",
    options=("Demo local", "Modelo local", "API externa"),
)

api_settings: Optional[APISettings] = None
config_errors: List[str] = []
local_model = None
local_model_ready = False
local_model_error: Optional[str] = None

if inference_mode == "API externa":
    api_endpoint = st.sidebar.text_input(
        "Endpoint de inferencia",
        value=DEFAULT_API_ENDPOINT,
        help="Introduce la URL de tu servicio de predicción (método POST).",
    ).strip()
    st.sidebar.caption("La solicitud enviará la imagen en el campo `file`.")

    api_timeout = st.sidebar.slider(
        "Tiempo de espera (segundos)",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Tiempo máximo de espera antes de cancelar la solicitud.",
    )

    api_token = st.sidebar.text_input(
        "Token Bearer (opcional)",
        type="password",
        help="Se enviará como encabezado Authorization si se completa.",
    ).strip()

    header_json = st.sidebar.text_area(
        "Encabezados adicionales (JSON opcional)",
        placeholder='{"X-Hospital-ID": "central-01"}',
        height=80,
    )
    extra_fields_json = st.sidebar.text_area(
        "Campos extra en la petición (JSON opcional)",
        placeholder='{"patient_id": "12345"}',
        help="Se envían como campos de formulario junto con la imagen.",
        height=80,
    )
    verify_ssl = st.sidebar.checkbox(
        "Verificar certificado SSL",
        value=True,
        help="Desmárcalo sólo si tu API usa certificados autofirmados.",
    )

    parsed_headers: Dict[str, str] = {}
    parsed_fields: Dict[str, str] = {}

    if api_token:
        parsed_headers["Authorization"] = (
            api_token if api_token.lower().startswith("bearer ") else f"Bearer {api_token}"
        )

    if header_json.strip():
        try:
            header_payload = json.loads(header_json)
            if not isinstance(header_payload, dict):
                raise ValueError("El JSON de encabezados debe ser un objeto.")
            parsed_headers.update({str(key): str(value) for key, value in header_payload.items()})
        except (json.JSONDecodeError, ValueError) as exc:
            config_errors.append(f"Encabezados inválidos: {exc}")

    if extra_fields_json.strip():
        try:
            form_payload = json.loads(extra_fields_json)
            if not isinstance(form_payload, dict):
                raise ValueError("El JSON de campos extra debe ser un objeto.")
            parsed_fields.update({str(key): str(value) for key, value in form_payload.items()})
        except (json.JSONDecodeError, ValueError) as exc:
            config_errors.append(f"Campos extra inválidos: {exc}")

    for error in config_errors:
        st.sidebar.error(error)

    api_settings = APISettings(
        endpoint=api_endpoint,
        timeout=float(api_timeout),
        headers=parsed_headers,
        extra_fields=parsed_fields,
        verify_ssl=verify_ssl,
    )
elif inference_mode == "Modelo local":
    st.sidebar.caption(
        "Utiliza el modelo TensorFlow empaquetado. Requiere que el archivo de pesos esté disponible localmente."
    )
    st.sidebar.markdown(f"Pesos esperados en: `{MODEL_WEIGHTS_PATH}`")
    try:
        local_model, local_model_ready = get_local_model(MODEL_WEIGHTS_PATH)
    except Exception as exc:  # pragma: no cover - mostramos el error en la UI
        local_model_error = str(exc)
    if local_model_error:
        st.sidebar.error(f"No se pudo inicializar el modelo: {local_model_error}")
    elif local_model_ready:
        st.sidebar.success("Modelo clínico cargado correctamente.")
    else:
        st.sidebar.warning(
            "Modelo cargado sin pesos reales. Sustituye el archivo de pesos para obtener predicciones confiables."
        )
else:  # Demo local
    st.sidebar.caption(
        "La demo local utiliza una heurística basada en la luminosidad de la imagen."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Formatos admitidos**: PNG, JPG y JPEG.")


def adapt_model_response(payload: Dict[str, Any]) -> PredictionResult:
    """Normaliza la respuesta del modelo externo en una PredictionResult."""
    if not isinstance(payload, dict):
        raise ValueError("La respuesta de la API debe ser un objeto JSON.")

    label = payload.get("label") or payload.get("prediction") or "Resultado no disponible"

    raw_confidence = payload.get("confidence", payload.get("probability"))
    if raw_confidence is None:
        raise ValueError(
            "La respuesta de la API debe incluir un puntaje de confianza (`confidence` o `probability`)."
        )
    try:
        confidence_value = float(raw_confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("El puntaje de confianza debe ser numérico.") from exc

    confidence_value = max(0.0, min(1.0, confidence_value))

    details = payload.get("details") or payload.get("explanation", "")

    return PredictionResult(
        label=str(label),
        confidence=confidence_value,
        details=str(details),
    )


def demo_prediction(image: Image.Image) -> PredictionResult:
    """Calcula una predicción heurística basada en la luminosidad media."""
    grayscale = image.convert("L")
    mean_intensity = ImageStat.Stat(grayscale).mean[0]

    score = max(0.0, min(1.0, (255.0 - mean_intensity) / 255.0))
    label = "Probable neumonía" if score >= 0.5 else "Poco probable"
    details = f"Luminosidad media: {mean_intensity:.2f}"

    return PredictionResult(label=label, confidence=score, details=details)


def call_inference_api(
    endpoint: str,
    file_bytes: bytes,
    filename: str,
    mime_type: Optional[str],
    *,
    timeout: float,
    headers: Optional[Dict[str, str]],
    form_fields: Optional[Dict[str, str]],
    verify_ssl: bool,
) -> Tuple[PredictionResult, Dict[str, Any], float]:
    """Envía la imagen al endpoint proporcionado y devuelve la predicción y la respuesta completa."""
    if not endpoint:
        raise ValueError("Debes indicar un endpoint de inferencia válido.")

    request_headers = headers or {}
    request_data = form_fields or {}
    files = {
        "file": (
            filename,
            file_bytes,
            mime_type or "application/octet-stream",
        )
    }

    start = time.perf_counter()
    response = requests.post(
        endpoint,
        files=files,
        data=request_data,
        headers=request_headers,
        timeout=timeout,
        verify=verify_ssl,
    )
    latency_s = time.perf_counter() - start
    response.raise_for_status()
    payload: Dict[str, Any] = response.json()

    prediction = adapt_model_response(payload)
    return prediction, payload, latency_s


def local_model_prediction(
    model,
    image_bytes: bytes,
    filename: str,
    *,
    weights_ready: bool,
) -> Tuple[PredictionResult, Dict[str, Any]]:
    """Ejecuta inferencia con el modelo TensorFlow empaquetado."""
    batch = preprocess_image(image_bytes)
    preds = model.predict(batch)
    prob = float(preds[0][0])
    if prob >= MODEL_RULE_IN_THRESHOLD:
        risk_level = "red"
        raw_label = "PNEUMONIA"
        friendly_label = "Muy alta sospecha de neumonía"
        recommendation = "Prioriza revisión experta/inmediata."
    elif prob >= MODEL_AMBER_THRESHOLD:
        risk_level = "amber"
        raw_label = "REVIEW"
        friendly_label = "Zona ámbar: no descartar neumonía"
        recommendation = "Requiere revisión manual y/o estudios complementarios."
    else:
        risk_level = "green"
        raw_label = "NORMAL"
        friendly_label = "Sin sospecha de neumonía"
        recommendation = "Continúa con vigilancia clínica habitual."
    weights_msg = (
        "Pesos clínicos cargados correctamente."
        if weights_ready
        else "Atención: usando pesos aleatorios. Sustituye el archivo de pesos."
    )
    details = (
        f"{weights_msg} {recommendation} Umbrales: rojo ≥ {MODEL_RULE_IN_THRESHOLD:.2f}, "
        f"ámbar ≥ {MODEL_AMBER_THRESHOLD:.2f}. (p={prob:.3f})"
    )

    prediction = PredictionResult(
        label=friendly_label,
        confidence=prob,
        details=details,
    )
    raw_payload: Dict[str, Any] = {
        "label": friendly_label,
        "confidence": prob,
        "details": details,
        "filename": filename,
        "weights_loaded": weights_ready,
        "mode": "local_tensorflow",
        "prediction_label": raw_label,
        "risk_level": risk_level,
        "thresholds": {
            "rule_in": MODEL_RULE_IN_THRESHOLD,
            "active_monitor": MODEL_AMBER_THRESHOLD,
        },
    }
    return prediction, raw_payload


RESULT_KEY = "pneumoscan_result"
FILE_SIGNATURE_KEY = "pneumoscan_file_signature"


def reset_result_state() -> None:
    st.session_state.pop(RESULT_KEY, None)


def store_result(
    prediction: PredictionResult,
    raw_response: Dict[str, Any],
    mode: str,
    filename: str,
    *,
    latency_s: Optional[float],
    endpoint: Optional[str],
) -> None:
    latency_ms = None if latency_s is None else max(0, round(latency_s * 1000))
    st.session_state[RESULT_KEY] = {
        "label": prediction.label,
        "confidence": float(prediction.confidence),
        "details": prediction.details,
        "mode": mode,
        "filename": filename,
        "latency_ms": latency_ms,
        "endpoint": endpoint,
        "raw": raw_response,
    }


def confidence_feedback(confidence: float) -> str:
    if confidence >= MODEL_RULE_IN_THRESHOLD:
        return "Confianza roja: prioriza revisión experta inmediata y coordina acciones clínicas."
    if confidence >= MODEL_AMBER_THRESHOLD:
        return "Confianza ámbar: no descartar; revisa antecedentes y complementa con estudios adicionales."
    return "Confianza verde: sin sospecha actual, mantén vigilancia clínica y seguimiento habitual."


def render_result_card(data: Dict[str, Any]) -> None:
    confidence_pct = max(0, min(100, int(data["confidence"] * 100)))
    latency_ms = data.get("latency_ms")
    latency_chip = (
        f'<span class="metric-chip">Tiempo: {latency_ms} ms</span>'
        if isinstance(latency_ms, int) and latency_ms >= 0
        else ""
    )

    endpoint_caption = data.get("endpoint") or ""
    endpoint_html = (
        f"<p style='color: var(--text-muted); font-size: 0.8rem; margin-top: 0.4rem;'>"
        f"Fuente: {endpoint_caption}"
        f"</p>"
        if endpoint_caption
        else ""
    )
    friendly_tip = confidence_feedback(float(data["confidence"]))

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{data['label']}</div>
            <div class="metric-grid">
                <span class="metric-chip">Confianza: {confidence_pct}%</span>
                <span class="metric-chip">Modo: {data['mode']}</span>
                {latency_chip}
            </div>
            <p style="color: var(--text-muted); margin-top: 1rem;">
                {data.get('details') or "Estimación probabilística generada por IA clínica asistida."}
            </p>
            <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.6rem;">
                Este resultado orienta la toma de decisiones. Confirme siempre con valoración profesional presencial.
            </p>
            <div class="result-insight">
                <span>Recomendación:</span> {friendly_tip}
            </div>
            {endpoint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence_pct / 100)
    with st.expander("Ver respuesta completa", expanded=False):
        st.json(data.get("raw", {}))


with tabs[0]:
    upload_col, preview_col = st.columns((1.2, 1), gap="large")

    with upload_col:
        st.markdown("#### Paso 1 · Carga la radiografía")
        uploaded_file = st.file_uploader(
            "Sube la radiografía de tórax",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
        )
        st.caption(
            "Asegúrate de que la imagen esté anonimizada y, si es posible, en escala de grises para facilitar la lectura."
        )

        analyze_button = st.button(
            "🔍 Analizar imagen",
            use_container_width=True,
            disabled=uploaded_file is None
            or (
                inference_mode == "API externa"
                and (
                    not api_settings
                    or not api_settings.endpoint
                    or bool(config_errors)
                )
            )
            or (
                inference_mode == "Modelo local"
                and (local_model_error is not None)
            ),
        )

    preview_bytes: Optional[bytes] = None
    preview_image: Optional[Image.Image] = None
    current_signature: Optional[Tuple[str, int]] = None

    if uploaded_file is not None:
        preview_bytes = uploaded_file.getvalue()
        preview_image = Image.open(io.BytesIO(preview_bytes)).convert("RGB")
        current_signature = (uploaded_file.name, uploaded_file.size)

    if st.session_state.get(FILE_SIGNATURE_KEY) != current_signature:
        st.session_state[FILE_SIGNATURE_KEY] = current_signature
        reset_result_state()

    with preview_col:
        st.markdown("#### Paso 2 · Vista previa")
        if preview_image is not None:
            st.image(preview_image, caption=uploaded_file.name, use_column_width=True)
            st.caption(
                f"Resolución: {preview_image.width}×{preview_image.height} px · "
                f"Tamaño: {len(preview_bytes or b'') / 1024:.1f} KB"
            )
        else:
            st.markdown(
                """
                <div class="placeholder-card">
                    <strong>No se ha cargado ninguna imagen.</strong><br>
                    Arrastre la radiografía o selecciónela desde su equipo para iniciar la lectura asistida.
                </div>
                """,
                unsafe_allow_html=True,
            )

    result_state: Optional[Dict[str, Any]] = st.session_state.get(RESULT_KEY)

    if analyze_button:
        if preview_image is None or preview_bytes is None:
            st.error("Carga una imagen antes de solicitar la predicción.")
        else:
            with st.spinner("Procesando imagen..."):
                try:
                    if inference_mode == "Demo local":
                        prediction = demo_prediction(preview_image)
                        raw_response = {
                            "label": prediction.label,
                            "confidence": prediction.confidence,
                            "details": prediction.details,
                            "mode": "demo_local",
                        }
                        latency = None
                    elif inference_mode == "Modelo local":
                        if local_model is None:
                            raise ValueError(
                                "No se pudo cargar el modelo local. Revisa el panel lateral para más detalles."
                            )
                        prediction, raw_response = local_model_prediction(
                            local_model,
                            preview_bytes,
                            uploaded_file.name,
                            weights_ready=local_model_ready,
                        )
                        latency = None
                    else:
                        if not api_settings or not api_settings.endpoint:
                            raise ValueError(
                                "Configura un endpoint válido antes de solicitar la predicción."
                            )
                        prediction, raw_response, latency = call_inference_api(
                            endpoint=api_settings.endpoint,
                            file_bytes=preview_bytes,
                            filename=uploaded_file.name,
                            mime_type=uploaded_file.type,
                            timeout=api_settings.timeout,
                            headers=api_settings.headers,
                            form_fields=api_settings.extra_fields,
                            verify_ssl=api_settings.verify_ssl,
                        )
                except requests.exceptions.RequestException as exc:
                    st.error(f"Error de comunicación con la API: {exc}")
                except json.JSONDecodeError:
                    st.error(
                        "La respuesta no es un JSON válido. Verifica que tu API responda con JSON."
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    endpoint_caption = None
                    if inference_mode == "API externa" and api_settings:
                        endpoint_caption = api_settings.endpoint
                    elif inference_mode == "Modelo local":
                        endpoint_caption = (
                            "Modelo local · pesos clínicos"
                            if local_model_ready
                            else "Modelo local · pesos aleatorios"
                        )
                    store_result(
                        prediction=prediction,
                        raw_response=raw_response,
                        mode=inference_mode,
                        filename=uploaded_file.name,
                        latency_s=latency,
                        endpoint=endpoint_caption,
                    )
                    result_state = st.session_state.get(RESULT_KEY)

    if result_state is not None:
        render_result_card(result_state)
    else:
        if preview_image is None:
            st.markdown(
                """
                <div class="help-banner">
                    <strong>¿Listo para comenzar?</strong>
                    Arrastra una radiografía o selecciónala desde tu equipo para activar el análisis asistido.
                    Recuerda verificar que la proyección y el formato sean correctos.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="help-banner">
                    <strong>Analiza la imagen cuando estés listo.</strong>
                    Revisa los datos del paciente y pulsa <em>Analizar imagen</em> para obtener la estimación.
                </div>
                """,
                unsafe_allow_html=True,
            )
