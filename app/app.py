"""Minimal Streamlit front-end for PneumoScan."""

from __future__ import annotations

import io
import os
from typing import Tuple

import streamlit as st
from PIL import Image

from pneumoscan.data.preprocess import preprocess_image
from pneumoscan.model.load import load_model
from pneumoscan.model.predict import DETECTION_THRESHOLD

st.set_page_config(page_title="AI PneumoScan", page_icon="🫁", layout="centered")

MODEL_WEIGHTS_PATH = os.getenv(
    "PNEUMOSCAN_MODEL_PATH",
    "models/efficientnet_pneumonia.weights.h5",
)


@st.cache_resource(show_spinner="Cargando modelo local...")
def load_local_model(path: str) -> Tuple[object, bool]:
    """Carga el modelo TensorFlow y retorna si los pesos se encuentran listos."""
    model = load_model(path)
    weights_ready = bool(getattr(model, "_pneumoscan_weights_loaded", False))
    return model, weights_ready


def run_local_prediction(model, image_bytes: bytes) -> Tuple[str, float]:
    """Ejecuta inferencia y devuelve la etiqueta amigable y la probabilidad cruda."""
    batch = preprocess_image(image_bytes)
    preds = model.predict(batch)
    prob = float(preds[0][0])
    label = "NEUMONÍA DETECTADA" if prob > DETECTION_THRESHOLD else "PULMONES NORMALES"
    return label, prob


CUSTOM_CSS = """
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% -10%, rgba(84, 165, 255, 0.28), rgba(9, 33, 56, 0.86));
    color: #f5faff;
    padding-top: 1.5rem;
}

.main .block-container {
    padding: 1.2rem 1.5rem 2.4rem 1.5rem;
    max-width: 960px;
}

.hero-card {
    backdrop-filter: blur(14px);
    background: linear-gradient(135deg, rgba(15, 76, 117, 0.78), rgba(75, 195, 189, 0.7));
    border-radius: 20px;
    box-shadow: 0 26px 50px rgba(5, 15, 35, 0.45);
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.hero-icon {
    width: 62px;
    height: 62px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.1rem;
    background: rgba(255, 255, 255, 0.22);
    box-shadow: inset 0 1px 4px rgba(255, 255, 255, 0.4);
}

.hero-card h1 {
    margin: 0;
    font-size: 2.05rem;
    font-weight: 700;
    letter-spacing: 0.015em;
    color: #f7fbff;
}

.hero-card p {
    margin: 0.2rem 0 0 0;
    color: rgba(240, 248, 255, 0.85);
    line-height: 1.45;
    font-size: 0.95rem;
}

.hero-pill-row {
    margin-top: 0.9rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.hero-pill {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    padding: 0.45rem 1.1rem;
    font-size: 0.87rem;
    font-weight: 600;
    color: #f4fbff;
    letter-spacing: 0.01em;
}

.hero-footnote {
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: rgba(235, 244, 255, 0.75);
    font-weight: 500;
}

.status-chip {
    margin-top: 0.9rem;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.16);
    color: #f0f6ff;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.02em;
}

.glass-panel {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
    box-shadow: 0 18px 36px rgba(11, 23, 40, 0.28);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.glass-panel h3 {
    color: #f0f6ff;
    margin-bottom: 0.6rem;
    font-size: 1.08rem;
}

.glass-panel p {
    color: rgba(240, 246, 255, 0.9);
    font-size: 0.92rem;
    line-height: 1.45;
}

[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.10);
    border: 1px dashed rgba(255, 255, 255, 0.35);
    padding: 1.1rem;
    border-radius: 15px;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.1);
}

[data-testid="stFileUploader"] > div:first-child {
    color: rgba(240, 246, 255, 0.86);
    font-weight: 600;
}

.stButton > button {
    padding: 0.75rem 1.4rem;
    border-radius: 12px;
    border: none;
    background: linear-gradient(135deg, #3bd0c9, #1a7dd7);
    color: white;
    font-weight: 700;
    letter-spacing: 0.03em;
    box-shadow: 0 16px 30px rgba(18, 120, 215, 0.32);
}

.stButton > button:hover {
    filter: brightness(1.05);
}

.preview-card {
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 24px 48px rgba(3, 15, 32, 0.45);
}

.preview-caption {
    background: rgba(12, 26, 44, 0.85);
    padding: 0.8rem 1rem;
    color: rgba(225, 235, 246, 0.92);
    font-size: 0.92rem;
}

.result-card {
    margin-top: 1.4rem;
    border-radius: 18px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    font-weight: 700;
    font-size: 1.35rem;
    letter-spacing: 0.015em;
    box-shadow: 0 30px 65px rgba(8, 23, 48, 0.45);
    color: #0f1621;
}

.result-card.result-positive {
    background: linear-gradient(135deg, #ffe2e2, #ff8c8c);
}

.result-card.result-negative {
    background: linear-gradient(135deg, #d7fdeb, #8ce3b6);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05111f 0%, #0d243f 100%);
    padding-top: 1rem;
}

section[data-testid="stSidebar"] .sidebar-container {
    padding: 1rem 0.6rem 1.4rem 0.6rem;
}

.sidebar-container h2 {
    color: #dff3ff;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}

.sidebar-container p {
    color: rgba(215, 230, 255, 0.7);
    font-size: 0.88rem;
}

.sidebar-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: rgba(59, 208, 201, 0.18);
    color: #5ce5e0;
    font-weight: 600;
    font-size: 0.85rem;
}

.sidebar-warning {
    background: rgba(255, 177, 94, 0.18);
    color: #ffc78b;
}

.feature-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1rem;
    margin-bottom: 1.3rem;
}

.feature-card {
    background: rgba(255, 255, 255, 0.13);
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 18px 32px rgba(6, 18, 36, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.feature-card h4 {
    margin: 0.2rem 0 0.4rem 0;
    color: #f4fbff;
    font-size: 1rem;
}

.feature-card p {
    margin: 0;
    color: rgba(232, 241, 255, 0.8);
    font-size: 0.9rem;
    line-height: 1.35;
}

.feature-icon {
    font-size: 1.35rem;
    margin-bottom: 0.2rem;
    display: inline-block;
}

.workflow-grid {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 1.1rem;
    margin-bottom: 1.5rem;
}

.workflow-panel {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 20px 40px rgba(7, 20, 38, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.workflow-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f4fbff;
    margin-bottom: 0.8rem;
}

.step-list {
    counter-reset: step-counter;
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.65rem;
}

.step-list li {
    background: rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 0.75rem 0.85rem 0.75rem 2.4rem;
    position: relative;
    color: rgba(229, 239, 255, 0.9);
    font-size: 0.92rem;
    line-height: 1.35;
}

.step-list li::before {
    counter-increment: step-counter;
    content: counter(step-counter);
    position: absolute;
    left: 0.85rem;
    top: 50%;
    transform: translateY(-50%);
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    color: #07223f;
}

.checklist-list {
    padding-left: 1rem;
    margin: 0 0 1rem 0;
    color: rgba(229, 239, 255, 0.9);
    font-size: 0.9rem;
    line-height: 1.4;
}

.checklist-list li {
    margin-bottom: 0.35rem;
}

.reminder-card {
    background: rgba(92, 229, 224, 0.12);
    border-radius: 14px;
    padding: 0.85rem 1rem;
    color: #8bf3ee;
    font-size: 0.9rem;
    border: 1px solid rgba(92, 229, 224, 0.25);
}

@media (max-width: 820px) {
    .workflow-grid {
        grid-template-columns: 1fr;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-icon">🫁</div>
        <div>
            <h1>AI PneumoScan</h1>
            <p>
                Herramienta digital para apoyar la lectura de radiografías de tórax
                y estimar la probabilidad de neumonía. Diseñada para equipos médicos,
                con una experiencia clara para pacientes y cuidadores que participan
                en la toma de decisiones.
            </p>
            <div class="hero-pill-row">
                <span class="hero-pill">Compatible con flujos hospitalarios</span>
                <span class="hero-pill">Notas explicativas para pacientes</span>
                <span class="hero-pill">Predicción asistida al instante</span>
            </div>
            <p class="hero-footnote">Uso asistido · No sustituye la valoración médica presencial.</p>
            <span class="status-chip">Modo disponible: Modelo local</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="feature-cards">
        <div class="feature-card">
            <span class="feature-icon">🗂️</span>
            <h4>Estudios compatibles</h4>
            <p>Radiografías PNG/JPG/JPEG, idealmente proyección PA, con soporte para AP o lateral.</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🔐</span>
            <h4>Privacidad ante todo</h4>
            <p>Anonimiza la imagen y confirma el consentimiento del paciente antes de subirla.</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">⚙️</span>
            <h4>Análisis asistido</h4>
            <p>Obtén una probabilidad estimada de neumonía para apoyar la decisión clínica.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="workflow-grid">
        <div class="workflow-panel">
            <div class="workflow-title">🧭 Flujo guiado</div>
            <ol class="step-list">
                <li>Sube la radiografía y verifica que esté en buen contraste y sin datos sensibles.</li>
                <li>Confirma la vista previa y los datos básicos del paciente antes de analizar.</li>
                <li>Revisa el resultado, agrega tu interpretación clínica y comunica al equipo tratante.</li>
            </ol>
        </div>
        <div class="workflow-panel">
            <div class="workflow-title">Checklist rápido</div>
            <ul class="checklist-list">
                <li>Identidad y anonimización del estudio confirmadas.</li>
                <li>Historia clínica y signos vitales recientes disponibles.</li>
                <li>Imagen nítida y sin artefactos que dificulten la lectura.</li>
                <li>El equipo sabe que la herramienta es apoyo, no diagnóstico final.</li>
            </ul>
            <div class="reminder-card">¿Necesitas recordatorio? Sube la radiografía y toca “Analizar imagen”. La app te devuelve una probabilidad estimada para apoyar tu decisión clínica.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="sidebar-container">
        <h2>Modo de inferencia</h2>
        <p><strong>Modelo local</strong><br><span style="opacity:0.7;">Predicción en tu equipo</span></p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, weights_ready = load_local_model(MODEL_WEIGHTS_PATH)
if not weights_ready:
    st.sidebar.markdown(
        '<div class="sidebar-container"><span class="sidebar-badge sidebar-warning">Pesos no encontrados</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        '<div class="sidebar-container"><span class="sidebar-badge">Pesos clínicos listos</span></div>',
        unsafe_allow_html=True,
    )

upload_col, display_col = st.columns((1, 1), gap="large")

with upload_col:
    uploaded_file = st.file_uploader(
        "Sube la radiografía (PNG o JPG)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )

if uploaded_file is None:
    with display_col:
        st.markdown(
            """
            <div class="glass-panel">
                <h3>Instrucciones rápidas</h3>
                <p>Selecciona una radiografía de tórax en formato PNG o JPG.
                El archivo permanece en tu dispositivo y se procesa únicamente de forma local.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    with display_col:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.image(image, caption=None, use_column_width=True)
        st.markdown(
            f'<div class="preview-caption">{uploaded_file.name}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Analizar imagen", use_container_width=True):
            with st.spinner("Procesando imagen..."):
                label, _ = run_local_prediction(model, image_bytes)
            result_class = (
                "result-positive" if label == "NEUMONÍA DETECTADA" else "result-negative"
            )
            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    {label}
                </div>
                """,
                unsafe_allow_html=True,
            )
