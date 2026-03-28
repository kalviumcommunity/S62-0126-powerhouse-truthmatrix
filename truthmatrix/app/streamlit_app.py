from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import calculate_credibility_score, generate_explanation
from src.fusion.fusion import calculate_truth_score, fuse_predictions
from src.image.predict import load_saved_image_model, resolve_image_model_path
from src.predict import predict as predict_text
from src.train import load_model as load_text_model


MODEL_DIR = PROJECT_ROOT / "models"
TEXT_MODEL_PATH = MODEL_DIR / "model.pkl"


st.set_page_config(
    page_title="TruthMatrix | Multimodal Detection",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def get_text_model():
    return load_text_model(TEXT_MODEL_PATH)


@st.cache_resource
def get_image_model():
    return load_saved_image_model(resolve_image_model_path())


def _is_text_model_ready() -> bool:
    return TEXT_MODEL_PATH.exists()


def _is_image_model_ready() -> bool:
    try:
        resolve_image_model_path()
        return True
    except FileNotFoundError:
        return False


def _truth_confidence(label: str, confidence: float) -> float:
    return confidence if str(label).strip().lower() == "real" else 1.0 - confidence


def _predict_image_with_loaded_model(model, image_path: Path) -> tuple[str, float]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        
        # The CIFAKE dataset consists of 32x32 images. Uploading a high-res image 
        # causes out-of-distribution high-frequency textures that the model interprets as "Fake". 
        # We artificially degrade the uploaded image to 32x32 before upscaling to 128x128 
        # to match the training dataset's structural fidelity.
        img_low_res = img.resize((32, 32), Image.BILINEAR)
        img = img_low_res.resize((128, 128), Image.BILINEAR)
        
        image_arr = np.asarray(img, dtype=np.float32)

    # Keep preprocessing aligned with MobileNetV2 transfer model
    image_arr = np.clip(image_arr, 0.0, 255.0)
    image_arr = (image_arr / 127.5) - 1.0
    image_arr = image_arr[np.newaxis, ...]

    raw = float(model.predict(image_arr, verbose=0)[0][0])
    if raw >= 0.5:
        return "real", raw
    return "fake", 1.0 - raw


def _save_uploaded_image(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _run_prediction(text: str | None, uploaded_image):
    text_result = None
    image_result = None
    warnings: list[str] = []

    if text and text.strip():
        if _is_text_model_ready():
            text_model = get_text_model()
            label, confidence = predict_text(text=text.strip(), model=text_model)
            text_label_mapped = "Real" if str(label) == "1" else "Fake" if str(label) == "0" else label.title()
            text_result = {"label": text_label_mapped, "confidence": float(confidence)}
        else:
            warnings.append("Text model missing at truthmatrix/models/model.pkl")

    temp_image_path: Path | None = None
    if uploaded_image is not None:
        if _is_image_model_ready():
            image_model = get_image_model()
            temp_image_path = _save_uploaded_image(uploaded_image)
            try:
                label, confidence = _predict_image_with_loaded_model(image_model, temp_image_path)
                image_label_mapped = "Real" if str(label) == "1" else "Fake" if str(label) == "0" else label.title()
                image_result = {"label": image_label_mapped, "confidence": float(confidence)}
            finally:
                if temp_image_path.exists():
                    temp_image_path.unlink(missing_ok=True)
        else:
            warnings.append("Image model missing. Train and save image_model.keras or image_model.h5")

    if text_result is None and image_result is None:
        return None, warnings

    if text_result is not None and image_result is not None:
        fusion = fuse_predictions(text_result, image_result)
        final_label = fusion["label"]
        final_confidence = float(fusion["confidence"])
        source = "fusion"

        text_truth = _truth_confidence(text_result["label"], text_result["confidence"])
        image_truth = _truth_confidence(image_result["label"], image_result["confidence"])
        fusion_truth = _truth_confidence(final_label, final_confidence)
        truth_score, risk_level, _ = calculate_truth_score(text_truth, image_truth, fusion_truth)
    elif text_result is not None:
        final_label = text_result["label"]
        final_confidence = float(text_result["confidence"])
        source = "text"
        text_truth = _truth_confidence(text_result["label"], text_result["confidence"])
        truth_score, risk_level, _ = calculate_truth_score(text_truth, text_truth, text_truth)
    else:
        final_label = image_result["label"]
        final_confidence = float(image_result["confidence"])
        source = "image"
        image_truth = _truth_confidence(image_result["label"], image_result["confidence"])
        truth_score, risk_level, _ = calculate_truth_score(image_truth, image_truth, image_truth)

    credibility_score = calculate_credibility_score(
        text=text or "",
        text_confidence=text_result["confidence"] if text_result else 0.5,
        image_confidence=image_result["confidence"] if image_result else 0.5,
        fusion_confidence=final_confidence,
    )

    explanations = generate_explanation(text or "", final_label)
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label": final_label,
        "confidence": final_confidence,
        "truth_score": round(float(truth_score), 2),
        "credibility_score": round(float(credibility_score), 2),
        "risk_level": risk_level,
        "source": source,
        "text": text_result,
        "image": image_result,
        "explanations": explanations,
    }
    return payload, warnings


def _score_card(title: str, value: str, tone: str = "neutral"):
    tone_map = {
        "good": "linear-gradient(135deg, #059669 0%, #10b981 100%)",
        "warn": "linear-gradient(135deg, #d97706 0%, #f59e0b 100%)",
        "bad": "linear-gradient(135deg, #dc2626 0%, #ef4444 100%)",
        "neutral": "linear-gradient(135deg, #475569 0%, #64748b 100%)",
    }
    bg = tone_map.get(tone, tone_map["neutral"])
    st.markdown(
        f"""
        <div style=\"background:{bg};color:white;padding:1.2rem;border-radius:12px;box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 1rem;\">
            <div style=\"font-size:0.85rem;letter-spacing:0.05em;opacity:0.9;text-transform:uppercase;font-weight:600;\">{title}</div>
            <div style=\"font-size:2.2rem;font-weight:800;margin-top:0.2rem;\">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
    /* Prevent hardcoded background colors that conflict with Streamlit themes */
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px;}
    .hero {
      background: linear-gradient(135deg, #1e293b 0%, #0f766e 100%);
      color: #ffffff;
      border-radius: 12px;
      padding: 2rem 2.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .hero h1 {margin: 0; font-size: 2.8rem; font-weight: 800; line-height: 1.2;}
    .hero p {margin: 0.8rem 0 1.2rem 0; opacity: 0.95; font-size: 1.15rem;}
    .status-chip {
      display:inline-block;
      padding:0.35rem 0.8rem;
      border-radius:999px;
      font-size:0.85rem;
      font-weight: 500;
      margin-right:0.75rem;
      background: rgba(255,255,255,0.15);
      border:1px solid rgba(255,255,255,0.3);
      backdrop-filter: blur(4px);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

text_status = "Ready" if _is_text_model_ready() else "Missing"
image_status = "Ready" if _is_image_model_ready() else "Missing"

st.markdown(
    f"""
    <div class="hero">
      <h1>TruthMatrix</h1>
      <p>Multimodal fake content detection using text + image fusion</p>
      <div style="margin-top:0.7rem;">
        <span class="status-chip">Text model: {text_status}</span>
        <span class="status-chip">Image model: {image_status}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("System")
    st.write("Model directory:")
    st.code(str(MODEL_DIR))
    st.caption("Train models first if status shows Missing.")
    
    st.divider()
    st.info("⚠️ **Demo Warning:** The image model is currently trained on a *mock placeholder dataset*. For accurate Real vs AI face detection, please place real images in `data/images/real` and `data/images/fake` and retrain.")


tab_detect, tab_analytics, tab_insights = st.tabs(["Detection", "Analytics", "Model Insights"])

with tab_detect:
    c1, c2 = st.columns([1.5, 1])

    with c1:
        text_input = st.text_area(
            "Text input",
            placeholder="Paste a headline, post, or article content...",
            height=220,
        )

    with c2:
        uploaded_image = st.file_uploader(
            "Optional image evidence",
            type=["png", "jpg", "jpeg", "webp"],
        )
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if analyze_clicked:
        if not text_input.strip() and uploaded_image is None:
            st.warning("Provide text, image, or both before analysis.")
        else:
            with st.spinner("Running multimodal analysis..."):
                result, warnings = _run_prediction(text_input, uploaded_image)

            for warning in warnings:
                st.warning(warning)

            if result is None:
                st.error("Prediction could not be generated. Check model files and inputs.")
            else:
                st.session_state.history.append(result)

                tone = "good" if result["label"] == "Real" else "bad"
                c_a, c_b, c_c, c_d = st.columns(4)
                with c_a:
                    _score_card("Verdict", result["label"], tone)
                with c_b:
                    _score_card("Confidence", f"{result['confidence']:.1%}", "neutral")
                with c_c:
                    risk_tone = "good" if result["risk_level"] == "Low" else "warn" if result["risk_level"] == "Medium" else "bad"
                    _score_card("Risk", result["risk_level"], risk_tone)
                with c_d:
                    _score_card("Source", result["source"].title(), "neutral")

                st.markdown("#### Scores")
                s1, s2 = st.columns(2)
                with s1:
                    st.progress(min(max(result["truth_score"] / 100.0, 0.0), 1.0), text=f"Truth Score: {result['truth_score']:.1f}/100")
                with s2:
                    st.progress(min(max(result["credibility_score"] / 100.0, 0.0), 1.0), text=f"Credibility Score: {result['credibility_score']:.1f}/100")

                with st.expander("Why this result?", expanded=True):
                    for reason in result["explanations"]:
                        st.markdown(f"- {reason}")

                with st.expander("Technical details"):
                    st.json(result)

with tab_analytics:
    history = st.session_state.history
    if not history:
        st.info("No runs yet. Analyze content in the Detection tab.")
    else:
        labels = [row["label"] for row in history]
        truth_scores = [row["truth_score"] for row in history]
        confidence_scores = [row["confidence"] * 100 for row in history]

        real_count = labels.count("Real")
        fake_count = labels.count("Fake")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Runs", len(history))
        m2.metric("Real", real_count)
        m3.metric("Fake", fake_count)

        left, right = st.columns(2)
        
        # Prepare DataFrames for Altair
        df_trend = pd.DataFrame({
            "Run": range(1, len(truth_scores) + 1),
            "Truth Score": truth_scores
        })
        
        df_dist = pd.DataFrame({
            "Confidence": confidence_scores
        })

        with left:
            st.markdown("#### Truth Score Trend")
            c = alt.Chart(df_trend).mark_line(point=True, color="#0ea5e9", strokeWidth=3).encode(
                x=alt.X('Run:O', title='Run #'),
                y=alt.Y('Truth Score:Q', scale=alt.Scale(domain=[0, 100]), title='Score'),
                tooltip=['Run', 'Truth Score']
            ).properties(height=320)
            st.altair_chart(c, use_container_width=True)

        with right:
            st.markdown("#### Confidence Distribution")
            hist = alt.Chart(df_dist).mark_bar(color="#8b5cf6", opacity=0.8, binSpacing=1).encode(
                x=alt.X('Confidence:Q', bin=alt.Bin(maxbins=10), title='Confidence (%)'),
                y=alt.Y('count()', title='Frequency'),
                tooltip=['count()', 'Confidence']
            ).properties(height=320)
            st.altair_chart(hist, use_container_width=True)

        st.markdown("#### Run History")
        rows = []
        for idx, row in enumerate(history, start=1):
            rows.append(
                {
                    "Run": idx,
                    "Timestamp": row["timestamp"],
                    "Verdict": row["label"],
                    "Confidence": f"{row['confidence']:.1%}",
                    "Truth Score": row["truth_score"],
                    "Credibility": row["credibility_score"],
                    "Risk": row["risk_level"],
                    "Source": row["source"],
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_insights:
    st.markdown("### Pipeline Overview")
    st.markdown(
        """
        - Text model: Classical ML classifier on engineered text features.
        - Image model: MobileNetV2 transfer learning binary classifier.
        - Fusion: Confidence-aware agreement strategy.
        - Truth score: weighted confidence converted to 0-100.
        - Credibility score: rule-based textual trust cues with confidence signals.
        """
    )

    st.markdown("### Health Check")
    hc1, hc2 = st.columns(2)
    with hc1:
        st.success("Text model is ready") if _is_text_model_ready() else st.error("Text model missing")
    with hc2:
        st.success("Image model is ready") if _is_image_model_ready() else st.error("Image model missing")

    st.markdown("### Training Commands")
    st.code(
        """python src/image/train.py --image-dir data/images --epochs 10"""
    )