from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Ensure `src` imports work when running: streamlit run app/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.append(str(PROJECT_ROOT))

from src.fusion.fusion import calculate_truth_score
from src.fusion.pipeline import unified_predict
from src.explain import generate_explanation, calculate_credibility_score


def _normalize_label(label: str | None) -> str:
	if not label:
		return "N/A"
	lowered = str(label).strip().lower()
	if lowered in {"fake", "false", "0"}:
		return "Fake"
	if lowered in {"real", "true", "1"}:
		return "Real"
	return str(label)


def _init_session_state() -> None:
	"""Initialize session state for tracking predictions."""
	if "predictions_history" not in st.session_state:
		st.session_state.predictions_history = []


def _show_prediction_block(title: str, prediction: Dict[str, Any] | None) -> None:
	"""Display a single prediction in a bordered container with progress bar."""
	with st.container(border=True):
		st.subheader(title)

		if not prediction:
			st.write("No prediction available")
			st.progress(0.0)
			return

		label = _normalize_label(prediction.get("label"))
		confidence = float(prediction.get("confidence", 0.0) or 0.0)
		confidence = max(0.0, min(confidence, 1.0))

		st.write(f"Label: {label}")
		st.write(f"Confidence: {confidence:.2%}")
		st.progress(confidence)


def _plot_fake_vs_real_pie() -> None:
	"""Plot pie chart for Fake vs Real distribution."""
	history = st.session_state.predictions_history
	if not history:
		st.warning("No predictions yet. Run detections to populate analytics.")
		return

	fake_count = sum(1 for p in history if p["final_label"] == "Fake")
	real_count = sum(1 for p in history if p["final_label"] == "Real")
	total = fake_count + real_count

	fig, ax = plt.subplots(figsize=(8, 6))
	labels = ["Fake", "Real"]
	sizes = [fake_count, real_count]
	colors = ["#ff6b6b", "#51cf66"]
	ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, textprops={"fontsize": 12})
	ax.set_title("Fake vs Real Distribution", fontsize=14, fontweight="bold")
	st.pyplot(fig)


def _plot_accuracy_comparison() -> None:
	"""Plot bar chart for model accuracy comparison."""
	history = st.session_state.predictions_history
	if not history:
		st.warning("No predictions yet. Run detections to populate analytics.")
		return

	text_correct = sum(
		1 for p in history 
		if p.get("text_label") == p.get("final_label")
	)
	image_correct = sum(
		1 for p in history 
		if p.get("image_label") and p.get("image_label") == p.get("final_label")
	)
	fusion_correct = sum(
		1 for p in history 
		if p.get("final_label") is not None
	)

	text_total = sum(1 for p in history if p.get("text_label"))
	image_total = sum(1 for p in history if p.get("image_label"))
	fusion_total = len(history)

	text_acc = (text_correct / text_total * 100) if text_total > 0 else 0
	image_acc = (image_correct / image_total * 100) if image_total > 0 else 0
	fusion_acc = (fusion_correct / fusion_total * 100) if fusion_total > 0 else 0

	fig, ax = plt.subplots(figsize=(10, 6))
	models = ["Text Model", "Image Model", "Fusion Model"]
	accuracies = [text_acc, image_acc, fusion_acc]
	colors_bars = ["#4c72b0", "#dd8452", "#55a868"]
	bars = ax.bar(models, accuracies, color=colors_bars, alpha=0.7, edgecolor="black", linewidth=1.5)
	ax.set_ylabel("Accuracy (%)", fontsize=12)
	ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
	ax.set_ylim(0, 105)
	ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
	for bar, acc in zip(bars, accuracies):
		height = bar.get_height()
		ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")
	st.pyplot(fig)


def _plot_confusion_matrix() -> None:
	"""Plot confusion matrix heatmap."""
	history = st.session_state.predictions_history
	if not history:
		st.warning("No predictions yet. Run detections to populate analytics.")
		return

	cm = np.zeros((2, 2), dtype=int)
	labels_map = {"Fake": 0, "Real": 1}
	for p in history:
		predicted = labels_map.get(p.get("final_label"))
		if predicted is not None:
			cm[predicted, predicted] += 1

	fig, ax = plt.subplots(figsize=(8, 6))
	im = ax.imshow(cm, cmap="Blues", aspect="auto")
	cbar = plt.colorbar(im, ax=ax)
	cbar.set_label("Count", rotation=270, labelpad=20)

	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(["Fake", "Real"], fontsize=11)
	ax.set_yticklabels(["Fake", "Real"], fontsize=11)
	ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
	ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
	ax.set_title("Confusion Matrix (Diagonal Shows Agreement)", fontsize=14, fontweight="bold")

	for i in range(2):
		for j in range(2):
			text = ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")

	st.pyplot(fig)


def _detection_tab() -> None:
	"""Detection tab: text/image input and prediction."""
	st.header("Input")
	input_col1, input_col2 = st.columns(2)

	with input_col1:
		input_text = st.text_area(
			"Text Input",
			placeholder="Paste or type content to analyze...",
			height=180,
		)

	with input_col2:
		uploaded_image = st.file_uploader(
			"Image Upload",
			type=["png", "jpg", "jpeg", "webp"],
			accept_multiple_files=False,
		)
		if uploaded_image is not None:
			st.image(uploaded_image, caption="Uploaded image preview", use_container_width=True)

	analyze = st.button("Analyze Content", type="primary", use_container_width=True)

	if not analyze:
		return

	if not input_text.strip() and uploaded_image is None:
		st.warning("Provide text, an image, or both before running analysis.")
		return

	temp_image_path: str | None = None
	try:
		if uploaded_image is not None:
			suffix = Path(uploaded_image.name).suffix or ".jpg"
			with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
				temp_file.write(uploaded_image.getvalue())
				temp_image_path = temp_file.name

		with st.spinner("Running text, image, and fusion predictions..."):
			result = unified_predict(
				text=input_text if input_text.strip() else None,
				image_path=temp_image_path,
			)

		st.header("Predictions")
		pred_col1, pred_col2, pred_col3 = st.columns(3)

		with pred_col1:
			_show_prediction_block("Text Prediction", result.get("text"))

		with pred_col2:
			_show_prediction_block("Image Prediction", result.get("image"))

		with pred_col3:
			_show_prediction_block("Final Fusion Result", result.get("final"))

		final = result.get("final") or {}
		final_label = _normalize_label(final.get("label"))
		final_confidence = float(final.get("confidence", 0.0) or 0.0)

		if final_label == "Real":
			st.success(f"Final Result: {final_label} ({final_confidence:.2%} confidence)")
		elif final_label == "Fake":
			st.error(f"Final Result: {final_label} ({final_confidence:.2%} confidence)")
		else:
			st.warning("Final result is unavailable.")

		text_conf = float((result.get("text") or {}).get("confidence", 0.0) or 0.0)
		image_conf = float((result.get("image") or {}).get("confidence", 0.0) or 0.0)
		fusion_conf = float((result.get("fusion") or {}).get("confidence", final_confidence) or 0.0)

		truth_score, risk_level, _ = calculate_truth_score(
			text_confidence=text_conf,
			image_confidence=image_conf,
			fusion_confidence=fusion_conf,
		)

		credibility_score = calculate_credibility_score(
			text=input_text,
			text_confidence=text_conf,
			image_confidence=image_conf,
			fusion_confidence=fusion_conf,
		)

		st.header("📊 Score & Risk Analysis")
		score_col, cred_col, risk_col = st.columns(3)

		with score_col:
			st.markdown(
				f"""
				<div class="tm-card">
				  <div class="tm-label">Truth Score</div>
				  <div class="tm-value">{truth_score:.2f} / 100</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		with cred_col:
			st.markdown(
				f"""
				<div class="tm-card">
				  <div class="tm-label">Credibility Score</div>
				  <div class="tm-value">{credibility_score:.2f} / 100</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		with risk_col:
			st.markdown(
				f"""
				<div class="tm-card">
				  <div class="tm-label">Risk Level</div>
				  <div class="tm-value">{risk_level}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		st.subheader("🧠 Why this result?")
		reasons = generate_explanation(
			text=input_text,
			prediction_label=final_label,
		)
		for reason in reasons:
			st.markdown(f"- {reason}")

		warnings = result.get("warnings") or []
		for warning in warnings:
			st.info(str(warning))

		text_label = _normalize_label((result.get("text") or {}).get("label"))
		image_label = _normalize_label((result.get("image") or {}).get("label"))
		st.session_state.predictions_history.append({
			"text_label": text_label,
			"image_label": image_label,
			"final_label": final_label,
			"final_confidence": final_confidence,
			"truth_score": truth_score,
			"credibility_score": credibility_score,
			"risk_level": risk_level,
		})

	except Exception as exc:
		st.exception(exc)
	finally:
		if temp_image_path and os.path.exists(temp_image_path):
			os.remove(temp_image_path)


def _analytics_tab() -> None:
	"""Analytics tab: visualizations of predictions."""
	st.header("Analytics & Insights")

	if not st.session_state.predictions_history:
		st.info("No predictions recorded yet. Go to the Detection tab and analyze content to populate analytics.")
		return

	col1, col2 = st.columns(2)

	with col1:
		st.subheader("Fake vs Real Distribution")
		_plot_fake_vs_real_pie()

	with col2:
		st.subheader("Model Accuracy Comparison")
		_plot_accuracy_comparison()

	st.subheader("Confusion Matrix")
	_plot_confusion_matrix()

	st.subheader("Summary Statistics")
	total_predictions = len(st.session_state.predictions_history)
	avg_confidence = np.mean([p["final_confidence"] for p in st.session_state.predictions_history])
	avg_truth_score = np.mean([p["truth_score"] for p in st.session_state.predictions_history])
	avg_credibility_score = np.mean([p["credibility_score"] for p in st.session_state.predictions_history])

	stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
	with stat_col1:
		st.metric("Total Predictions", total_predictions)
	with stat_col2:
		st.metric("Avg Confidence", f"{avg_confidence:.2%}")
	with stat_col3:
		st.metric("Avg Truth Score", f"{avg_truth_score:.2f}/100")
	with stat_col4:
		st.metric("Avg Credibility", f"{avg_credibility_score:.2f}/100")


def _model_insights_tab() -> None:
	"""Model Insights tab: information about models and methodology."""
	st.header("Model Architecture & Insights")

	st.subheader("Text Model")
	st.write("""
		- **Type:** Logistic Regression Classifier
		- **Features:** Text length, word count, uppercase ratio
		- **Input:** User-provided text content
		- **Output:** Fake probability score [0, 1]
	""")

	st.subheader("Image Model")
	st.write("""
		- **Type:** Convolutional Neural Network (CNN)
		- **Architecture:** Deep learning model trained on image features
		- **Input:** Uploaded image (resized to 128x128)
		- **Output:** Real/Fake classification with confidence
	""")

	st.subheader("Fusion Model")
	st.write("""
		- **Strategy:** Multi-modal fusion
		- **Logic:** If text and image labels agree → average confidence
		- **Logic:** If they disagree → choose label with higher confidence
		- **Output:** Fused Fake/Real decision with combined confidence
	""")

	st.subheader("Truth Score Calculation")
	st.write("""
		- **Formula:** (0.4 × text_conf + 0.4 × image_conf + 0.2 × fusion_conf) × 100
		- **Score Range:** 0–100
		- **Risk Level:** 
		  - Low: score > 75
		  - Medium: 50–75
		  - High: score < 50
	""")

	st.subheader("Credibility Score Calculation")
	st.write("""
		- **Factors considered:**
		  - Sentiment polarity (neutral is more credible)
		  - Presence of exaggerated/sensational words (fewer is better)
		  - Model prediction confidence (higher is more credible)
		  - Text structure and length (well-structured is more credible)
		  - Capitalization patterns (excessive caps reduces credibility)
		  - Punctuation patterns (excessive punctuation reduces credibility)
		- **Score Range:** 0–100
		- **Interpretation:**
		  - 80–100: High credibility (balanced, structured, confident)
		  - 60–79: Medium credibility (some emotional cues present)
		  - 40–59: Low credibility (sensational style, emotional)
		  - 0–39: Very low credibility (highly exaggerated or short)
	""")

	st.subheader("Explanation Engine")
	st.write("""
		- Detects sensational/clickbait keywords
		- Analyzes sentiment cues (positive/negative words)
		- Checks text length, capitalization, and punctuation patterns
		- Identifies mismatches between writing style and predicted class
	""")


def main() -> None:
	"""Main app with tabbed interface."""
	_init_session_state()

	st.set_page_config(page_title="TruthMatrix - Fake Content Detection", layout="wide")
	st.title("TruthMatrix: Fake Content Detection")
	st.caption("Multimodal detection system combining text and image analysis.")

	st.markdown(
		"""
		<style>
		  .tm-card {
			border: 1px solid #d7dee7;
			border-radius: 12px;
			padding: 14px;
			background: linear-gradient(180deg, #f8fbff 0%, #f3f8fc 100%);
		  }
		  .tm-label {
			font-size: 0.85rem;
			color: #4d5b6a;
			margin-bottom: 0.2rem;
		  }
		  .tm-value {
			font-size: 1.4rem;
			font-weight: 700;
			color: #16324a;
		  }
		</style>
		""",
		unsafe_allow_html=True,
	)

	tab1, tab2, tab3 = st.tabs(["Detection", "Analytics", "Model Insights"])

	with tab1:
		_detection_tab()

	with tab2:
		_analytics_tab()

	with tab3:
		_model_insights_tab()


if __name__ == "__main__":
	main()
