from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple


def fuse_predictions(
    text_prediction: Mapping[str, Any],
    image_prediction: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fuse text and image predictions using agreement/confidence rules.

    Args:
        text_prediction: Mapping with keys "label" and "confidence"
        image_prediction: Mapping with keys "label" and "confidence"

    Returns:
        {
            "label": "Fake" or "Real",
            "confidence": fusion_confidence
        }
    """
    def _extract(prediction: Mapping[str, Any], source: str) -> Tuple[str, float]:
        if "label" not in prediction or "confidence" not in prediction:
            raise ValueError(f"{source} must include 'label' and 'confidence' keys")

        raw_label = str(prediction["label"]).strip().lower()
        confidence = float(prediction["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"{source} confidence must be between 0 and 1. Got: {confidence}"
            )

        fake_aliases = {"fake", "false", "0"}
        real_aliases = {"real", "true", "1"}

        if raw_label in fake_aliases:
            return "Fake", confidence
        if raw_label in real_aliases:
            return "Real", confidence

        raise ValueError(
            f"Unsupported {source} label '{prediction['label']}'. "
            "Expected Fake/Real (or supported aliases)."
        )

    text_label, text_conf = _extract(text_prediction, "text_prediction")
    image_label, image_conf = _extract(image_prediction, "image_prediction")

    if text_label == image_label:
        fusion_label = text_label
        fusion_confidence = (text_conf + image_conf) / 2.0
    elif text_conf >= image_conf:
        fusion_label = text_label
        fusion_confidence = text_conf
    else:
        fusion_label = image_label
        fusion_confidence = image_conf

    return {
        "label": fusion_label,
        "confidence": float(fusion_confidence),
    }


def calculate_truth_score(
    text_confidence: float,
    image_confidence: float,
    fusion_confidence: float,
) -> Tuple[float, str, Dict[str, Any]]:
    """Calculate a truth score and risk level from model confidences.

    Args:
        text_confidence: Text confidence in [0, 1]
        image_confidence: Image confidence in [0, 1]
        fusion_confidence: Fusion confidence in [0, 1]

    Returns:
        (truth_score, risk_level, intermediate_values)
        - truth_score: Weighted score in [0, 100]
        - risk_level: "Low", "Medium", or "High"
        - intermediate_values: Dictionary containing weighted components and raw totals
    """
    inputs = {
        "text_confidence": text_confidence,
        "image_confidence": image_confidence,
        "fusion_confidence": fusion_confidence,
    }
    for name, value in inputs.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. Got: {value}")

    weights = {
        "text_weight": 0.4,
        "image_weight": 0.4,
        "fusion_weight": 0.2,
    }

    weighted_text = weights["text_weight"] * float(text_confidence)
    weighted_image = weights["image_weight"] * float(image_confidence)
    weighted_fusion = weights["fusion_weight"] * float(fusion_confidence)

    weighted_sum = weighted_text + weighted_image + weighted_fusion
    truth_score = weighted_sum * 100.0

    if truth_score > 75.0:
        risk_level = "Low"
    elif 50.0 <= truth_score <= 75.0:
        risk_level = "Medium"
    else:
        risk_level = "High"

    intermediate_values: Dict[str, Any] = {
        **inputs,
        **weights,
        "weighted_text": weighted_text,
        "weighted_image": weighted_image,
        "weighted_fusion": weighted_fusion,
        "weighted_sum": weighted_sum,
        "truth_score": truth_score,
        "risk_level": risk_level,
    }

    return truth_score, risk_level, intermediate_values
