from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from src.fusion.fusion import fuse_predictions
from src.image.predict import predict_image
from src.predict import predict as predict_text
from src.train import load_model as load_text_model


def _to_fake_probability(
    label: str,
    confidence: float,
    fake_aliases: Iterable[str] = ("fake", "false", "0"),
    real_aliases: Iterable[str] = ("real", "true", "1"),
) -> float:
    """Convert (label, confidence) into fake-class probability."""
    label_norm = str(label).strip().lower()

    if label_norm in set(fake_aliases):
        return float(confidence)

    if label_norm in set(real_aliases):
        return 1.0 - float(confidence)

    raise ValueError(
        f"Unsupported label '{label}'. Expected one of fake aliases {tuple(fake_aliases)} "
        f"or real aliases {tuple(real_aliases)}."
    )


def unified_predict(
    text: str | None = None,
    image_path: str | Path | None = None,
    text_model_path: str | Path = Path(__file__).resolve().parents[2] / "models" / "model.pkl",
    image_model_path: str | Path = Path(__file__).resolve().parents[2] / "models" / "image_model.h5",
    text_weight: float = 0.5,
    image_weight: float = 0.5,
) -> Dict[str, Any]:
    """Run text/image predictions and fuse results when both inputs are provided.

    Behavior:
    - If both text and image are available, returns fused Fake/Real output.
    - If only one modality is available, returns that modality output.
    - If neither input is valid, returns a graceful error payload.
    """
    result: Dict[str, Any] = {
        "text": None,
        "image": None,
        "fusion": None,
        "final": None,
        "used_modalities": [],
        "warnings": [],
    }

    text_score: float | None = None
    image_score: float | None = None

    if text is not None and str(text).strip():
        try:
            model = load_text_model(text_model_path)
            text_label, text_conf = predict_text(text=str(text), model=model)
            text_score = _to_fake_probability(text_label, text_conf)
            result["text"] = {
                "label": str(text_label),
                "confidence": float(text_conf),
                "fake_score": float(text_score),
            }
            result["used_modalities"].append("text")
        except Exception as exc:
            result["warnings"].append(f"Text prediction failed: {exc}")

    if image_path is not None and str(image_path).strip():
        try:
            image_label, image_conf = predict_image(
                image_path=image_path,
                model_path=image_model_path,
            )
            image_score = _to_fake_probability(image_label, image_conf)
            result["image"] = {
                "label": str(image_label),
                "confidence": float(image_conf),
                "fake_score": float(image_score),
            }
            result["used_modalities"].append("image")
        except Exception as exc:
            result["warnings"].append(f"Image prediction failed: {exc}")

    used_count = len(result["used_modalities"])

    if used_count == 2 and text_score is not None and image_score is not None:
        final_label, final_conf = fuse_predictions(
            text_score=text_score,
            image_score=image_score,
            text_weight=text_weight,
            image_weight=image_weight,
        )
        result["fusion"] = {
            "text_weight": float(text_weight),
            "image_weight": float(image_weight),
            "label": final_label,
            "confidence": float(final_conf),
        }
        result["final"] = {
            "label": final_label,
            "confidence": float(final_conf),
            "source": "fusion",
        }
        return result

    if used_count == 1:
        if "text" in result["used_modalities"] and result["text"] is not None:
            result["final"] = {
                "label": "Fake" if result["text"]["fake_score"] >= 0.5 else "Real",
                "confidence": float(
                    result["text"]["fake_score"]
                    if result["text"]["fake_score"] >= 0.5
                    else 1.0 - result["text"]["fake_score"]
                ),
                "source": "text",
            }
        elif "image" in result["used_modalities"] and result["image"] is not None:
            result["final"] = {
                "label": "Fake" if result["image"]["fake_score"] >= 0.5 else "Real",
                "confidence": float(
                    result["image"]["fake_score"]
                    if result["image"]["fake_score"] >= 0.5
                    else 1.0 - result["image"]["fake_score"]
                ),
                "source": "image",
            }

        result["warnings"].append(
            "Only one modality was used. Fusion requires both text and image inputs."
        )
        return result

    result["final"] = {
        "label": None,
        "confidence": None,
        "source": None,
    }
    result["warnings"].append(
        "No valid input provided. Pass text and/or image_path to generate a prediction."
    )
    return result
