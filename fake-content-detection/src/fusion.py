from __future__ import annotations


def multimodal_fusion(text_probability: float, image_probability: float) -> tuple[str, float]:
    if not 0.0 <= text_probability <= 1.0:
        raise ValueError("text_probability must be between 0 and 1")

    if not 0.0 <= image_probability <= 1.0:
        raise ValueError("image_probability must be between 0 and 1")

    final_score = (text_probability + image_probability) / 2

    if final_score > 0.5:
        return "Fake", final_score

    return "Real", final_score