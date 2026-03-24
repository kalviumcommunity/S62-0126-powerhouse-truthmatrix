from __future__ import annotations

from typing import Tuple


def fuse_predictions(
    text_score: float,
    image_score: float,
    text_weight: float = 0.5,
    image_weight: float = 0.5,
    threshold: float = 0.5,
) -> Tuple[str, float]:
    """Fuse text and image fake-probability scores using weighted average.

    Args:
        text_score: Fake probability from text model in [0, 1]
        image_score: Fake probability from image model in [0, 1]
        text_weight: Weight for text score contribution
        image_weight: Weight for image score contribution
        threshold: Classification threshold for final fake probability

    Returns:
        (label, confidence)
        - label: "Fake" or "Real"
        - confidence: Confidence of returned label in [0, 1]
    """
    for name, score in (("text_score", text_score), ("image_score", image_score)):
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1. Got: {score}")

    for name, weight in (("text_weight", text_weight), ("image_weight", image_weight)):
        if weight < 0:
            raise ValueError(f"{name} must be >= 0. Got: {weight}")

    total_weight = text_weight + image_weight
    if total_weight == 0:
        raise ValueError("text_weight and image_weight cannot both be zero")

    # Normalize weights in case they do not sum to 1.
    norm_text_weight = text_weight / total_weight
    norm_image_weight = image_weight / total_weight

    final_fake_score = (text_score * norm_text_weight) + (image_score * norm_image_weight)

    if final_fake_score >= threshold:
        return "Fake", final_fake_score

    return "Real", 1.0 - final_fake_score
