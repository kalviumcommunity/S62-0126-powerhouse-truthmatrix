from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def load_saved_image_model(model_path: str | Path) -> tf.keras.Model:
    """Load a saved Keras CNN model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return tf.keras.models.load_model(path)


def _preprocess_image(image_path: str | Path, target_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Load image with PIL, resize, normalize, and add batch dimension."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(target_size)
        image_array = np.asarray(img, dtype=np.float32) / 255.0

    return np.expand_dims(image_array, axis=0)


def predict_image(
    image_path: str | Path,
    model_path: str | Path = Path(__file__).resolve().parents[2] / "models" / "image_model.h5",
    threshold: float = 0.5,
) -> Tuple[str, float]:
    """Predict fake vs real from a single image.

    Returns:
    - label: "fake" or "real"
    - confidence: confidence score for the predicted label in [0, 1]
    """
    model = load_saved_image_model(model_path)
    image_batch = _preprocess_image(image_path=image_path, target_size=(128, 128))

    prediction = model.predict(image_batch, verbose=0)
    real_probability = float(prediction[0][0])

    if real_probability >= threshold:
        return "real", real_probability

    return "fake", 1.0 - real_probability
