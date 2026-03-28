from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image


def resolve_image_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve image model path with fallback priority: explicit -> .keras -> .h5."""
    if model_path is not None:
        path = Path(model_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Model file not found: {path}")

    model_dir = Path(__file__).resolve().parents[2] / "models"
    candidates = [model_dir / "image_model.keras", model_dir / "image_model.h5"]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No image model found. Expected one of: {', '.join(str(c) for c in candidates)}"
    )


def load_saved_image_model(model_path: str | Path) -> tf.keras.Model:
    """Load a saved Keras CNN model from disk."""
    path = resolve_image_model_path(model_path)
    return tf.keras.models.load_model(path)


def _preprocess_image(image_path: str | Path, target_size: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Load image with PIL, resize, normalize, and add batch dimension."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        
        # Artificial degradation to match CIFAKE 32x32 dataset distribution
        img_low_res = img.resize((32, 32), Image.BILINEAR)
        img = img_low_res.resize(target_size, Image.BILINEAR)
        
        image_array = np.asarray(img, dtype=np.float32)
        image_array = (image_array / 127.5) - 1.0

    return np.expand_dims(image_array, axis=0)


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> Tuple[str, float]:
    """Predict fake vs real from a single image.

    Returns:
    - label: "fake" or "real"
    - confidence: confidence score for the predicted label in [0, 1]
    """
    model = load_saved_image_model(model_path or resolve_image_model_path())
    image_batch = _preprocess_image(image_path=image_path, target_size=(128, 128))

    prediction = model.predict(image_batch, verbose=0)
    real_probability = float(prediction[0][0])

    if real_probability >= threshold:
        return "real", real_probability

    return "fake", 1.0 - real_probability
