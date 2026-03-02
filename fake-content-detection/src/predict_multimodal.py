from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

from fusion import multimodal_fusion
from project_utils import validate_model_artifacts


def preprocess_text(text: str) -> str:
    return text.strip().lower()


def preprocess_image(image_path: Path, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    loaded_image = keras_image.load_img(image_path, target_size=target_size)
    image_array = keras_image.img_to_array(loaded_image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)


def load_models(model_dir: Path):
    validate_model_artifacts(model_dir)
    text_model = joblib.load(model_dir / "text_model.pkl")
    tfidf_vectorizer = joblib.load(model_dir / "tfidf_vectorizer.pkl")
    image_model = tf.keras.models.load_model(model_dir / "image_model.h5")
    return text_model, tfidf_vectorizer, image_model


def predict_text_probability(text_model, tfidf_vectorizer, text: str) -> float:
    processed_text = preprocess_text(text)
    text_features = tfidf_vectorizer.transform([processed_text])
    probability = text_model.predict_proba(text_features)[0][1]
    return float(probability)


def predict_image_probability(image_model, image_path: Path) -> float:
    processed_image = preprocess_image(image_path)
    probability = image_model.predict(processed_image, verbose=0)[0][0]
    return float(probability)


def run_prediction(text_input: str, image_path: Path, model_dir: Path) -> tuple[float, float, str, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    text_model, tfidf_vectorizer, image_model = load_models(model_dir)

    text_probability = predict_text_probability(text_model, tfidf_vectorizer, text_input)
    image_probability = predict_image_probability(image_model, image_path)
    final_label, final_score = multimodal_fusion(text_probability, image_probability)

    return text_probability, image_probability, final_label, final_score


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run multimodal fake content prediction.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=project_root / "models",
        help="Directory containing text_model.pkl, tfidf_vectorizer.pkl, and image_model.h5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    text_input = input("Enter news text: ").strip()
    image_input = input("Enter image path: ").strip()
    image_path = Path(image_input)

    try:
        text_probability, image_probability, final_label, final_score = run_prediction(
            text_input=text_input,
            image_path=image_path,
            model_dir=args.model_dir,
        )
    except FileNotFoundError as error:
        print(f"Error: {error}")
        return
    except Exception as error:
        print(f"Prediction failed: {error}")
        return

    print(f"Text probability (fake): {text_probability:.4f}")
    print(f"Image probability (fake): {image_probability:.4f}")
    print(f"Final fused decision: {final_label} (score: {final_score:.4f})")


if __name__ == "__main__":
    main()