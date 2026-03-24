from __future__ import annotations

import argparse
from pathlib import Path

from data_loader import load_image_datasets
from model import build_cnn_model
from preprocess import apply_preprocessing_to_dataset, build_preprocessing_pipeline


def train_image_model(image_dir: str | Path, epochs: int = 8):
    """Train CNN model for binary image classification and save it to disk."""
    print("[INFO] Starting image model training pipeline...")

    print(f"[INFO] Loading image dataset from: {image_dir}")
    train_ds, val_ds = load_image_datasets(image_dir=image_dir)

    print("[INFO] Building preprocessing pipeline...")
    preprocessing = build_preprocessing_pipeline()

    print("[INFO] Applying preprocessing and augmentation...")
    train_ds = apply_preprocessing_to_dataset(train_ds, preprocessing=preprocessing, training=True)
    val_ds = apply_preprocessing_to_dataset(val_ds, preprocessing=preprocessing, training=False)

    print("[INFO] Building CNN model...")
    model = build_cnn_model(input_shape=(128, 128, 3))
    model.summary()

    print(f"[INFO] Training model for {epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "image_model.h5"

    print(f"[INFO] Saving trained model to: {model_path}")
    model.save(model_path)

    print("[INFO] Training completed successfully.")
    return model, history, model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for fake vs real image classification.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "images"),
        help="Path to image dataset directory containing 'fake' and 'real' folders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs (recommended 5-10).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_image_model(image_dir=args.image_dir, epochs=args.epochs)
