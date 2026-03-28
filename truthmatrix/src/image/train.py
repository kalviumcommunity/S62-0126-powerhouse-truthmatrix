from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.image.data_loader import load_image_datasets
from src.image.model import build_cnn_model
from src.image.preprocess import apply_preprocessing_to_dataset, build_preprocessing_pipeline


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

    class_weights = {0: 1.0, 1: 1.0}
    fake_folder = Path(image_dir) / "fake"
    real_folder = Path(image_dir) / "real"
    fake_count = len(list(fake_folder.glob("*")))
    real_count = len(list(real_folder.glob("*")))
    if fake_count > 0 and real_count > 0:
        total = fake_count + real_count
        class_weights = {
            0: total / (2 * fake_count),
            1: total / (2 * real_count),
        }

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"[INFO] Training model for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    project_root = Path(__file__).resolve().parents[2]
    model_dir = project_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "image_model.keras"

    print(f"[INFO] Saving trained model to: {model_path}")
    model.save(model_path)

    compatibility_path = model_dir / "image_model.h5"
    print(f"[INFO] Saving compatibility copy to: {compatibility_path}")
    model.save(compatibility_path)

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
