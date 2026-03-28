from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def load_image_datasets(
    image_dir: str | Path,
    image_size: tuple[int, int] = (128, 128),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train/validation datasets from directory with class subfolders.

    Expected folder structure:
    - image_dir/real
    - image_dir/fake

    Returns:
    - train_ds: training dataset
    - val_ds: validation dataset
    """
    root = Path(image_dir)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    required_folders = [root / "fake", root / "real"]
    missing_folders = [str(folder) for folder in required_folders if not folder.exists()]
    if missing_folders:
        raise FileNotFoundError(
            "Image dataset must include both class folders: 'fake' and 'real'. "
            f"Missing: {', '.join(missing_folders)}"
        )

    fake_count = len(list((root / "fake").glob("*")))
    real_count = len(list((root / "real").glob("*")))
    if fake_count == 0 or real_count == 0:
        raise ValueError(
            "Both class folders must contain at least one image. "
            f"Found fake={fake_count}, real={real_count}."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(root),
        labels="inferred",
        label_mode="binary",
        class_names=["fake", "real"],
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(root),
        labels="inferred",
        label_mode="binary",
        class_names=["fake", "real"],
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)

    return train_ds, val_ds
