from __future__ import annotations

import tensorflow as tf


def build_preprocessing_pipeline() -> tf.keras.Sequential:
    """Create preprocessing pipeline for image datasets.

    Includes:
    - Rescaling pixel values to [0, 1]
    - Random horizontal flip
    - Random rotation
    - Random zoom
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.RandomFlip(mode="horizontal"),
            tf.keras.layers.RandomRotation(factor=0.1),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        ],
        name="image_preprocessing",
    )


def apply_preprocessing_to_dataset(
    dataset: tf.data.Dataset,
    preprocessing: tf.keras.Sequential | None = None,
    training: bool = True,
) -> tf.data.Dataset:
    """Apply preprocessing pipeline to an image dataset.

    Args:
        dataset: Dataset yielding (images, labels)
        preprocessing: Optional custom preprocessing model
        training: If True, augmentation is active; if False, augmentation layers run in inference mode
    """
    preprocessing = preprocessing or build_preprocessing_pipeline()

    return dataset.map(
        lambda images, labels: (preprocessing(images, training=training), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)
