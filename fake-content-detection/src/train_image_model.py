from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from project_utils import DEFAULT_SEED, set_global_seed, validate_dataset_structure


def build_data_generators(
    data_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True,
    )

    validation_generator = datagen.flow_from_directory(
        directory=str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False,
    )

    return train_generator, validation_generator


def build_model(input_shape: tuple[int, int, int] = (224, 224, 3)) -> Model:
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(model: Model, train_generator, validation_generator, epochs: int = 5):
    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    return history


def save_accuracy_plot(history, output_plot_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()


def save_metrics_report(
    history,
    report_path: Path,
    epochs: int,
    seed: int,
    architecture_name: str = "MobileNetV2",
) -> None:
    final_train_accuracy = history.history["accuracy"][-1]
    final_validation_accuracy = history.history["val_accuracy"][-1]

    report_content = (
        f"Model Architecture: {architecture_name}\n"
        f"Seed: {seed}\n"
        f"Number of Epochs: {epochs}\n"
        f"Final Training Accuracy: {final_train_accuracy:.4f}\n"
        f"Final Validation Accuracy: {final_validation_accuracy:.4f}\n"
    )

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_content)


def save_model(model: Model, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)


def run_training(data_dir: Path, output_path: Path, epochs: int, batch_size: int) -> None:
    project_root = Path(__file__).resolve().parents[1]
    set_global_seed(DEFAULT_SEED)
    validate_dataset_structure(project_root)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_generator, validation_generator = build_data_generators(
        data_dir=data_dir,
        image_size=(224, 224),
        batch_size=batch_size,
    )

    expected_classes = {"real", "fake"}
    found_classes = set(train_generator.class_indices.keys())
    if expected_classes != found_classes:
        raise ValueError(
            "Dataset must contain exactly two class folders named 'real' and 'fake'. "
            f"Found classes: {sorted(found_classes)}"
        )

    model = build_model(input_shape=(224, 224, 3))
    history = train_model(
        model=model,
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=epochs,
    )

    outputs_dir = project_root / "outputs"
    reports_dir = project_root / "reports"
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    output_plot_path = outputs_dir / "image_training_plot.png"
    report_path = reports_dir / "image_metrics.txt"

    save_accuracy_plot(history, output_plot_path)
    save_metrics_report(
        history,
        report_path,
        epochs=epochs,
        seed=DEFAULT_SEED,
        architecture_name="MobileNetV2",
    )

    save_model(model, output_path)
    print(f"Model saved to: {output_path}")
    print(f"Training plot saved to: {output_plot_path}")
    print(f"Metrics report saved to: {report_path}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data_dir = project_root / "data" / "images"
    default_output_path = project_root / "models" / "image_model.h5"

    parser = argparse.ArgumentParser(description="Train image classifier for fake/manipulated detection.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to image dataset root containing 'real/' and 'fake/' folders.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=default_output_path,
        help="Output path for saved model file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (recommended 5-10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    args = parser.parse_args()

    if args.epochs < 5 or args.epochs > 10:
        raise ValueError("Epochs must be between 5 and 10 as per project requirement.")

    return args


def main() -> None:
    args = parse_args()
    run_training(
        data_dir=args.data_dir,
        output_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()