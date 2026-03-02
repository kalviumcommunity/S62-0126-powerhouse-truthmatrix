from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from project_utils import DEFAULT_SEED, set_global_seed, validate_dataset_structure


def load_dataset(csv_path: Path) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    if not required_columns.issubset(data.columns):
        missing = required_columns.difference(data.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data[["text", "label"]].copy()
    cleaned = cleaned.dropna(subset=["text", "label"])
    cleaned["text"] = cleaned["text"].astype(str).str.lower()
    cleaned["label"] = cleaned["label"].astype(int)

    invalid_labels = set(cleaned["label"].unique()) - {0, 1}
    if invalid_labels:
        raise ValueError(
            f"Label column must contain only 0 (real) and 1 (fake). Found: {sorted(invalid_labels)}"
        )

    return cleaned


def split_features_and_labels(
    data: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    x = data["text"]
    y = data["label"]
    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def vectorize_text(
    x_train: pd.Series,
    x_test: pd.Series,
    max_features: int = 5000,
) -> Tuple[TfidfVectorizer, object, object]:
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    return vectorizer, x_train_tfidf, x_test_tfidf


def train_classifier(x_train_tfidf, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=DEFAULT_SEED)
    model.fit(x_train_tfidf, y_train)
    return model


def evaluate_model(model: LogisticRegression, x_test_tfidf, y_test: pd.Series) -> tuple[dict, object]:
    predictions = model.predict(x_test_tfidf)

    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="binary", zero_division=0
    )
    report = classification_report(y_test, predictions, digits=4)
    matrix = confusion_matrix(y_test, predictions)

    print("Accuracy Score:", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }
    return metrics, matrix


def save_text_evaluation_artifacts(
    metrics: dict,
    confusion: object,
    reports_dir: Path,
    outputs_dir: Path,
    seed: int,
) -> None:
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    metrics_path = reports_dir / "text_metrics.txt"
    confusion_path = outputs_dir / "text_confusion_matrix.png"

    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        metrics_file.write(f"Seed: {seed}\n")
        metrics_file.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        metrics_file.write(f"Precision: {metrics['precision']:.4f}\n")
        metrics_file.write(f"Recall: {metrics['recall']:.4f}\n")
        metrics_file.write(f"F1-score: {metrics['f1_score']:.4f}\n")

    plt.figure(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=["Real", "Fake"]).plot(
        cmap="Blues", values_format="d"
    )
    plt.title("Text Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()

    print(f"Saved text metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {confusion_path}")


def save_artifacts(model: LogisticRegression, vectorizer: TfidfVectorizer, model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "text_model.pkl")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.pkl")


def run_training(csv_path: Path, model_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    set_global_seed(DEFAULT_SEED)
    validate_dataset_structure(project_root)

    data = load_dataset(csv_path)
    processed_data = preprocess_data(data)

    x_train, x_test, y_train, y_test = split_features_and_labels(processed_data)
    vectorizer, x_train_tfidf, x_test_tfidf = vectorize_text(x_train, x_test, max_features=5000)

    model = train_classifier(x_train_tfidf, y_train)
    metrics, matrix = evaluate_model(model, x_test_tfidf, y_test)

    reports_dir = project_root / "reports"
    outputs_dir = project_root / "outputs"
    save_text_evaluation_artifacts(
        metrics=metrics,
        confusion=matrix,
        reports_dir=reports_dir,
        outputs_dir=outputs_dir,
        seed=DEFAULT_SEED,
    )

    save_artifacts(model, vectorizer, model_dir)

    print(f"\nSaved model to: {model_dir / 'text_model.pkl'}")
    print(f"Saved vectorizer to: {model_dir / 'tfidf_vectorizer.pkl'}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_csv = project_root / "data" / "raw" / "news.csv"
    default_model_dir = project_root / "models"

    parser = argparse.ArgumentParser(description="Train a fake news text detection model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=default_csv,
        help="Path to CSV dataset containing 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=default_model_dir,
        help="Directory where model and vectorizer are saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(csv_path=args.data, model_dir=args.model_dir)


if __name__ == "__main__":
    main()