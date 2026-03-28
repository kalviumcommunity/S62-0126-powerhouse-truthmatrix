from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_loader import load_data
from src.evaluate import compare_models
from src.preprocess import add_features, clean_data, prepare_features


def split_data(
	x: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.2,
	random_state: int = 42,
	stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Split features and target into train/test sets."""
	stratify_target = y if stratify else None
	return train_test_split(
		x,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify_target,
	)


def build_model_pipelines(random_state: int = 42) -> Dict[str, ClassifierMixin]:
	"""Build model pipelines with scaling where needed.

	Uses a hybrid feature stack:
	- TF-IDF from raw text
	- Numeric engineered features
	"""
	numeric_features = ["text_length", "word_count", "uppercase_ratio"]

	feature_transformer = ColumnTransformer(
		transformers=[
			(
				"text_tfidf",
				TfidfVectorizer(
					ngram_range=(1, 2),
					max_features=4000,
					min_df=1,
				),
				"text",
			),
			(
				"numeric",
				Pipeline(steps=[("scaler", StandardScaler())]),
				numeric_features,
			),
		],
		remainder="drop",
	)

	models: Dict[str, ClassifierMixin] = {
		"logistic_regression": Pipeline(
			steps=[
				("features", feature_transformer),
				(
					"classifier",
					LogisticRegression(
						max_iter=2000,
						random_state=random_state,
						class_weight="balanced",
					),
				),
			]
		),
		"linear_svm": Pipeline(
			steps=[
				("features", feature_transformer),
				("classifier", LinearSVC(C=1.0, random_state=random_state)),
			]
		),
		"decision_tree": Pipeline(
			steps=[
				("features", feature_transformer),
				("classifier", DecisionTreeClassifier(random_state=random_state)),
			]
		),
		"random_forest": Pipeline(
			steps=[
				("features", feature_transformer),
				(
					"classifier",
					RandomForestClassifier(
						n_estimators=300,
						random_state=random_state,
						class_weight="balanced",
						n_jobs=-1,
					),
				),
			]
		),
	}
	return models


def train_models(
	x: pd.DataFrame,
	y: pd.Series,
	test_size: float = 0.2,
	random_state: int = 42,
	stratify: bool = True,
) -> Tuple[
	Dict[str, ClassifierMixin],
	pd.DataFrame,
	pd.DataFrame,
	pd.Series,
	pd.Series,
]:
	"""Train Logistic Regression, Decision Tree, and KNN models.

	Returns:
	- trained_models: dict of fitted models
	- x_train, x_test, y_train, y_test: split datasets for evaluation
	"""
	x_train, x_test, y_train, y_test = split_data(
		x=x,
		y=y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify,
	)

	models = build_model_pipelines(random_state=random_state)

	for model in models.values():
		model.fit(x_train, y_train)

	return models, x_train, x_test, y_train, y_test


def save_model(model: ClassifierMixin, file_path: str | Path) -> Path:
	"""Save a trained model to disk using pickle."""
	path = Path(file_path).expanduser()
	if not path.is_absolute():
		path = Path.cwd() / path

	path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("wb") as file_obj:
		pickle.dump(model, file_obj)

	return path


def save_best_model(
	models: Dict[str, ClassifierMixin],
	comparison_df: pd.DataFrame,
	file_path: str | Path,
	model_name_col: str = "model",
	metric_col: str = "f1_score",
) -> Tuple[str, Path]:
	"""Select the best model from comparison results and save it with pickle."""
	if comparison_df.empty:
		raise ValueError("comparison_df is empty. Cannot select a best model.")

	if model_name_col not in comparison_df.columns:
		raise ValueError(f"Missing required column in comparison_df: {model_name_col}")

	if metric_col not in comparison_df.columns:
		raise ValueError(f"Missing required metric column in comparison_df: {metric_col}")

	best_row = comparison_df.sort_values(by=metric_col, ascending=False).iloc[0]
	best_model_name = str(best_row[model_name_col])

	if best_model_name not in models:
		raise ValueError(f"Best model '{best_model_name}' is not present in models dictionary.")

	saved_path = save_model(models[best_model_name], file_path=file_path)
	return best_model_name, saved_path


def load_model(file_path: str | Path) -> ClassifierMixin:
	"""Load a pickled model from disk."""
	path = Path(file_path).expanduser()
	if not path.is_absolute():
		path = Path.cwd() / path

	if not path.exists():
		raise FileNotFoundError(f"Model file not found: {path}")

	with path.open("rb") as file_obj:
		model = pickle.load(file_obj)

	return model


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and save the best text model.")
	default_data = Path(__file__).resolve().parents[1] / "data" / "text" / "fake_news.csv"
	default_model = Path(__file__).resolve().parents[1] / "models" / "model.pkl"
	parser.add_argument("--data", type=str, default=str(default_data), help="Path to text CSV dataset.")
	parser.add_argument("--output", type=str, default=str(default_model), help="Path to save best model pickle.")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	df = load_data(args.data)
	if df.empty:
		raise ValueError("Dataset is empty or could not be loaded.")

	df = clean_data(df)
	df = add_features(df)
	x, y = prepare_features(df)

	models, x_train, x_test, y_train, y_test = train_models(x=x, y=y)
	comparison_df = compare_models(models=models, x_test=x_test, y_test=y_test, print_results=True)
	best_name, saved_path = save_best_model(models, comparison_df, file_path=args.output)

	print(f"\nBest model: {best_name}")
	print(f"Saved to: {saved_path}")


if __name__ == "__main__":
	main()

