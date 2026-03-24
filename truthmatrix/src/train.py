from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


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

	Scaling is applied to Logistic Regression and KNN.
	Decision Tree is used without scaling.
	"""
	models: Dict[str, ClassifierMixin] = {
		"logistic_regression": Pipeline(
			steps=[
				("scaler", StandardScaler()),
				("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
			]
		),
		"decision_tree": DecisionTreeClassifier(random_state=random_state),
		"knn": Pipeline(
			steps=[
				("scaler", StandardScaler()),
				("classifier", KNeighborsClassifier(n_neighbors=5)),
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

