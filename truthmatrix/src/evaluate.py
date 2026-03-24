from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_model(
	model: ClassifierMixin,
	x_test: pd.DataFrame,
	y_test: pd.Series,
	average: str = "weighted",
) -> Dict[str, Any]:
	"""Evaluate a single classification model and return key metrics.

	Returns a dictionary containing:
	- accuracy
	- precision
	- recall
	- f1_score
	- confusion_matrix
	"""
	y_pred = model.predict(x_test)

	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test,
		y_pred,
		average=average,
		zero_division=0,
	)

	results: Dict[str, Any] = {
		"accuracy": float(accuracy_score(y_test, y_pred)),
		"precision": float(precision),
		"recall": float(recall),
		"f1_score": float(f1),
		"confusion_matrix": confusion_matrix(y_test, y_pred),
	}
	return results


def compare_models(
	models: Dict[str, ClassifierMixin],
	x_test: pd.DataFrame,
	y_test: pd.Series,
	average: str = "weighted",
	print_results: bool = True,
) -> pd.DataFrame:
	"""Evaluate multiple models and return a sorted comparison DataFrame.

	The returned table includes accuracy, precision, recall, and F1-score.
	It is sorted by F1-score (descending) for quick interpretation.
	"""
	rows = []

	for model_name, model in models.items():
		metrics = evaluate_model(model=model, x_test=x_test, y_test=y_test, average=average)
		rows.append(
			{
				"model": model_name,
				"accuracy": metrics["accuracy"],
				"precision": metrics["precision"],
				"recall": metrics["recall"],
				"f1_score": metrics["f1_score"],
			}
		)

	results_df = pd.DataFrame(rows).sort_values(by="f1_score", ascending=False).reset_index(drop=True)

	if print_results:
		print("\n=== Model Comparison (sorted by F1-score) ===")
		print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

		for model_name, model in models.items():
			metrics = evaluate_model(model=model, x_test=x_test, y_test=y_test, average=average)
			cm = pd.DataFrame(metrics["confusion_matrix"])
			print(f"\nConfusion Matrix - {model_name}")
			print(cm.to_string(index=False, header=False))

	return results_df

