from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin


def _build_feature_row(text: str) -> pd.DataFrame:
	"""Convert raw text input into the model feature schema."""
	text = text or ""
	text_length = len(text)
	word_count = len(text.split())
	uppercase_ratio = (
		sum(1 for char in text if char.isupper()) / text_length if text_length > 0 else 0.0
	)

	return pd.DataFrame(
		[
			{
				"text": text,
				"text_length": text_length,
				"word_count": word_count,
				"uppercase_ratio": uppercase_ratio,
			}
		]
	)


def predict(text: str, model: ClassifierMixin) -> Tuple[str, float]:
	"""Predict class label and confidence score for a single input text.

	Returns:
	- prediction: predicted class label (for example, fake/real)
	- confidence: probability score in the range [0, 1]
	"""
	features = _build_feature_row(text)
	prediction = model.predict(features)[0]

	if hasattr(model, "predict_proba"):
		probabilities = model.predict_proba(features)[0]
		if hasattr(model, "classes_"):
			class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
			confidence = float(probabilities[class_to_index[prediction]])
		else:
			confidence = float(np.max(probabilities))
	elif hasattr(model, "decision_function"):
		scores = np.atleast_1d(model.decision_function(features)[0])
		if scores.ndim == 0 or len(scores) == 1:
			confidence = float(1.0 / (1.0 + np.exp(-float(scores[0]))))
		else:
			exp_scores = np.exp(scores - np.max(scores))
			softmax_scores = exp_scores / exp_scores.sum()
			confidence = float(np.max(softmax_scores))
	else:
		confidence = float("nan")

	return str(prediction), confidence

