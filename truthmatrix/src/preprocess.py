from __future__ import annotations

from typing import Iterable

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean raw dataframe for downstream feature engineering.

	Steps:
	- Standardize column names
	- Drop duplicate rows
	- Handle missing values in required and optional columns
	"""
	cleaned = df.copy()

	# Standardize column names: lowercase, strip spaces, replace spaces with underscores.
	cleaned.columns = [str(col).strip().lower().replace(" ", "_") for col in cleaned.columns]

	cleaned = cleaned.drop_duplicates()

	required_cols = [col for col in ["text", "label"] if col in cleaned.columns]
	if required_cols:
		cleaned = cleaned.dropna(subset=required_cols)

	for col in cleaned.columns:
		if cleaned[col].dtype == "object":
			cleaned[col] = cleaned[col].fillna("unknown")
		else:
			cleaned[col] = cleaned[col].fillna(0)

	return cleaned


def add_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Create text-derived features used for modeling.

	Features:
	- text_length
	- word_count
	- uppercase_ratio
	"""
	featured = df.copy()

	if "text" not in featured.columns:
		raise ValueError("Expected 'text' column to create text features.")

	featured["text"] = featured["text"].astype(str)
	featured["text_length"] = featured["text"].str.len()
	featured["word_count"] = featured["text"].str.split().str.len()
	featured["uppercase_ratio"] = featured["text"].apply(
		lambda x: (sum(1 for ch in x if ch.isupper()) / len(x)) if len(x) > 0 else 0.0
	)

	return featured


def prepare_features(
	df: pd.DataFrame,
	feature_cols: Iterable[str] = ("text", "text_length", "word_count", "uppercase_ratio"),
	target_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series]:
	"""Select relevant features and split into X and y."""
	prepared = df.copy()
	feature_cols = list(feature_cols)

	missing_features = [col for col in feature_cols if col not in prepared.columns]
	if missing_features:
		raise ValueError(
			f"Missing required feature columns: {missing_features}. "
			"Run add_features(df) before prepare_features(df)."
		)

	if target_col not in prepared.columns:
		raise ValueError(f"Missing target column: {target_col}")

	x = prepared[feature_cols]
	y = prepared[target_col]
	return x, y

