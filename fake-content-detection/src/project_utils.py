from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tensorflow as tf


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_dataset_structure(project_root: Path) -> None:
    required_paths = [
        project_root / "data" / "raw" / "news.csv",
        project_root / "data" / "images" / "real",
        project_root / "data" / "images" / "fake",
    ]

    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing_text = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing required dataset paths:\n"
            f"{missing_text}\n"
            "Expected: data/raw/news.csv, data/images/real/, data/images/fake/"
        )


def validate_model_artifacts(model_dir: Path) -> None:
    required_files = [
        model_dir / "text_model.pkl",
        model_dir / "tfidf_vectorizer.pkl",
        model_dir / "image_model.h5",
    ]

    missing_files = [path for path in required_files if not path.exists()]
    if missing_files:
        missing_text = "\n".join(f"- {path}" for path in missing_files)
        raise FileNotFoundError(
            "Missing required model artifacts:\n"
            f"{missing_text}\n"
            "Run training scripts before inference."
        )