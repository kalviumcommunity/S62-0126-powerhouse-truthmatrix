from pathlib import Path
import logging

import pandas as pd


# Basic logging configuration for debugging in scripts/notebooks.
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def load_data(file_path) -> pd.DataFrame:
	"""Load a CSV file and return it as a pandas DataFrame.

	Returns an empty DataFrame if the file cannot be loaded.
	"""
	path = Path(file_path)
	logger.info("Loading data from: %s", path)

	try:
		df = pd.read_csv(path)
		logger.info("Loaded %d rows and %d columns", df.shape[0], df.shape[1])
		return df
	except FileNotFoundError:
		logger.error("File not found: %s", path)
	except pd.errors.EmptyDataError:
		logger.error("CSV file is empty: %s", path)
	except Exception as exc:
		logger.exception("Failed to load CSV from %s. Error: %s", path, exc)

	return pd.DataFrame()
