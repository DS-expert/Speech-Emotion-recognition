from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

LOGS_DIR = BASE_DIR / "logs"

RAW_DATA = DATA_DIR / "raw"

PROCESSED_DATA = DATA_DIR / "preprocessed"

RESULT_DIR = BASE_DIR / "results"

# RAVDESS DATA PATH

RAVDESS = RAW_DATA / "RAVDESS"

RANDOM_STATE = 42
TEST_SIZE = 0.2