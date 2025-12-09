from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

LOGS_DIR = BASE_DIR / "logs"

RAW_DATA = DATA_DIR / "raw"

PROCESSED_DATA = DATA_DIR / "preprocessed"

RESULT_DIR = BASE_DIR / "result"

# RAVDESS DATA PATH

RAVDESS = RAW_DATA / "RAVDESS"

X_TRAIN = PROCESSED_DATA / "X_train.pkl"
X_TEST = PROCESSED_DATA / "X_test.pkl"
Y_TRAIN = PROCESSED_DATA / "y_train.pkl"
Y_TEST = PROCESSED_DATA / "y_test.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2