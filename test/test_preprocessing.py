from src.preprocessing import preprocessing
from src.preprocessing import extract_label
from src.preprocessing import return_features
from src.preprocessing import label_encoder
from src.preprocessing import preprocessing_pipeline
from config.config import RAVDESS
import random
import pandas as pd
import numpy as np

def test_preprocessing():

    # Arrange

    audio_files = list(RAVDESS.rglob("*.wav"))

    random_file_path = random.choice(audio_files)

    # Act

    n_mfcc = 13

    result = preprocessing(random_file_path, n_mfcc=n_mfcc)

    # Assertion

    assert isinstance(result, np.ndarray), "Output must be numpy array"

    expected_shape = (n_mfcc*3, 300)

    assert result.shape == expected_shape, f"Expected shape: {expected_shape}, got {result.shape}"

    # Value check
    assert np.isfinite(result).all(), "Feature contain NaN or Inf values"

    # Non empty check
    assert result.size > 0, "Result got empty array"

def test_extract_label():

    # Arrange 
    audio_files = list(RAVDESS.rglob("*.wav"))
    random_file_path = random.choice(audio_files)

    # Act

    result = extract_label(random_file_path)

    # assertion

    # Check str
    assert isinstance(result, str), "The label should be in string"

    # Valid emotion
    valid_emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    assert result in valid_emotions, f"label {result} is not in valid emotion"

def test_return_features():

    # Act
    result = return_features()

    # Assertion

    # Should be numpy array
    assert isinstance(result, np.ndarray), "Feature should be in numpy array"

    assert np.size(result) > 0, "Array is empty!"

    assert np.isfinite(result).all(), "Array have nan or Inf Values"

def test_label_encoder():

    # Arrange

    df = pd.DataFrame({
        "target": ["sad", "happy", "digust"]
    })

    # Act
    result = label_encoder(df)

    # Assertion

    assert isinstance(result, np.ndarray), "Encoder result is not in numpy array"

    assert len(result) == len(df), "Encoded array length must match to input length"

    assert result.dtype == np.int64, "Encoded label must be in integer"

    unique_class = df["target"].nunique()
    assert result.min() >=0 and result.max() < unique_class, "Encoded label are outside of valid range"

def test_preprocessing_pipeline():

    # Act 
    X_train, X_test, y_train, y_test = preprocessing_pipeline()

    # Assertion
    
    assert isinstance(X_train, np.ndarray), "X_train must be in numpy array"
    assert isinstance(X_test, np.ndarray), "X_test must be in numpy array"
    assert isinstance(y_train, np.ndarray), "y_train must be in numpy array"
    assert isinstance(y_test, np.ndarray), "y_test must be in numpy array"

    assert X_train.shape[0] == len(y_train), "X_train and y_train size mismatch"
    assert X_test.shape[0] == len(y_test), "X_test and y_test size mismatch"
    
    n_mfcc = 13

    assert X_train.shape[1] == n_mfcc*3, f"Expected 39 feature rows (Mfcc+delta+delta2), got {X_train.shape[1]}"

    assert X_train.shape[0] > 0, "X_train array is empty"
    assert X_test.shape[0] > 0, "X_test array is empty"

    assert len(np.unique(y_train)) > 1, "Label should contain multiple classes"
