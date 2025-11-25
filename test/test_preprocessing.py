from src.preprocessing import preprocessing
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
