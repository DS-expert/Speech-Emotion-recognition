import pandas as pd
import numpy as np
import librosa
from config.config import RAVDESS, RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
logger = get_logger(__name__)

def preprocessing(file_path, n_mfcc=13, desired_length=3*16000, fixed_frame=300):

    # Load the file
    audio, sr = librosa.load(file_path, sr=16000)
    logger.info(f"Loaded {file_path} with sample rate {sr}")

    # Trim the Silence
    logger.info("Trimming silence...")
    trimmed_audio, _ = librosa.effects.trim(audio)
    logger.info(f"silence trimmed, new length: {len(trimmed_audio)} samples")

    # Normalize the audio
    logger.info("Normalizing audio...")
    normalized_audio = librosa.util.normalize(trimmed_audio)
    logger.info("Audio normalized")

    # Fix the audio length
    fixed_audio = librosa.util.fix_length(normalized_audio, size=desired_length)
    logger.info(f"Audio length fixed to {desired_length} samples")

    # Extract the Mfcc
    logger.info("Extracting MFCC features...")
    mfcc = librosa.feature.mfcc(y=fixed_audio, sr=sr, n_mfcc=n_mfcc)
    logger.info(f"Mfcc features extracted with shape: {mfcc.shape}")

    # Fix the length of mfcc
    logger.info("Fixing the mfcc length....")
    fixed_mfcc = librosa.util.fix_length(mfcc, size=fixed_frame, axis=1)
    logger.info(f"MFCC features fixed with shape: {fixed_mfcc.shape}")

    # Extract the delta and delta-delta
    logger.info(f"Extracting the delta variables...")
    delta = librosa.feature.delta(fixed_mfcc)
    delta2 = librosa.feature.delta(fixed_mfcc, order=2)
    logger.info(f"delta and delta-delta extracted with {delta.shape} and {delta2.shape}")

    # Stack the mfcc with delta and delta-delta
    features = np.vstack([fixed_mfcc, delta, delta2])

    return features

def load_audio_files(folder_path):

    logger.info(f"Loading audio files from {folder_path}...")
    audio_files = list(folder_path.rglob("*.wav"))
    logger.info(f"Audio files from {folder_path} successfully loaded")
    return audio_files

def extract_label(file_path):
    emotion_labels = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
    }

    parts = file_path.stem.split("-")
    emotion_code = parts[2]
    return emotion_labels[emotion_code]

def implement_label_extractor():

    audio_files = load_audio_files(RAVDESS)
    labels = [extract_label(file) for file in audio_files]
    return np.array(labels)

def return_features():

    audio_files = load_audio_files(RAVDESS)

    X = []

    for file in audio_files:
        feature = preprocessing(file)
        X.append(feature)
    
    return np.array(X)

def label_encoder(y):
    labelencoder = LabelEncoder()
    y_encoded = labelencoder.fit_transform(y)
    return y_encoded

def train_test_split_func(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, file_path):

    joblib.dump(X_train, file_path/"X_train.pkl")
    joblib.dump(X_test, file_path/"X_test.pkl")
    joblib.dump(y_train, file_path/"y_train.pkl")
    joblib.dump(y_test, file_path/"y_test.pkl")

def preprocessing_pipeline():
    """
    Apply the all function into one function
    """

    X = return_features()
    
    y = implement_label_extractor()

    y_encoded = label_encoder(y)

    # Split into train and test

    X_train, X_test, y_train, y_test = train_test_split_func(X, y_encoded)

    return X_train, X_test, y_train, y_test
