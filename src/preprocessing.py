import pandas as pd
import numpy as np
import librosa
from config.config import RAVDESS
from src.utils.logger import get_logger

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
    features = np.vstack([mfcc, delta, delta2])

    return features