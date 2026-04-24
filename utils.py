# utils.py

import librosa
import numpy as np
import os

def extract_features(file_path):

    if not os.path.exists(file_path):
        return None

    try:
        y, sr = librosa.load(file_path, sr=22050, duration=5)

        if len(y) == 0:
            return None

        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        # Noise reduction
        y = librosa.effects.preemphasis(y)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        if len(y) < 2048:
            return None

        # Features
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        tonnetz = np.mean(
            librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T,
            axis=0
        )

        features = np.hstack([mfcc, chroma, mel, contrast, tonnetz])

        if np.isnan(features).any():
            return None

        return features

    except:
        return None