# utils.py

import librosa
import numpy as np
import os

def extract_features(file_path):

    # --------------------------------------
    # Check file exists
    # --------------------------------------
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return None

    try:
        # --------------------------------------
        # Load audio
        # --------------------------------------
        audio, sr = librosa.load(file_path, sr=22050)

        # Skip empty audio
        if len(audio) == 0:
            print("Empty audio:", file_path)
            return None

        # --------------------------------------
        # Normalize safely
        # --------------------------------------
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # --------------------------------------
        # Remove silence
        # --------------------------------------
        audio, _ = librosa.effects.trim(audio)

        if len(audio) < 2048:
            print("Audio too short:", file_path)
            return None

        # --------------------------------------
        # MFCC Features
        # --------------------------------------
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)

        # --------------------------------------
        # Spectral Centroid
        # --------------------------------------
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=audio, sr=sr)
        )

        # --------------------------------------
        # Spectral Bandwidth
        # --------------------------------------
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        )

        # --------------------------------------
        # Zero Crossing Rate
        # --------------------------------------
        zcr = np.mean(
            librosa.feature.zero_crossing_rate(audio)
        )

        # --------------------------------------
        # Combine Features
        # --------------------------------------
        features = np.hstack([
            mfcc_mean,
            spectral_centroid,
            spectral_bandwidth,
            zcr
        ])

        # Check NaN values
        if np.isnan(features).any():
            print("NaN detected:", file_path)
            return None

        return features

    except Exception as e:
        print("Error processing:", file_path)
        print("Reason:", e)
        return None