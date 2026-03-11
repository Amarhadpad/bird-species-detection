# utils.py

import librosa
import numpy as np
import os

N_MFCC = 13

def extract_features(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        audio, sr = librosa.load(file_path, sr=None)

        # Normalize audio
        audio = audio / np.max(np.abs(audio))

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        return np.mean(mfcc, axis=1)

    except Exception as e:
        print("Error processing:", file_path)
        print(e)
        return None