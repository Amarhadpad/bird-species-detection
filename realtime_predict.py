# realtime_predict.py

import sounddevice as sd
import numpy as np
import soundfile as sf
import joblib
import os
from utils import extract_features

SAMPLE_RATE = 44100
DURATION = 5
MODEL_PATH = "bird_species_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model not found. Train first.")
    exit()

model = joblib.load(MODEL_PATH)

print("Recording for 5 seconds...")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1
)

sd.wait()

audio = recording.flatten()

sf.write("realtime.wav", audio, SAMPLE_RATE)

features = extract_features("realtime.wav")

if features is not None:
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    print("Predicted Species:", prediction[0])
else:
    print("Prediction failed.")