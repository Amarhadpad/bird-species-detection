# realtime_predict.py

import sounddevice as sd
import numpy as np
import soundfile as sf
import joblib
from utils import extract_features

model = joblib.load("bird_species_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Recording...")

sr = 22050
duration = 5

audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
sd.wait()

audio = audio.flatten()
sf.write("temp.wav", audio, sr)

features = extract_features("temp.wav")

if features is not None:
    features = scaler.transform([features])
    pred = model.predict(features)[0]
    prob = np.max(model.predict_proba(features)) * 100

    print("\nPrediction:", pred)
    print("Confidence:", round(prob, 2), "%")
else:
    print("Failed")