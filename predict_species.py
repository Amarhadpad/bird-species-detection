# predict_species.py

import joblib
import numpy as np
import sys
import os
from utils import extract_features

model = joblib.load("bird_species_model.pkl")
scaler = joblib.load("scaler.pkl")

if len(sys.argv) < 2:
    print("Usage: python predict_species.py <audio>")
    exit()

file_path = sys.argv[1]

features = extract_features(file_path)

if features is None:
    print("Feature error")
    exit()

features = scaler.transform([features])

probs = model.predict_proba(features)[0]
classes = model.classes_

top3 = np.argsort(probs)[-3:][::-1]

print("\nTop Predictions:\n")

for i in top3:
    print(classes[i], "→", round(probs[i]*100, 2), "%")