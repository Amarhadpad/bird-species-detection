# predict_species.py

import joblib
import numpy as np
import os
import sys
from utils import extract_features

MODEL_PATH = "bird_species_model.pkl"

# -----------------------------
# Check model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found. Train the model first.")
    exit()

model = joblib.load(MODEL_PATH)
classes = model.classes_

# -----------------------------
# Get audio file
# -----------------------------
if len(sys.argv) < 2:
    print("Usage: python predict_species.py <audio_file>")
    exit()

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("❌ Audio file not found:", file_path)
    exit()

# -----------------------------
# Extract features
# -----------------------------
features = extract_features(file_path)

if features is None:
    print("❌ Feature extraction failed.")
    exit()

features = features.reshape(1, -1)

# -----------------------------
# Predict
# -----------------------------
probabilities = model.predict_proba(features)[0]

top3 = np.argsort(probabilities)[-3:][::-1]

print("\n🐦 Top Bird Predictions:\n")

for i in top3:
    print(
        f"{classes[i]}  →  {round(probabilities[i]*100,2)}%"
    )