# predict_species.py

import joblib
import numpy as np
from utils import extract_features
import os

MODEL_PATH = "bird_species_model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Model file not found. Train model first.")
    exit()

model = joblib.load(MODEL_PATH)

file_path = "test.wav"

features = extract_features(file_path)

if features is not None:
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    print("Predicted Species:", prediction[0])
else:
    print("Prediction failed.")