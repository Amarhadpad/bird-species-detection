# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features

DATASET_PATH = "dataset"

# --------------------------------------------------
# Check dataset folder
# --------------------------------------------------
if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found.")
    exit()

features = []
labels = []

print("Loading dataset...")

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
for species in os.listdir(DATASET_PATH):

    species_path = os.path.join(DATASET_PATH, species)

    if os.path.isdir(species_path):

        for file in os.listdir(species_path):

            if file.endswith(".wav"):

                file_path = os.path.join(species_path, file)

                feature = extract_features(file_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(species)

# --------------------------------------------------
# Check if data exists
# --------------------------------------------------
if len(features) == 0:
    print("No valid audio files found.")
    exit()

X = np.array(features)
y = np.array(labels)

print("Total samples loaded:", len(X))
print("Unique species:", len(set(y)))

# --------------------------------------------------
# Validate minimum dataset requirements
# --------------------------------------------------
if len(set(y)) < 2:
    print("\nERROR: You need at least 2 bird species to train a classifier.")
    exit()

if len(X) < 5:
    print("\nWARNING: Very small dataset. Model accuracy will not be reliable.")

# --------------------------------------------------
# Train/Test Split (Safe Version)
# --------------------------------------------------
if len(X) < 2:
    print("\nNot enough samples to split. Training on all data.")
    X_train, y_train = X, y
    X_test, y_test = X, y
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# --------------------------------------------------
# Train Model
# --------------------------------------------------
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
if len(X) >= 2:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
else:
    print("\nSkipping evaluation (not enough data).")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
joblib.dump(model, "bird_species_model.pkl")

print("\nModel saved as bird_species_model.pkl")
print("Training completed successfully.")