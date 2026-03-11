# train_model.py

import os
import numpy as np
import warnings
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from utils import extract_features

# ----------------------------------------
# Ignore warnings
# ----------------------------------------
warnings.filterwarnings("ignore")

DATASET_PATH = "dataset"

# ----------------------------------------
# Check dataset folder
# ----------------------------------------
if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found.")
    exit()

features = []
labels = []

print("Loading dataset...\n")

# ----------------------------------------
# Load dataset
# ----------------------------------------
for species in os.listdir(DATASET_PATH):

    species_path = os.path.join(DATASET_PATH, species)

    if not os.path.isdir(species_path):
        continue

    print("Processing species:", species)

    for file in os.listdir(species_path):

        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):

            file_path = os.path.join(species_path, file)

            try:
                feature = extract_features(file_path)

                if feature is None:
                    continue

                if np.isnan(feature).any():
                    continue

                features.append(feature)
                labels.append(species)

            except Exception:
                print("Skipped corrupted file:", file_path)

# ----------------------------------------
# Convert to numpy
# ----------------------------------------
X = np.array(features)
y = np.array(labels)

print("\n----------------------------------")
print("Total samples loaded:", len(X))
print("Unique species:", len(set(y)))
print("----------------------------------")

# ----------------------------------------
# Remove rare classes (<2 samples)
# ----------------------------------------
class_counts = Counter(y)

valid_indices = [i for i, label in enumerate(y) if class_counts[label] >= 2]

X = X[valid_indices]
y = y[valid_indices]

print("\nAfter removing rare species:")
print("Total samples:", len(X))
print("Unique species:", len(set(y)))

# ----------------------------------------
# Validate dataset
# ----------------------------------------
if len(X) == 0:
    print("ERROR: No valid audio files found.")
    exit()

if len(set(y)) < 2:
    print("ERROR: Need at least 2 bird species.")
    exit()

# ----------------------------------------
# Train/Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# Train Model
# ----------------------------------------
print("\nTraining RandomForest model...\n")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------------------
# Evaluate Model
# ----------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------------------
# Save Model
# ----------------------------------------
joblib.dump(model, "bird_species_model.pkl")

print("\nModel saved as: bird_species_model.pkl")
print("\nTraining completed successfully.")