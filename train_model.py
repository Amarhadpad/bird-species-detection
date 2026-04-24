# train_model.py

import os
import numpy as np
import warnings
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import joblib
from utils import extract_features

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset"

features = []
labels = []

print("Loading dataset...\n")

# ---------------- LOAD DATA ----------------
for species in os.listdir(DATASET_PATH):

    species = species.lower().strip()   # 🔥 CLEAN LABEL

    species_path = os.path.join(DATASET_PATH, species)

    if not os.path.isdir(species_path):
        continue

    print("Processing:", species)

    for file in os.listdir(species_path):

        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):

            file_path = os.path.join(species_path, file)

            feature = extract_features(file_path)

            if feature is not None:
                features.append(feature)
                labels.append(species)

# Convert
X = np.array(features)
y = np.array(labels)

print("Samples:", len(X))

# Remove rare classes
counts = Counter(y)
valid = [i for i, label in enumerate(y) if counts[label] >= 2]

X = X[valid]
y = y[valid]

# ---------------- ENCODE LABELS ----------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

# ---------------- SCALE ----------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ---------------- MODEL ----------------
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "bird_species_model.pkl")

print("✅ Training Complete")