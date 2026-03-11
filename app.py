from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os
from utils import extract_features

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
MODEL_PATH = "bird_species_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise Exception("Model file not found. Train the model first.")

model = joblib.load(MODEL_PATH)
classes = model.classes_

# --------------------------------------------------
# Upload Folder
# --------------------------------------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# Home Page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction Route
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    try:

        # Extract features
        features = extract_features(file_path)

        if features is None:
            return jsonify({"error": "Feature extraction failed"}), 500

        features = features.reshape(1, -1)

        # Predict probabilities
        probabilities = model.predict_proba(features)[0]

        # Best prediction
        best_index = np.argmax(probabilities)
        prediction = classes[best_index]
        confidence = round(float(probabilities[best_index] * 100), 2)

        # Top 3 predictions
        top3_idx = np.argsort(probabilities)[-3:][::-1]

        top3 = []
        for i in top3_idx:
            top3.append({
                "species": classes[i],
                "confidence": round(float(probabilities[i] * 100), 2)
            })

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "top3": top3
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Delete uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)