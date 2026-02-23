from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
import os
from utils import extract_features

app = Flask(__name__)
CORS(app)

MODEL_PATH = "bird_species_model.pkl"
model = joblib.load(MODEL_PATH)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ✅ This renders your UI
@app.route("/")
def home():
    return render_template("index.html")


# ✅ This handles prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    features = extract_features(file_path)

    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = float(np.max(model.predict_proba(features)) * 100)

    return jsonify({
        "prediction": prediction,
        "confidence": round(probability, 2)
    })


if __name__ == "__main__":
    app.run(debug=True)
    if __name__ == "__main__":
    app.run(debug=True)