from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
from utils import extract_features

app = Flask(__name__)

# Load all
model = joblib.load("bird_species_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        path = "temp.wav"
        file.save(path)

        features = extract_features(path)

        if features is None:
            return jsonify({"error": "Bad audio"})

        # Check match
        if len(features) != model.n_features_in_:
            return jsonify({"error": "Feature mismatch. Retrain model."})

        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]

        best = np.argmax(probs)

        # 🔥 Decode label
        bird_name = encoder.inverse_transform([best])[0]

        return jsonify({
            "prediction": bird_name,
            "confidence": round(float(np.max(probs)*100), 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)