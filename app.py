from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os
from utils import extract_features

app = Flask(__name__)

# ==============================
# LOAD MODEL (SAFE WAY)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    print("🔄 Loading model files...")

    model = joblib.load(os.path.join(BASE_DIR, "bird_species_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

    print("✅ Model loaded successfully")

except Exception as e:
    print("❌ MODEL LOAD ERROR:", e)
    model = None


# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded properly"})

        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No file uploaded"})

        # Save temp file safely
        temp_path = os.path.join(BASE_DIR, "temp.wav")
        file.save(temp_path)

        # Extract features
        features = extract_features(temp_path)

        if features is None:
            return jsonify({"error": "Invalid audio file"})

        # Ensure feature size matches
        if len(features) != model.n_features_in_:
            return jsonify({"error": "Feature mismatch. Retrain model."})

        # Scale features
        features = scaler.transform([features])

        # Predict probabilities
        probs = model.predict_proba(features)[0]
        best_index = np.argmax(probs)

        # Decode label
        bird_name = encoder.inverse_transform([best_index])[0]

        return jsonify({
            "prediction": bird_name,
            "confidence": round(float(np.max(probs) * 100), 2)
        })

    except Exception as e:
        print("❌ PREDICTION ERROR:", str(e))
        return jsonify({"error": str(e)})


# ==============================
# RUN SERVER (Render Compatible)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)