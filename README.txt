# 🐦 Bird Species Detection System

An AI-powered web application that detects bird species from audio recordings using Machine Learning and Audio Signal Processing.

---

## 🚀 Features

* 🎤 Record bird sound in real-time
* 📂 Upload audio files (`.wav`, `.mp3`, `.flac`, `.ogg`)
* 🤖 AI-based bird species prediction
* 📊 Confidence score with progress bar
* 🐦 Bird information (image + description via Wikipedia API)
* 🎧 Audio playback support
* 📈 (Optional) Spectrogram visualization

---

## 🧠 How It Works

```text
Audio Input → Preprocessing → Feature Extraction → Scaling → ML Model (XGBoost) → Prediction
```

### 🔍 Steps:

1. **Audio Input**

   * Record via microphone OR upload file

2. **Preprocessing**

   * Normalize audio
   * Remove noise (pre-emphasis)
   * Trim silence
   * Fix duration (5 seconds)

3. **Feature Extraction (Librosa)**

   * MFCC (Mel Frequency Cepstral Coefficients)
   * Chroma Features
   * Mel Spectrogram
   * Spectral Contrast
   * Tonnetz

4. **Scaling**

   * StandardScaler normalizes feature values

5. **Model Prediction**

   * XGBoost classifier predicts bird species
   * Outputs probability/confidence score

6. **Frontend Display**

   * Bird name
   * Confidence bar
   * Bird image & description

---

## 🛠️ Tech Stack

### 🔹 Backend

* Python
* Flask
* Flask-CORS

### 🔹 Machine Learning

* XGBoost (Classifier)
* Scikit-learn (Scaler, LabelEncoder)

### 🔹 Audio Processing

* Librosa
* NumPy

### 🔹 Frontend

* HTML5
* CSS3 (Glass UI)
* JavaScript (Fetch API, Audio API)

### 🔹 Visualization

* Matplotlib (optional)

---

## 📂 Project Structure

```bash
bird_species_project/
│
├── dataset/
│   ├── sparrow/
│   ├── crow/
│   └── ...
│
├── utils.py
├── train_model.py
├── app.py
├── predict_species.py
├── realtime_predict.py
│
├── templates/
│   └── index.html
│
├── requirements.txt
├── bird_species_model.pkl
├── scaler.pkl
├── label_encoder.pkl
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/bird-species-detection.git
cd bird-species-detection
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the Model

```bash
python train_model.py
```

This will generate:

* `bird_species_model.pkl`
* `scaler.pkl`
* `label_encoder.pkl`

---

### 4️⃣ Run the Application

```bash
python app.py
```

---

### 5️⃣ Open in Browser

```
http://127.0.0.1:5000
```

---

## 📊 Model Details

* Algorithm: **XGBoost Classifier**
* Input: Audio features extracted using Librosa
* Output: Bird species + confidence score
* Accuracy: ~80–90% (depends on dataset quality)

---

## 📈 Improving Accuracy

* Use clean audio recordings
* Increase dataset size (20–50 samples per bird)
* Apply data augmentation (noise, pitch shift)
* Use balanced dataset

---

## ⚠️ Known Issues

* Low confidence for noisy recordings
* Similar bird sounds may cause confusion
* Real-time audio may include background noise

---

## 🚀 Future Enhancements

* 🎯 Deep Learning (CNN on spectrograms)
* 🎤 Continuous real-time detection
* 📱 Mobile app version
* 📊 Live audio waveform & spectrogram
* 🌍 Location-based bird suggestions
* 🧠 Hybrid model (image + audio)

---

## 👨‍💻 Author

**Siddhant Shedge**
Intern Developer
NetGains Technologies Pvt. Ltd.

---

## 📜 License

This project is for educational and research purposes.

---
