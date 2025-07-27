# 📱 SMS Spam Detection System (TF-IDF + LightGBM)

A lightweight and efficient machine learning project to classify **SMS messages as SPAM or HAM** using traditional NLP techniques and a tuned LightGBM classifier. Designed for performance and clarity, this pipeline is ideal for beginners and intermediate ML practitioners.

---

## 🚀 Highlights

- 📊 Uses **TF-IDF** vectorization to convert SMS text into numeric features
- 🌳 Trains a **LightGBM model** using GridSearchCV for best performance
- 📈 Evaluates model using metrics like accuracy, precision, recall, F1, and ROC AUC
- 💾 Saves preprocessed vectors and model to disk
- 🧹 Clean and modular structure (Preprocessing → Training → Prediction-ready)

---

## 🧱 Project Structure

```bash
├── PREPROCESSING.py        # Loads, cleans, and vectorizes SMS messages using TF-IDF
├── TRAINING.py             # Trains the best model using GridSearch on LightGBM
├── PREDICTION.py          # Loads trained model and classifies input sms messages
├── models/                   # Stores the trained model and vectorizer
├── data/                     # Stores processed feature data
