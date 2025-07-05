# 📦 SMS Classifier Script
import joblib
import re

class SMSClassifier:
    def __init__(self):
        # 📁 Load model and vectorizer
        self.model = joblib.load(r'models\best_sms_spam_model.joblib')
        self.vectorizer = joblib.load(r'models\tfidf_vectorizer.joblib')

    def clean_text(self, text):
        # 🧹 Clean input text
        text = re.sub(r'[^\w\s]|\d', '', text.lower())
        return text

    def predict(self, text):
        # 🔮 Predict using the loaded model
        cleaned = self.clean_text(text)
        features = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(features)[0]
        return "Spam" if prediction == 1 else "Not Spam"

if __name__ == "__main__":
    classifier = SMSClassifier()

    while True:
        user_input = input("\n📩 Enter SMS (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        result = classifier.predict(user_input)
        print(f"🔍 Prediction: {result}")
