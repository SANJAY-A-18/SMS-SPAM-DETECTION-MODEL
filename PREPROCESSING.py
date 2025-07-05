# 📦 Data Loading & Cleaning
import pandas as pd
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# 📂 Define file paths
data_path = r'C:\Users\sanja\OneDrive\Desktop\SPAM SHEILD\SMSSpamCollection'
output_data_dir = r'C:\Users\sanja\OneDrive\Desktop\SPAM SHEILD\data'
model_dir = r'C:\Users\sanja\OneDrive\Desktop\SPAM SHEILD\models'

def load_and_clean():
    # 📄 Load dataset
    df = pd.read_csv(data_path, sep='\t', names=['label', 'text'])

    # 🔁 Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # 🧹 Clean the text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]|\d', '', text)
        return text

    df['clean_text'] = df['text'].apply(clean_text)
    return df

def extract_features(df):
    # 🔤 TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']
    return X, y, tfidf

if __name__ == "__main__":
    # 📁 Ensure output directories exist
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 🔄 Load, clean, and transform
    df = load_and_clean()
    X, y, tfidf = extract_features(df)

    # 💾 Save preprocessed data and model
    pd.to_pickle((X, y), os.path.join(output_data_dir, 'processed_data.pkl'))
    pd.to_pickle(tfidf, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
