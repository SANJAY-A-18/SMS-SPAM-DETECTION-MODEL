# 📦 best_model_train.py
import os
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# 📂 Define paths
data_dir = r'C:\Users\sanja\OneDrive\Desktop\SPAM SHEILD\data'
model_dir = r'C:\Users\sanja\OneDrive\Desktop\SPAM SHEILD\models'

def load_processed_data():
    # 📄 Load processed data
    X, y = pd.read_pickle(os.path.join(data_dir, 'processed_data.pkl'))
    return X, y

def train_best_model():
    # 🔄 Load data
    X, y = load_processed_data()

    # 🔀 Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 🌳 Define model
    model = LGBMClassifier(objective='binary', is_unbalance=True)

    # 🔧 Define hyperparameter grid
    param_grid = {
        'num_leaves': [31, 50],
        'learning_rate': [0.1, 0.05],
        'n_estimators': [100, 200],
        'max_depth': [5, 10, -1]
    }

    # 🔍 Grid Search with Cross-Validation
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # 🏋️ Train model
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 📈 Predict and evaluate
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    print("\n📊 Evaluation Results (Best Model):")
    print(classification_report(y_test, y_pred))
    print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"\n✅ Best Params: {grid.best_params_}")

    # 💾 Save the best model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'best_sms_spam_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"✅ Best model saved to: {model_path}")

if __name__ == "__main__":
    train_best_model()
