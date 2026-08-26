from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import f1_score



DATASET_PATH = Path("data/processed/dataset.csv")
MODEL_PATH = Path("models/phishing_model.joblib")
SCALER_PATH = Path("models/scaler.joblib")

FEATURE_COLUMNS = ["length", "digits", "hyphens", "brand_sim", "suspicious_tld", "keywords"]
RANDOM_STATE = 42


def load_data() -> tuple:
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLUMNS].values
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    return X, y, le


def get_candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=RANDOM_STATE),
    }


def evaluate_baseline(X: np.ndarray, y: np.ndarray) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = get_candidate_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = f1_score(y_test, y_pred)
        results[name] = score
        print(f"{name} F1(phishing) = {score:.3f}")

    return results



if __name__ == "__main__":
    X, y, le = load_data()
    print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("X shape:", X.shape)

    baseline_results = evaluate_baseline(X, y)