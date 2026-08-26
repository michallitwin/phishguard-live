from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


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


if __name__ == "__main__":
    X, y, le = load_data()
    print("Classes:", dict(zip(le.classes_, le.transform(le.classes_))))
    print("X shape:", X.shape)

    models = get_candidate_models()
    print("Models:", list(models.keys()))