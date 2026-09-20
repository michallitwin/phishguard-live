from pathlib import Path

import joblib
import numpy as np

from src.features.extractor import extract_features

MODEL_PATH = Path("models/phishing_model.joblib")
ENCODER_PATH = Path("models/label_encoder.joblib")

FEATURE_COLUMNS = [
    "length",
    "digits",
    "hyphens",
    "brand_sim",
    "suspicious_tld",
    "keywords",
]

PHISHING_THRESHOLD = 0.25

def load_artifacts() -> tuple:
    """
    Loads the trained model and label encoder from disk.
    """
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le


def score_domain(domain: str, model, le) -> dict:
    """
    Extracts features from a domain and predicts its phishing probability.
    """
    features = extract_features(domain)
    X = np.array([[features[col] for col in FEATURE_COLUMNS]])
    probabilities = model.predict_proba(X)

    phishing_proba = probabilities[0][1]
    label = "phishing" if phishing_proba >= PHISHING_THRESHOLD else "legit"

    return {
        "domain": domain,
        "prediction": label,
        "phishing_probability": round(float(phishing_proba), 3),
    }




 