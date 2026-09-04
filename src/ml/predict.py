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


def load_artifacts() -> tuple:
    """
    Loads the trained model and label encoder from disk.
    
    Returns:
        tuple: The loaded model and label encoder.
    """
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le


def score_domain(domain: str, model, le) -> dict:
    """
    Extracts features from a domain and predicts its phishing probability.
    
    Args:
        domain (str): The domain name to evaluate.
        model: The trained machine learning model.
        le: The trained label encoder.
        
    Returns:
        dict: A dictionary containing the domain, predicted label, and phishing probability.
    """
    features = extract_features(domain)
    X = np.array([[features[col] for col in FEATURE_COLUMNS]])
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)

    label = le.inverse_transform(prediction)[0]
    phishing_proba = probabilities[0][1]

    return {
        "domain": domain,
        "prediction": label,
        "phishing_probability": round(float(phishing_proba), 3),
    }




