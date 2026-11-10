import joblib
import numpy as np

from src.config import ENCODER_PATH, MODEL_CONFIG, MODEL_PATH, VECTORIZER_PATH
from src.features.extractor import extract_features

# Read from the same config as training, so serving can't drift from training.
FEATURE_COLUMNS: list[str] = MODEL_CONFIG["feature_columns"]
PHISHING_THRESHOLD: float = MODEL_CONFIG["phishing_threshold"]

def load_artifacts() -> tuple:
    """Loads the trained model, label encoder, and TF-IDF vectorizer from disk."""
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, le, vectorizer


def score_domain(domain: str, model, le, vectorizer) -> dict:
    """Extracts features from a domain and predicts its phishing probability.

    Args:
        domain: raw domain string to classify, e.g. "paypal-verify.tk".
        model: trained classifier exposing predict_proba().
        le: LabelEncoder used to map "legit"/"phishing" during training.
        vectorizer: fitted TfidfVectorizer producing character n-gram
            features; must be the same instance saved alongside the model.
    """
    features = extract_features(domain)
    structured = np.array([[features[col] for col in FEATURE_COLUMNS]])

    tfidf_vec = vectorizer.transform([domain]).toarray()
    X = np.hstack([structured, tfidf_vec])
    
    # Look up the phishing column from the encoder instead of assuming index 1.
    phishing_idx = list(le.classes_).index("phishing")
    phishing_proba = model.predict_proba(X)[0][phishing_idx]
    label = "phishing" if phishing_proba >= PHISHING_THRESHOLD else "legit"

    return {
        "domain": domain,
        "prediction": label,
        "phishing_probability": round(float(phishing_proba), 3),
    }




 