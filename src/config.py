"""Central place for project paths.

Every path is built from PROJECT_ROOT (absolute), so scripts work no matter
which folder they are run from. Import from here instead of redefining paths.
"""
from pathlib import Path

# src/config.py -> parents[0] = src/, parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folders
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_DIR = MODELS_DIR / "profiles"

# Data
DATASET_PATH = DATA_DIR / "processed" / "dataset.csv"
TRANCO_CACHE_PATH = DATA_DIR / "raw" / "tranco_top1m.csv"

# Production model artifacts
MODEL_PATH = MODELS_DIR / "phishing_model.joblib"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
MANUAL_EVAL_LOG_PATH = MODELS_DIR / "test_manual_eval_log.json"

# Config files
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.json"
FEATURE_PROFILES_PATH = CONFIG_DIR / "feature_profiles.json"

# Streamlit demo models (18 variants) are not stored in git; the app downloads
# them once from this GitHub Release asset when models/profiles/ is empty.
PROFILE_MODELS_URL = (
    "https://github.com/michallitwin/phishguard-live/releases/download/profiles-v1/profile_models.zip"
)
