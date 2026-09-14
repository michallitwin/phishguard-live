"""Web interface for checking domains for phishing indicators.
Dynamically loads models from models/profiles/ and feature profiles from config/.
"""
import sys
import json
from pathlib import Path
import numpy as np
import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "models" / "profiles"
CONFIG_DIR = PROJECT_ROOT / "config"
PROFILES_CONFIG = CONFIG_DIR / "feature_profiles.json"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.extractor import DomainFeatureExtractor

st.set_page_config(page_title="PhishGuard Live", page_icon="🛡️")
st.title("🛡️ PhishGuard Live")
st.caption("Phishing domain detection based on domain structure")


MODEL_OPTIONS = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}


@st.cache_resource
def load_base_assets():
    profiles = json.loads(PROFILES_CONFIG.read_text(encoding="utf-8"))
    le = joblib.load(PROFILES_DIR / "label_encoder.joblib")
    return profiles, le


@st.cache_resource
def load_model(model_slug: str, profile_name: str):
    model_path = PROFILES_DIR / f"{model_slug}_{profile_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model: {model_path.name}")
    return joblib.load(model_path)


profiles, le = load_base_assets()
extractor = DomainFeatureExtractor()
phishing_idx = list(le.classes_).index("phishing")

col1, col2 = st.columns(2)
with col1:
    selected_model_label = st.selectbox("Pick Model", list(MODEL_OPTIONS.keys()))
with col2:
    selected_profile = st.selectbox("Feature Profile", list(profiles.keys()))

model_slug = MODEL_OPTIONS[selected_model_label]
model = load_model(model_slug, selected_profile)
feature_cols = profiles[selected_profile]

domain = st.text_input("Enter a domain to check", placeholder="e.g. paypal-verify-login.tk")

if st.button("Check", type="primary") and domain:
    if not extractor.is_valid_domain(domain):
        st.warning("⚠️ Please enter a valid domain (e.g. example.com)")
    else:
        all_features = extractor.extract(domain)
        X = np.array([[all_features[col] for col in feature_cols]])

        prediction = model.predict(X)
        proba = model.predict_proba(X)[0][phishing_idx]
        label = le.inverse_transform(prediction)[0]

        if label == "phishing":
            st.error(f"⚠️ PHISHING — probability: {proba:.1%}")
        else:
            st.success(f"✅ LEGIT — phishing probability: {proba:.1%}")

        st.progress(float(proba))
        st.caption(
            f"Engine: **{selected_model_label}** | Profile: **{selected_profile}** ({len(feature_cols)} cech) | "
            f"Features: `{', '.join(feature_cols)}`"
        )

st.divider()
st.caption(f"Active Model: {selected_model_label} · Profile: {selected_profile} · Data: OpenPhish + Tranco Top 1M")