"""Simple web interface for checking domains for phishing indicators.
Lets the user choose which feature profile the model was trained on."""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import streamlit as st
from src.features.extractor import DomainFeatureExtractor
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILES_DIR = PROJECT_ROOT / "models" / "profiles"
PROFILES_CONFIG = PROJECT_ROOT / "config" / "feature_profiles.json"

st.set_page_config(page_title="PhishGuard Live", page_icon="🛡️")
st.title("🛡️ PhishGuard Live")
st.caption("Phishing domain detection based on domain structure")


@st.cache_resource
def load_profile_data():
    profiles = json.loads(PROFILES_CONFIG.read_text())
    le = joblib.load(PROFILES_DIR / "label_encoder.joblib")
    models = {name: joblib.load(PROFILES_DIR / f"{name}.joblib") for name in profiles}
    return profiles, models, le


profiles, models, le = load_profile_data()
extractor = DomainFeatureExtractor()

profile_name = st.selectbox("Feature profile", list(profiles.keys()))
domain = st.text_input("Enter a domain to check", placeholder="e.g. paypal-verify-login.tk")

if st.button("Check", type="primary") and domain:
    feature_cols = profiles[profile_name]
    all_features = extractor.extract(domain)
    X = np.array([[all_features[col] for col in feature_cols]])

    model = models[profile_name]
    prediction = model.predict(X)
    proba = model.predict_proba(X)[0][1]
    label = le.inverse_transform(prediction)[0]

    if label == "phishing":
        st.error(f"⚠️ PHISHING — probability: {proba:.1%}")
    else:
        st.success(f"✅ LEGIT — phishing probability: {proba:.1%}")

    st.progress(float(proba))
    st.caption(f"Features used ({len(feature_cols)}): {', '.join(feature_cols)}")

st.divider()
st.caption("Model: Gradient Boosting (tuned per feature profile) · Data: OpenPhish + Tranco Top 1M")