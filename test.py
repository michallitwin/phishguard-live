import joblib
import json
from pathlib import Path

PROFILES_DIR = Path("models/profiles")
profiles = json.loads(Path("config/feature_profiles.json").read_text())

# Sprawdzamy np. XGBoost na profilu core6
model_name = "xgboost_core6.joblib"
model = joblib.load(PROFILES_DIR / model_name)
cols = profiles["core6"]

importances = dict(zip(cols, model.feature_importances_))
sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

print(f"--- Ważność cech: {model_name} ---")
for feature, score in sorted_imp:
    print(f"{feature:<20} {score:.4f} ({score*100:.1f}%)")