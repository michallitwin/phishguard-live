"""Trains Gradient Boosting (with hyperparameter tuning) separately for each
feature profile (core6/extended9/minimal3). Saves each variant separately for
use in the Streamlit demo — does NOT touch the production model."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from src.ml.train import tune_model, PARAM_GRIDS, RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
PROFILES_PATH = PROJECT_ROOT / "config" / "feature_profiles.json"
OUT_DIR = PROJECT_ROOT / "models" / "profiles"

OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATASET_PATH)
profiles = json.loads(PROFILES_PATH.read_text())

le = LabelEncoder()
y = le.fit_transform(df["label"])
phishing_idx = list(le.classes_).index("phishing")

results = {}
for profile_name, cols in profiles.items():
    X = df[cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Same tuning approach as the production model, for a fair comparison
    grid = tune_model(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        PARAM_GRIDS["Gradient Boosting"],
        X_train,
        y_train,
    )
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, phishing_idx]
    f1 = f1_score(y_test, y_pred, pos_label=phishing_idx)
    auc = roc_auc_score(y_test, y_proba)

    joblib.dump(best_model, OUT_DIR / f"{profile_name}.joblib")
    results[profile_name] = {
        "features": cols,
        "best_params": grid.best_params_,
        "f1_phishing": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
    }
    print(f"{profile_name}: F1={f1:.3f}  AUC={auc:.3f}  params={grid.best_params_}")

joblib.dump(le, OUT_DIR / "label_encoder.joblib")
(OUT_DIR / "profiles_metrics.json").write_text(json.dumps(results, indent=2))
print(f"\nSaved 3 tuned profile variants to {OUT_DIR}/")