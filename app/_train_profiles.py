"""Trains Gradient Boosting (with hyperparameter tuning) separately for each
feature profile (core6/extended9/minimal3). Saves each variant separately for
use in the Streamlit demo — does NOT touch the production model."""
import sys
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from src.ml.train import tune_model, get_candidate_models, PARAM_GRIDS, RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
PROFILES_PATH = PROJECT_ROOT / "config" / "feature_profiles.json"
OUT_DIR = PROJECT_ROOT / "models" / "profiles"

OUT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATASET_PATH)
profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))

le = LabelEncoder()
y = le.fit_transform(df["label"])
phishing_idx = list(le.classes_).index("phishing")
joblib.dump(le, OUT_DIR / "label_encoder.joblib")
domains = df["domain"].astype(str).to_numpy(dtype=object)

TARGET_MODELS = ["Gradient Boosting", "Random Forest", "XGBoost"]

results = {}

for profile_name, cols in profiles.items():
    X = df[cols]
    X_train, X_test, domains_train, domains_test, y_train, y_test = train_test_split(
    X, domains, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100)
    tfidf_train = vectorizer.fit_transform(domains_train).toarray()
    tfidf_test = vectorizer.transform(domains_test).toarray()

    X_train = np.hstack([X_train.to_numpy(), tfidf_train])
    X_test = np.hstack([X_test.to_numpy(), tfidf_test])

    joblib.dump(vectorizer, OUT_DIR / f"vectorizer_{profile_name}.joblib")

    results[profile_name] = {}

    for model_name in TARGET_MODELS:
        model_slug = model_name.lower().replace(" ", "_")

        estimator = get_candidate_models()[model_name]
        param_grid = PARAM_GRIDS[model_name]

        grid = tune_model(estimator, param_grid, X_train, y_train)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, phishing_idx]

        f1 = f1_score(y_test, y_pred, pos_label=phishing_idx)
        auc = roc_auc_score(y_test, y_proba)

        model_filename = f"{model_slug}_{profile_name}.joblib"
        joblib.dump(best_model, OUT_DIR / model_filename)

        results[profile_name][model_name] = {
            "best_params": grid.best_params_,
            "f1_phishing": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
        }
        print(f"[{model_name}] F1={f1:.3f} AUC={auc:.3f} -> Saved: {model_filename}")

(OUT_DIR / "profiles_metrics.json").write_text(json.dumps(results, indent=2))
print(f"\n 9 models saved succesfully in {OUT_DIR}/")