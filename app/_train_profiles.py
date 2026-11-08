"""Trains Gradient Boosting, Random Forest and XGBoost (with hyperparameter
tuning) separately for each feature profile (core6/extended9/minimal3), with
and without TF-IDF. Saves each variant to models/profiles/ for use in the
Streamlit demo — does NOT touch the production model."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_PATH, FEATURE_PROFILES_PATH, PROFILES_DIR
from src.ml.train import PARAM_GRIDS, RANDOM_STATE, get_candidate_models, tune_model

TARGET_MODELS = ["Gradient Boosting", "Random Forest", "XGBoost"]
TEST_SIZE = 0.2


def load_data() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Loads the processed dataset and the feature profiles config."""
    df = pd.read_csv(DATASET_PATH)
    profiles = json.loads(FEATURE_PROFILES_PATH.read_text(encoding="utf-8"))
    return df, profiles


def encode_labels(df: pd.DataFrame) -> tuple[np.ndarray, LabelEncoder, int]:
    """Encodes 'legit'/'phishing' labels and returns the phishing class index."""
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    phishing_idx = list(le.classes_).index("phishing")
    return y, le, phishing_idx


def build_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    domains_train: np.ndarray,
    domains_test: np.ndarray,
    use_tfidf: bool,
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer | None]:
    """Builds final feature matrices, optionally adding char n-gram TF-IDF.

    The vectorizer is fitted on training domains only (fit_transform) and
    applied to test domains with transform(), to avoid data leakage.
    """
    if not use_tfidf:
        return X_train.to_numpy(), X_test.to_numpy(), None

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=100)
    tfidf_train = vectorizer.fit_transform(domains_train).toarray()
    tfidf_test = vectorizer.transform(domains_test).toarray()
    X_train_final = np.hstack([X_train.to_numpy(), tfidf_train])
    X_test_final = np.hstack([X_test.to_numpy(), tfidf_test])
    return X_train_final, X_test_final, vectorizer


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, phishing_idx: int) -> dict[str, float]:
    """Computes F1 (phishing class) and ROC-AUC on the test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, phishing_idx]
    return {
        "f1_phishing": round(float(f1_score(y_test, y_pred, pos_label=phishing_idx)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def train_and_save(model_name: str, X_train: np.ndarray, y_train: np.ndarray, filename: str):
    """Tunes one model with grid search, saves the best estimator, returns the grid."""
    estimator = get_candidate_models()[model_name]
    grid = tune_model(estimator, PARAM_GRIDS[model_name], X_train, y_train)
    joblib.dump(grid.best_estimator_, PROFILES_DIR / filename)
    return grid


def main() -> None:
    """Trains all model x profile x TF-IDF variants and saves their metrics."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    df, profiles = load_data()
    y, le, phishing_idx = encode_labels(df)
    joblib.dump(le, PROFILES_DIR / "label_encoder.joblib")
    domains = df["domain"].astype(str).to_numpy(dtype=object)

    results: dict = {}
    for profile_name, cols in profiles.items():
        # One split for features, domains and labels -> rows stay aligned.
        X_train, X_test, domains_train, domains_test, y_train, y_test = train_test_split(
            df[cols], domains, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        results[profile_name] = {}

        for use_tfidf in [True, False]:
            tfidf_tag = "tfidf" if use_tfidf else "notfidf"
            X_train_final, X_test_final, vectorizer = build_features(
                X_train, X_test, domains_train, domains_test, use_tfidf
            )
            if vectorizer is not None:
                joblib.dump(vectorizer, PROFILES_DIR / f"vectorizer_{profile_name}.joblib")

            results[profile_name][tfidf_tag] = {}
            for model_name in TARGET_MODELS:
                model_slug = model_name.lower().replace(" ", "_")
                filename = f"{model_slug}_{profile_name}_{tfidf_tag}.joblib"

                grid = train_and_save(model_name, X_train_final, y_train, filename)
                metrics = evaluate_model(grid.best_estimator_, X_test_final, y_test, phishing_idx)

                results[profile_name][tfidf_tag][model_name] = {"best_params": grid.best_params_, **metrics}
                print(
                    f"[{model_name}][{tfidf_tag}] F1={metrics['f1_phishing']:.3f} "
                    f"AUC={metrics['roc_auc']:.3f} -> Saved: {filename}"
                )

    (PROFILES_DIR / "profiles_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll models saved successfully in {PROFILES_DIR}/")


if __name__ == "__main__":
    main()
