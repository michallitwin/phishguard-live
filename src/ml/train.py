import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.joblib"
ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"  
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"

with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
    _config = json.load(f)

RANDOM_STATE = _config["random_state"]
FEATURE_COLUMNS = _config["feature_columns"]
TFIDF_MAX_FEATURES = _config["tfidf"]["max_features"]
TFIDF_NGRAM_RANGE = tuple(_config["tfidf"]["ngram_range"])


PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_features": [0.5, 0.7, None],
        "subsample": [0.7, 0.85, 1.0],
    },
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2", None],
    },
    "Logistic Regression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "solver": ["lbfgs", "liblinear"],
    },
    "SVM": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
    },
    "XGBoost": {
        "n_estimators": [100,200],
        "max_depth": [3,5,7],
        "learning_rate": [0.05, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 1.0],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "Decision Tree": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
    },

}


def load_data(
    dataset_path: Path = DATASET_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Loads dataset, extracts feature matrix, domains, and encodes target labels."""
    df = pd.read_csv(dataset_path)
    X = df[feature_columns].values
    domains = df["domain"].astype(str).to_numpy(dtype=object)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    return X, domains, y, le


def get_candidate_models() -> dict[str, BaseEstimator]:
    """Model factory returning fresh instances of candidate estimators."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", 
            max_iter=1000, 
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", 
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            class_weight="balanced", 
            probability=True, 
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            eval_metric ="logloss", 
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", 
            random_state=RANDOM_STATE
        ),
    }


def evaluate_baselines(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: dict[str, BaseEstimator] | None = None,
) -> dict[str, float]:
    """Evaluates candidate models on scaled training data using F1 score."""
    if models is None:
        models = get_candidate_models()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = float(f1_score(y_test, y_pred))
        results[name] = score
        print(f"[{name}] F1(phishing) = {score:.4f}")

    return results


def tune_model(
    estimator: BaseEstimator,
    param_grid: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: int = 5,
) -> GridSearchCV:
    """Performs cross-validated hyperparameter tuning for any estimator."""
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV score (F1): {grid.best_score_:.4f}")

    return grid

def final_evaluation(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Evaluates final model on held-out test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Final Evaluation (Test Set) ---")
    print(
        classification_report(y_test, y_pred, target_names=["legit", "phishing"])
    )
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return y_pred, y_proba

def save_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_path: Path = METRICS_PATH,
) -> None:
    """Calculates and persists model evaluation metrics to disk as JSON."""
    metrics = {
        "model_name": model_name,
        "f1_phishing": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved successfully to: {output_path}")


def save_artifacts(model, le, model_path=MODEL_PATH, encoder_path = ENCODER_PATH) -> None:
    """Saves model and label encoder artifacts to disk."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"\nArtifacts saved successfully to: {model_path.parent}/")


def run_pipeline(
        dataset_path: Path = DATASET_PATH,
        feature_columns: list[str] = FEATURE_COLUMNS,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = ENCODER_PATH,
        metrics_path: Path = METRICS_PATH,
        vectorizer_path: Path = VECTORIZER_PATH,
        use_tfidf: bool = True,
    ) -> None:
    """Orchestrates data loading, benchmarking, tuning, and evaluation."""
    X, domains, y, le = load_data(dataset_path, feature_columns)
    print(f"Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Dataset shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    domains_train, domains_test = train_test_split(
        domains, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    if use_tfidf:
        print("\n--- Fitting TF-IDF on character n-grams (train domains only) ---")
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=TFIDF_NGRAM_RANGE,
            max_features=TFIDF_MAX_FEATURES,
        )
        tfidf_train = vectorizer.fit_transform(domains_train).toarray()
        tfidf_test = vectorizer.transform(domains_test).toarray()

        X_train = np.hstack([X_train, tfidf_train])
        X_test = np.hstack([X_test, tfidf_test])
        print(f"Combined feature shape: train={X_train.shape}, test={X_test.shape}")

        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, vectorizer_path)
    else:
        print("Skipping TF-IDF")

    print("\n--- Benchmarking Baseline Models ---")
    baseline_results = evaluate_baselines(X_train, X_test, y_train, y_test)

    best_model_name = max(baseline_results, key=baseline_results.get)
    print(f"\nSelected best baseline model: {best_model_name}")

    print(f"\n--- Hyperparameter Tuning: {best_model_name} ---")
    candidate_models = get_candidate_models()
    best_estimator = candidate_models[best_model_name]
    param_grid = PARAM_GRIDS.get(best_model_name, {})

    grid = tune_model(best_estimator, param_grid, X_train, y_train)
    y_pred, y_proba = final_evaluation(grid.best_estimator_, X_test, y_test)
    save_artifacts(grid.best_estimator_, le, model_path, encoder_path)
    save_metrics(y_test, y_pred, y_proba, best_model_name, metrics_path)