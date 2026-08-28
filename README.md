# PhishGuard Live

ML system for detecting phishing domains, built from live public data (OpenPhish, Tranco) rather than a static Kaggle dataset. Trains a real supervised classifier instead of relying on rules or an LLM call.

## What it does

Extracts structural features from a domain name (length, brand similarity, suspicious TLD, keyword patterns), scores it with a trained Gradient Boosting model, and returns a phishing probability via a REST API.

## Results

- Baseline comparison: Logistic Regression, Random Forest, Gradient Boosting, SVM → Gradient Boosting selected
- Tuned via GridSearchCV (5-fold stratified CV)
- Test set: ROC-AUC 0.92, F1 0.73 (phishing class)
- Full EDA, PCA, and metric rationale in `notebooks/eda.ipynb`

Metrics are regenerated automatically on every training run (see below) and reflect the currently deployed model, not a fixed snapshot.

## Known limitations

- Trained mainly on brand-impersonation phishing (dominant pattern in OpenPhish); doesn't reliably flag generic suspicious names unrelated to the 7 monitored brands.
- No WHOIS or page-content analysis — structural domain features only.
- `crt.sh` brand-monitoring module (`src/data/crtsh.py`) is implemented but excluded from the production pipeline due to frequent upstream outages.

## Run it

```bash
git clone https://github.com/michallitwin/phishguard-live.git
cd phishguard-live
docker compose up --build
```

Then open http://localhost:8000/docs

## Model monitoring

Each training run writes fresh evaluation metrics to `models/metrics.json`, served live via:

```bash
GET /api/model/metrics
```

This makes it possible to track model performance over time (or catch drift) without manually re-running evaluation.

## Tech stack

Python 3.13, uv, pandas, scikit-learn, FastAPI, Docker, pytest

## Project structure

```
src/
├── data/        # OpenPhish, Tranco, crt.sh fetchers
├── features/    # feature extraction + dataset building
├── ml/          # training, tuning, evaluation, prediction
└── api/         # FastAPI app
notebooks/eda.ipynb   # EDA + PCA analysis
tests/                 # unit tests
```