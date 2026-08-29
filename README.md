![CI Pipeline](https://github.com/michallitwin/phishguard-live/actions/workflows/tests.yml/badge.svg)

# PhishGuard Live

ML system for detecting phishing domains, built from live public data
(OpenPhish, Tranco) rather than a static Kaggle dataset. Trains a real
supervised classifier instead of relying on rules or an LLM call.

## What it does

Extracts structural features from a domain name (length, brand similarity,
suspicious TLD, keyword patterns), scores it with a trained Gradient
Boosting model, and returns a phishing probability via a REST API.

## Results

- Baseline comparison: Logistic Regression, Random Forest, Gradient
  Boosting, SVM → **Gradient Boosting** selected
- Tuned via GridSearchCV (5-fold stratified CV)
- Test set: **ROC-AUC 0.92**, F1 0.73 (phishing class)
- Full EDA, PCA, and metric rationale in `notebooks/eda.ipynb`

Metrics are regenerated automatically on every training run (see
[Model monitoring](#model-monitoring)) and reflect the currently deployed
model, not a fixed snapshot.

## Known limitations

- Trained mainly on brand-impersonation phishing (dominant pattern in
  OpenPhish); doesn't reliably flag generic suspicious names unrelated
  to the 7 monitored brands.
- No WHOIS or page-content analysis — structural domain features only.
- `crt.sh` brand-monitoring module (`src/data/crtsh.py`) is implemented
  but excluded from the production pipeline due to frequent upstream
  outages.

## Run it

```bash
git clone https://github.com/michallitwin/phishguard-live.git
cd phishguard-live
docker compose up --build
```

Then open http://localhost:8000/docs

## API usage

```bash
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{"domain": "paypal-verify-login.tk"}'
```

```json
{
  "domain": "paypal-verify-login.tk",
  "prediction": "phishing",
  "phishing_probability": 0.94
}
```

Input is validated (`min_length=3` on `domain`) — malformed requests are
rejected with a `422` before reaching the model.

## Model monitoring

Each training run writes fresh evaluation metrics to `models/metrics.json`,
served live via:

GET /api/model/metrics


This makes it possible to track model performance over time (or catch
drift) without manually re-running evaluation.

## CI/CD

Every push to `main` automatically runs the full test suite via GitHub
Actions (`.github/workflows/ci.yml`) on a clean Ubuntu environment,
using the locked dependency set (`uv sync --frozen`) to catch environment
drift, not just logic errors.

## Tech stack

Python 3.13, uv, pandas, scikit-learn, FastAPI, Docker, pytest,
GitHub Actions

## Project structure

```
src/
├── data/        # OpenPhish, Tranco, crt.sh fetchers
├── features/    # feature extraction + dataset building
├── ml/          # training, tuning, evaluation, prediction
└── api/         # FastAPI app
notebooks/eda.ipynb   # EDA + PCA analysis
tests/                 # unit tests
.github/workflows/ # CI pipeline
config/features.json # brand list, suspicious TLDs, keywords
```

## Rebuilding the dataset and retraining

```bash
uv run -m src.features.build_dataset
uv run -m src.ml.train
```

## Running tests

```bash
uv run pytest -v
```
