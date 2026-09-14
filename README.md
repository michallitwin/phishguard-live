![CI Pipeline](https://github.com/michallitwin/phishguard-live/actions/workflows/tests.yml/badge.svg)

# PhishGuard Live

ML system for detecting phishing domains, built from live public data
(OpenPhish, Tranco) rather than a static Kaggle dataset. Trains a real
supervised classifier instead of relying on rules or an LLM call.

## What it does

Extracts structural features from a domain name (length, brand similarity,
suspicious TLD, keyword patterns), scores it with a trained model, and
returns a phishing probability via a REST API and a Streamlit demo UI.
Supports multiple model architectures (Gradient Boosting, Random Forest,
XGBoost) and multiple feature profiles (`minimal3`, `core6`, `extended9`)
for side-by-side comparison.

## Results

- Baseline comparison: Logistic Regression, Random Forest, Gradient
  Boosting, SVM, Decision Tree, XGBoost → **Gradient Boosting** selected
  for production
- Tuned via GridSearchCV (5-fold stratified CV)
- Test set: **ROC-AUC 0.92**, F1 0.73 (phishing class)
- Full EDA, PCA, and DBSCAN outlier analysis in `notebooks/eda.ipynb`

Metrics are regenerated automatically on every training run (see
[Model monitoring](#model-monitoring)) and reflect the currently deployed
model, not a fixed snapshot.

## Fixing shortcut learning

Early versions of the model relied almost entirely on domain length (66%
feature importance) while ignoring `keywords` and `suspicious_tld` (0%) —
a classic case of shortcut learning, caused by these features appearing too
rarely in the original dataset. The feature config was expanded to reflect
modern phishing patterns (crypto wallet terms, free hosting platforms),
which rebalanced the dataset and brought every feature's contribution to
≥10%.

## Configuration sources

Values in `config/features.json` are informed by public threat-intel
references rather than picked arbitrarily:
- Suspicious TLDs — [Cybercrime Information Center](https://www.cybercrimeinfocenter.org/top-20-tlds-by-malicious-phishing-domains)
- Most-impersonated brands — [Check Point Research Brand Phishing Report](https://blog.checkpoint.com/research/which-brands-are-impersonated-most-inside-the-q2-2026-brand-phishing-report/)
- Common phishing keywords — [Expel: Top Phishing Keywords](https://expel.com/blog/top-phishing-keywords/)

## Known limitations

- Trained mainly on brand-impersonation phishing (dominant pattern in
  OpenPhish); doesn't reliably flag generic suspicious names unrelated
  to the monitored brands.
- No WHOIS or page-content analysis — structural domain features only.
- `crt.sh` brand-monitoring module (`src/data/crtsh.py`) is implemented
  but excluded from the production pipeline due to frequent upstream
  outages.
- Trained models (`models/**/*.joblib`) are gitignored as generated
  artifacts — must be produced locally before serving predictions.
- Training data volatility — phishing examples are pulled live from
  OpenPhish at build time, so model quality varies between runs
  (observed range: ~200-360 phishing examples per build).
- Some OpenPhish snapshots contain many near-duplicate domains from the
  same automated phishing campaign (e.g. shared naming patterns on
  `pages.dev`). Since `train_test_split` splits randomly rather than by
  campaign, this can inflate apparent test-set performance in some runs —
  see `notebooks/eda.ipynb` for a documented example. Grouping by campaign
  identity before splitting is a candidate future improvement.
- Model architecture affects stability on this dataset size — Random
  Forest showed noticeably less stable predictions on clear-cut legitimate
  domains than Gradient Boosting/XGBoost in manual testing.

## Run it

```bash
uv sync

uv run app/_build_dataset.py
uv run app/_train_models.py
```

Then build and run the API:

```bash
docker compose up --build
```

Open http://localhost:8000/docs

### Streamlit demo

```bash
uv run streamlit run app/_streamlit_app.py
```

Open http://localhost:8501 — lets you pick a model and feature profile,
and check any domain interactively.

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

Input is validated (must look like a real domain, e.g. `example.com`) —
malformed requests are rejected with a `422` before reaching the model.

## Model monitoring

GET /api/model/metrics


Serves the latest training metrics from `models/metrics.json`, refreshed
on every training run.

## CI/CD

Every push runs the full test suite via GitHub Actions on a clean Ubuntu
environment, using the locked dependency set (`uv sync --frozen`) to catch
environment drift, not just logic errors.

## Tech stack

Python 3.13, uv, pandas, scikit-learn, XGBoost, FastAPI, Streamlit, Docker,
pytest, GitHub Actions

## Project structure

app/ # entrypoint scripts (run these — not importable modules)
├── _build_dataset.py # fetch data + extract features + save dataset.csv
├── _train_models.py # train + tune the production model
├── _train_profiles.py # train Gradient Boosting across feature profiles
└── _streamlit_app.py # interactive demo UI
src/ # library code — no side effects on import
├── data/ # OpenPhish, Tranco, crt.sh fetchers
├── features/ # feature extraction + dataset building
├── ml/ # training, tuning, evaluation, prediction
└── api/ # FastAPI app
notebooks/eda.ipynb # EDA, PCA, DBSCAN outlier analysis
tests/ # unit tests
config/
├── features.json # brand list, suspicious TLDs, keywords
└── feature_profiles.json # named feature-column subsets
.github/workflows/ # CI pipeline