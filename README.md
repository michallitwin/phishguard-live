![CI Pipeline](https://github.com/michallitwin/phishguard-live/actions/workflows/tests.yml/badge.svg)

# PhishGuard Live

ML system for detecting phishing domains, built from live public data
(OpenPhish, Tranco) rather than a static Kaggle dataset. Trains a real
supervised classifier instead of relying on rules or an LLM call.

## What it does

Extracts structural features from a domain name (length, brand similarity,
suspicious TLD, keyword patterns) plus 100 character n-gram (TF-IDF)
features learned automatically from the training data, scores it with a
trained model, and returns a phishing probability via a REST API. A
separate Streamlit demo lets you interactively compare different model
architectures (Gradient Boosting, Random Forest, XGBoost) and feature
profiles (`minimal3`, `core6`, `extended9`) side by side.

## Architecture: production vs. exploration

- **REST API** (`src/api/`) serves a **single, benchmarked model**
  (currently Gradient Boosting) for predictable, consistent production
  behavior.
- **Streamlit demo** (`app/_streamlit_app.py`) loads a separate set of
  9 models (3 architectures × 3 feature profiles, trained via
  `app/_train_profiles.py`) purely for exploration and comparison — not
  used in the API.

## Results (production model)

- Baseline comparison: Logistic Regression, Random Forest, Gradient
  Boosting, SVM, Decision Tree, XGBoost → **Gradient Boosting** selected
- Tuned via GridSearchCV (5-fold stratified CV)
- 106 total features: 6 structured + 100 TF-IDF character n-grams
- Test set: **ROC-AUC 0.93**, F1 0.80 (phishing class)
- Manual regression test set (19 known domains): **89% accuracy**
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
which brought `suspicious_tld` and `keywords` into meaningful use (10%
and 3%, up from 0% for both). Domain length remains the single strongest
predictor (55%), which is partly justified — brand-impersonation phishing
in this dataset tends to be systematically longer than legitimate domains
(e.g. `paypal-verify-login-secure.tk` vs. `paypal.com`) — but is a
documented limitation: short, low-effort phishing domains (e.g.
`scam.xyz`) remain harder for the model to catch. Reducing this reliance
further (e.g. via `max_features`/`subsample` tuning) was considered but
not pursued, to avoid trading real detection accuracy for a more evenly
distributed importance chart.

A related bug was found and fixed during this process: `suspicious_tld`
detection only checked the last dot-separated segment of a domain, so
multi-part suspicious hosts like `pages.dev` or `blogspot.com` were never
matched. Fixed by checking whether the domain ends with any configured
suffix instead of an exact last-segment match.

## Adding TF-IDF character n-grams

To improve detection of short phishing domains without a strong brand
match (e.g. `googl3.scam`), 100 character n-gram (2-4 chars) TF-IDF
features were added alongside the existing structured features, fit only
on training-set domains to avoid data leakage into the test set.

While validating the resulting model on a fixed 19-domain manual
regression test (`app/_manual_eval.py`), an initial run showed a large
jump in accuracy (78.9% → 94.7%). Investigating the top-weighted n-grams
revealed a data collection artifact: 22.1% of phishing domains contained
a literal "www." prefix versus only 0.05% of legitimate domains — a side
effect of phishing domains being extracted from full URLs (`urlparse`)
while legitimate domains come from Tranco's bare-domain list. Fixed by
stripping the "www." prefix during domain extraction and rebuilding the
dataset. The verified, artifact-free result: accuracy improved from 68.4%
to **89%** on the manual test set, a genuine gain rather than an inflated one.

TF-IDF features are included both in the production model and in the
Streamlit exploration set, fit independently per feature profile.

## Configuration sources

Values in `config/features.json` are informed by public threat-intel
references rather than picked arbitrarily:
- Suspicious TLDs/hosts — [Cybercrime Information Center](https://www.cybercrimeinfocenter.org/top-20-tlds-by-malicious-phishing-domains)
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
  OpenPhish at build time, so model quality varies between runs.
- Some OpenPhish snapshots contain near-duplicate domains from the same
  automated phishing campaign (shared naming patterns, e.g. on
  `pages.dev`). Since `train_test_split` splits randomly rather than by
  campaign, this can inflate apparent test-set performance in some runs —
  see `notebooks/eda.ipynb` for a documented example.
-  Model architecture and feature profile combinations vary significantly in
  reliability. Random Forest on the `extended9` profile, for example, flags
  `google.com` as 63% likely phishing — a clear false positive — while the
  production Gradient Boosting model scores the same domain at ~2-4%. This
  is a deliberate part of the Streamlit demo: comparing architectures
  side-by-side surfaces real differences in stability that a single
  aggregate metric (F1/AUC) can hide.
  - Brand similarity alone, without a suspicious TLD or keyword to
  reinforce it, is not always enough to trigger a phishing classification
  (e.g. `googl3.com` on a safe `.com` TLD scores as legit on some
  model/profile combinations) — the training data rarely contains
  typosquatting without an accompanying suspicious TLD or keyword, so the
  model under-weights this combination on its own.

## Run it

```bash
uv sync
uv run app/_build_dataset.py
uv run app/_train_models.py
docker compose up --build
```

Open http://localhost:8000/docs

### Streamlit demo (model & feature-profile comparison)

```bash
uv run app/_train_profiles.py   # one-time: trains the 9-model exploration set
uv run streamlit run app/_streamlit_app.py
```

Open http://localhost:8501

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

The classification threshold is set to 0.25, not the default 0.5 — since
missing a phishing domain is more costly than a false alarm, any domain
scoring ≥25% phishing probability is classified as `"phishing"`. This is
specific to the production API; the exploratory Streamlit models use the
standard 0.5 threshold for fair side-by-side comparison across
architectures.

Input is validated (must look like a real domain, e.g. `example.com`) —
malformed requests are rejected with a `422` before reaching the model.

## Model monitoring

GET /api/model/metrics


Serves the latest training metrics from `models/metrics.json`, refreshed
on every training run.

## CI/CD

Every push runs the full test suite via GitHub Actions on a clean Ubuntu
environment, using the locked dependency set (`uv sync --frozen`).

## Tech stack

Python 3.13, uv, pandas, scikit-learn, XGBoost, FastAPI, Streamlit, Docker,
pytest, GitHub Actions

## Project structure

app/ # entrypoint scripts (run these — not importable modules)

├── _build_dataset.py # fetch data + extract features + save dataset.csv

├── _train_models.py # train + tune the SINGLE production model

├── _train_profiles.py # train 9 models (3 architectures x 3 profiles) for demo

└── _streamlit_app.py # interactive model/profile comparison UI

src/ # library code — no side effects on import

├── data/ # OpenPhish, Tranco, crt.sh fetchers

├── features/ # feature extraction + dataset building

├── ml/ # training, tuning, evaluation, prediction

└── api/ # FastAPI app (serves the single production model)

notebooks/eda.ipynb # EDA, PCA, DBSCAN outlier analysis

tests/ # unit tests

config/

├── features.json # brand list, suspicious TLDs/hosts, keywords

└── feature_profiles.json # named feature-column subsets
models/

├── phishing_model.joblib # production model (gitignored, generated)

├── tfidf_vectorizer.joblib # production TF-IDF vectorizer (gitignored, generated)

└── profiles/ # 9 exploratory models + per-profile vectorizers (gitignored, generated)

.github/workflows/ # CI pipeline


## Rebuilding the dataset and retraining

```bash
uv run app/_build_dataset.py
uv run app/_train_models.py
```

## Running tests

```bash
uv run pytest -v
```
