![CI Pipeline](https://github.com/michallitwin/phishguard-live/actions/workflows/tests.yml/badge.svg)

# PhishGuard Live

ML system detecting phishing domains from live public data (OpenPhish, Tranco) — a trained classifier, not a static dataset or an LLM wrapper.

**[Diagram placeholder: data flow — OpenPhish/Tranco → feature extraction → model → REST API / Streamlit]**

## What it does

Scores a domain's phishing probability using 106 features (6 structural: length, brand similarity, suspicious TLD, keywords + 100 TF-IDF character n-grams), via a REST API. A Streamlit demo lets you compare 3 model architectures × 3 feature profiles × with/without TF-IDF (18 variants) side by side.

## Quick start

```bash
uv sync
uv run app/_build_dataset.py
uv run app/_train_models.py
docker compose up --build
```
API docs: http://localhost:8000/docs

Streamlit demo:
```bash
uv run app/_train_profiles.py   # one-time
uv run streamlit run app/_streamlit_app.py
```

## API usage

```bash
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{"domain": "paypal-verify-login.tk"}'
```
→ `{"domain": "...", "prediction": "phishing", "phishing_probability": 0.94}`

Threshold: 0.25 (not 0.5) — missing phishing is costlier than a false alarm.

## Results

- **ROC-AUC 0.93, F1 0.80** (production: Gradient Boosting, 106 features)
- **89% accuracy** on a 19-domain manual regression test
- Full EDA, PCA, DBSCAN analysis: `notebooks/eda.ipynb`

## Architecture

- **REST API** — one benchmarked, production model, fixed threshold.
- **Streamlit** — 18 exploratory models for architecture/feature comparison, not used by the API.

## Key engineering decisions (details in `notebooks/eda.ipynb` and interview-prep doc)

- **Shortcut learning fixed**: model relied 66% on domain length alone; expanded keyword/TLD config rebalanced it.
- **TF-IDF added**, then an artifact was caught and fixed: 22% of phishing domains had a stray "www." prefix the legit set never had — removed at the source, not just in features.
- **Regularization** (`max_features`, `subsample`, `reg_alpha`/`reg_lambda`) reduced Random Forest's false-positive rate on `google.com` from 63% to 40%.
- **Campaign leakage**: near-duplicate phishing domains from the same automated campaign can inflate test-set scores — documented, not yet fixed (`train_test_split` doesn't group by campaign).

## Known limitations

- No WHOIS or page-content analysis — domain structure only.
- Weak on generic phishing with no brand match and a safe TLD (e.g. `googl3.com`).
- `crt.sh` module implemented but excluded — frequent upstream outages.
- Training data varies run-to-run (live OpenPhish feed, ~200-360 examples).

## Config sources

- TLDs: [Cybercrime Information Center](https://www.cybercrimeinfocenter.org/top-20-tlds-by-malicious-phishing-domains)
- Brands: [Check Point Brand Phishing Report](https://blog.checkpoint.com/research/which-brands-are-impersonated-most-inside-the-q2-2026-brand-phishing-report/)
- Keywords: [Expel: Top Phishing Keywords](https://expel.com/blog/top-phishing-keywords/)

## Tech stack

Python 3.13, uv, scikit-learn, XGBoost, FastAPI, Streamlit, Docker, pytest, GitHub Actions

## Project structure

app/ # entrypoint scripts

├── _build_dataset.py

├── _train_models.py # production model

├── _train_profiles.py # 18 exploratory models

└── _streamlit_app.py

src/ # library code

├── data/ features/ ml/ api/

notebooks/eda.ipynb
tests/
config/ # features.json, feature_profiles.json


## Tests

```bash
uv run pytest -v
```