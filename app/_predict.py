import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.predict import load_artifacts, score_domain

model, le = load_artifacts()

test_domains = [
    "paypal.com",
    "paypal-verify-login-secure.tk",
    "google.com",
    "gooogle-account-update.xyz",
    "wikipedia.org",
    "scam.xyz",
]

for domain in test_domains:
    result = score_domain(domain, model, le)
    print(result)