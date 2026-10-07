import sys
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.predict import load_artifacts, score_domain


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "models" / "test_manual_eval_log.json"


TEST_SET = [
    # --- phishing: SHORT (testing if length bias causes misses) ---
    ("paypal.scam.xyz", "phishing"),
    ("apple.xyz.login", "phishing"),
    ("google-scam.top", "phishing"),
    ("chatgpt.buzz", "phishing"),
    ("googl3.scam", "phishing"),
    ("paypal.cfd", "phishing"),

    # --- phishing: LONGER (classic pattern, for comparison) ---
    ("paypal-verify-login-secure.tk", "phishing"),
    ("gooogle-account-update.scam", "phishing"),
    ("microsoft-support-portal-verify.top", "phishing"),
    ("apple-id-locked-verify-now.xyz", "phishing"),

    # --- legit: SHORT, well-known---
    ("google.com", "legit"),
    ("apple.com", "legit"),
    ("paypal.com", "legit"),
    ("github.com", "legit"),
    ("mozilla.com", "legit"),

    # --- legit: LONGER, well-known  ---
    ("wikipedia.org", "legit"),
    ("developer.mozilla.org", "legit"),
    ("stackoverflow.com", "legit"),
    ("microsoft.com", "legit"),
]

model, le, vectorizer = load_artifacts()

results = []
correct = 0

for domain, expected in TEST_SET:
    result = score_domain(domain, model, le, vectorizer)
    predicted = result["prediction"]
    is_correct = predicted == expected
    correct += is_correct

    marker = "✅" if is_correct else "❌"
    print(f"{marker} {domain:50s} expected={expected:10s} got={predicted:10s} ({result['phishing_probability']:.1%})")


    results.append({
        "domain": domain,
        "expected": expected,
        "predicted": predicted,
        "phishing_probability": result["phishing_probability"],
        "correct": is_correct,
    })

accuracy = correct / len(TEST_SET)
print(f"\nAccuracy on manual test set: {correct}/{len(TEST_SET)} ({accuracy:.1%})")

log_entry = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "accuracy": round(accuracy, 4),
    "results": results,
}

log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
log.append(log_entry)
LOG_PATH.write_text(json.dumps(log, indent=2))
print(f"Logged to {LOG_PATH}")