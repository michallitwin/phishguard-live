from pathlib import Path
import random
import pandas as pd

from src.data.openphish import fetch_phishing_domains
from src.data.tranco import load_tranco_domains
from src.features.extractor import extract_features

OUTPUT_PATH = Path("data/processed/dataset.csv")


LEGIT_SAMPLE_SIZE = 2000


def build_labeled_rows(domains: set[str], label: str) -> list[dict]:
    rows = []
    for domain in domains:
        features = extract_features(domain)
        features["domain"] = domain
        features["label"] = label
        rows.append(features)
    return rows


def build_dataset() -> pd.DataFrame:
    phishing_domains = fetch_phishing_domains()
    tranco_domains = load_tranco_domains()
    
    sample_size = min(len(tranco_domains), LEGIT_SAMPLE_SIZE)
    legit_sample = random.sample(list(tranco_domains), sample_size)

    phishing_rows = build_labeled_rows(phishing_domains, "phishing")
    legit_rows = build_labeled_rows(legit_sample, "legit")

    df = pd.DataFrame(phishing_rows + legit_rows)
    
    df = df.sample(frac=1.0).reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    print("Building dataset...")
    dataset = build_dataset()
    print(f"Built {len(dataset)} rows")
    print(dataset["label"].value_counts())

    save_dataset(dataset)
    print(f"Saved to {OUTPUT_PATH}")    