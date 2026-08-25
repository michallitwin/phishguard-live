import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
CACHE_PATH = Path("data/raw/tranco_top1m.csv")


def download_tranco_list() -> pd.DataFrame:
    response = requests.get(TRANCO_URL, timeout=15)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = [name for name in z.namelist() if name.endswith(".csv")][0]

        with z.open(csv_filename) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])

    return df

def load_tranco_domains(force_refresh: bool = False) -> set[str]:
    if CACHE_PATH.exists() and not force_refresh:
        df = pd.read_csv(CACHE_PATH)
    else:
        df = download_tranco_list()
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)

    return set(df["domain"].dropna().astype(str))


if __name__ == "__main__":
    domains = load_tranco_domains()

    print(f"Loaded {len(domains)} domains from Tranco")
    print("Sample:", list(domains)[:5])