import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"
CACHE_PATH = Path("data/raw/tranco_top1m.csv")


def download_tranco_list() -> pd.DataFrame:
    """
    Downloads the latest Tranco top 1 million domains list and extracts it from the ZIP archive.

    Returns:
        pd.DataFrame: A DataFrame containing 'rank' and 'domain' columns.
    """
    response = requests.get(TRANCO_URL, timeout=15)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_filename = [name for name in z.namelist() if name.endswith(".csv")][0]

        with z.open(csv_filename) as f:
            df = pd.read_csv(f, header=None, names=["rank", "domain"])

    return df

def load_tranco_domains(force_refresh: bool = False) -> set[str]:
    """
    Loads the Tranco domain list from a local cache or downloads it if missing.

    Args:
        force_refresh (bool, optional): If True, forces a fresh download ignoring the cache. Defaults to False.

    Returns:
        set[str]: A set of unique legitimate domains.
    """
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