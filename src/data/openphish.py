import requests
from urllib.parse import urlparse


FEED_URL = "https://openphish.com/feed.txt"


def fetch_phishing_urls() -> list[str]:
    response = requests.get(FEED_URL, timeout=15)
    response.raise_for_status()
    return response.text.strip().splitlines()


def extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = parsed.netloc
    domain = domain.split(":")[0]
    return domain


def fetch_phishing_domains() -> set[str]:
    urls = fetch_phishing_urls()
    domains = {extract_domain(url) for url in urls} 
    return domains


if __name__ == "__main__":
    domains = fetch_phishing_domains()
    print(f"Downloaded {len(domains)} unique phishing domains")
    for d in list(domains)[:10]:
        print(" -", d)