import requests
from urllib.parse import urlparse


FEED_URL = "https://openphish.com/feed.txt"


def fetch_phishing_urls() -> list[str]:
    """
    Fetches the latest phishing URLs from the OpenPhish feed.

    Returns:
        list[str]: A list of raw phishing URLs.
    """
    response = requests.get(FEED_URL, timeout=15)
    response.raise_for_status()
    return response.text.strip().splitlines()


def extract_domain(url: str) -> str:
    """
    Extracts the domain name from a given URL, removing any port numbers.

    Args:
        url (str): The full URL string.

    Returns:
        str: The extracted domain name.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    domain = domain.split(":")[0]
    return domain


def fetch_phishing_domains() -> set[str]:
    """
    Retrieves a set of unique phishing domains from the OpenPhish feed.

    Returns:
        set[str]: A collection of unique phishing domains.
    """
    urls = fetch_phishing_urls()
    domains = {extract_domain(url) for url in urls} 
    return domains

