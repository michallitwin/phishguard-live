import logging
import time

import requests

logger = logging.getLogger(__name__)

CRTSH_URL = "https://crt.sh/"
MONITORED_BRANDS = ["paypal", "allegro", "inpost", "santander", "google"]


def query_crtsh(
    search_term: str, retries: int = 3, timeout: int = 60
) -> list[dict]:
    """
    Queries crt.sh with automatic retry logic on timeout or server errors.
    
    Args:
        search_term (str): The domain or wildcard string to search for (e.g., '%.inpost.pl').
        retries (int, optional): Number of retry attempts. Defaults to 3.
        timeout (int, optional): Request timeout in seconds. Defaults to 60.
        
    Returns:
        list[dict]: A list of dictionaries containing certificate records, or an empty list if all retries fail.
    """
    params = {"q": search_term, "output": "json"}
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                CRTSH_URL, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response.json()

        except (requests.exceptions.RequestException, ValueError) as e:
            logger.warning("[Attempt %d/%d] crt.sh did not respond: %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(3) 

    return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    search_query = "inpost.pl"
    logger.info("Sending query to crt.sh for: %s...", search_query)

    result = query_crtsh(search_query)
    logger.info("Found %d records for '%s'", len(result), search_query)

    if result:
        logger.info("First record: %s", result[0])