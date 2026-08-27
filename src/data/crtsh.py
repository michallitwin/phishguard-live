import time
import requests

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
            print(f"[Attempt {attempt}/{retries}] crt.sh did not respond: {e}")
            if attempt < retries:
                time.sleep(3) 

    return []


if __name__ == "__main__":
    search_query = "%.inpost.pl"
    print(f"Sending query to crt.sh for: {search_query}...")

    result = query_crtsh(search_query)
    print(f"Found {len(result)} records for '{search_query}'")

    if result:
        print("First record:", result[0])