import Levenshtein

KNOWN_BRANDS = [
    "paypal",
    "google",
    "microsoft",
    "allegro",
    "inpost",
    "santander",
    "apple",
]
SUSPICIOUS_TLDS = {"tk", "xyz", "top", "click", "gq", "work", "buzz"}
PHISHING_KEYWORDS = [
    "login",
    "verify",
    "secure",
    "account",
    "update",
    "confirm",
    "signin",
    "scam",
]


def domain_length(domain: str) -> int:
    """
    Calculates the total length of the domain string.
    """
    return len(domain)


def count_digits(domain: str) -> int:
    """
    Counts the number of numerical digits in the domain.
    """
    return sum(c.isdigit() for c in domain)


def count_hyphens(domain: str) -> int:
    """
    Counts the number of hyphens in the domain.
    """
    return domain.count("-")


def brand_similarity(domain: str) -> float:
    """
    Calculates the maximum Levenshtein similarity ratio against known brands.
    """
    domain_core = domain.split(".")[0]
    return max(Levenshtein.ratio(domain_core, brand) for brand in KNOWN_BRANDS)


def has_suspicious_tld(domain: str) -> int:
    """
    Checks if the domain ends with a known suspicious TLD.
    
    Returns:
        int: 1 if suspicious, 0 otherwise.
    """
    tld = domain.split(".")[-1]
    return 1 if tld in SUSPICIOUS_TLDS else 0


def keyword_count(domain: str) -> int:
    """
    Counts the number of phishing-related keywords present in the domain.
    """
    return sum(1 for kw in PHISHING_KEYWORDS if kw in domain)


def extract_features(domain: str) -> dict:
    """
    Extracts and aggregates a dictionary of numerical features for a given domain.
    
    Args:
        domain (str): The domain name to analyze.
        
    Returns:
        dict: A dictionary of calculated features.
    """
    return {
        "length": domain_length(domain),
        "digits": count_digits(domain),
        "hyphens": count_hyphens(domain),
        "brand_sim": round(brand_similarity(domain), 4),
        "suspicious_tld": has_suspicious_tld(domain),
        "keywords": keyword_count(domain),
    }


if __name__ == "__main__":
    test_domains = [
        "paypal.com",
        "paypal-verify-login.tk",
        "google.com",
        "gooogle-secure.xyz",
    ]
    for d in test_domains:
        print(d, "->", extract_features(d))
