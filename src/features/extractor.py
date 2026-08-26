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
    return len(domain)


def count_digits(domain: str) -> int:
    return sum(c.isdigit() for c in domain)


def count_hyphens(domain: str) -> int:
    return domain.count("-")


def brand_similarity(domain: str) -> float:
    domain_core = domain.split(".")[0]
    return max(Levenshtein.ratio(domain_core, brand) for brand in KNOWN_BRANDS)


def has_suspicious_tld(domain: str) -> int:
    tld = domain.split(".")[-1]
    return 1 if tld in SUSPICIOUS_TLDS else 0


def keyword_count(domain: str) -> int:
    return sum(1 for kw in PHISHING_KEYWORDS if kw in domain)


def extract_features(domain: str) -> dict:
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
