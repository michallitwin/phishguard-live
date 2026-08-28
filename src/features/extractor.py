import Levenshtein
import json
from typing import Any
from pathlib import Path

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "features.json"
)

class DomainFeatureExtractor:
    """Extracts numerical and lexical features from domain strings."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.known_brands, self.suspicious_tlds, self.phishing_keywords = (
            self._load_config()
        )

    def _load_config(self) -> tuple[list[str], set[str], list[str]]:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}"
            )
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return (
            data.get("known_brands", []),
            set(data.get("suspicious_tlds", [])),
            data.get("phishing_keywords", []),
        )
    def _domain_length(self, domain: str) -> int:
        return len(domain)

    def _count_digits(self, domain: str) -> int:
        return sum(c.isdigit() for c in domain)

    def _count_hyphens(self, domain: str) -> int:
        return domain.count("-")

    def _brand_similarity(self, domain: str) -> float:
        domain_core = domain.split(".")[0]
        if not self.known_brands:
            return 0.0
        return max(
            Levenshtein.ratio(domain_core, brand) for brand in self.known_brands
        )

    def _has_suspicious_tld(self, domain: str) -> int:
        tld = domain.split(".")[-1]
        return 1 if tld in self.suspicious_tlds else 0

    def _keyword_count(self, domain: str) -> int:
        return sum(1 for kw in self.phishing_keywords if kw in domain)

    def extract(self, domain: str) -> dict[str, Any]:
        """Extracts all features for a given domain."""
        return {
            "length": self._domain_length(domain),
            "digits": self._count_digits(domain),
            "hyphens": self._count_hyphens(domain),
            "brand_sim": round(self._brand_similarity(domain), 4),
            "suspicious_tld": self._has_suspicious_tld(domain),
            "keywords": self._keyword_count(domain),
        }


_extractor = DomainFeatureExtractor()


def extract_features(domain: str) -> dict[str, Any]:
    """Helper function for backward compatibility with existing pipeline."""
    return _extractor.extract(domain)