import pytest
from src.features.extractor import DomainFeatureExtractor, extract_features


@pytest.fixture
def extractor():
    return DomainFeatureExtractor()


def test_domain_length(extractor):
    assert extractor._domain_length("google.com") == 10


def test_count_digits(extractor):
    assert extractor._count_digits("paypal123.com") == 3


def test_has_suspicious_tld(extractor):
    assert extractor._has_suspicious_tld("login.tk") == 1
    assert extractor._has_suspicious_tld("google.com") == 0


def test_extract_features():
    features = extract_features("paypal-verify.tk")
    assert isinstance(features, dict)
    assert features["suspicious_tld"] == 1
    assert features["keywords"] >= 1