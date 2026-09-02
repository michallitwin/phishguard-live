import pytest

from src.features.extractor import DomainFeatureExtractor, extract_features


@pytest.fixture
def extractor():
    return DomainFeatureExtractor()



def test_extract_returns_all_expected_keys(extractor):
    features = extractor.extract("paypal.com")
    expected_keys = {
        "length", "digits", "hyphens",
        "brand_sim", "brand_sim_jaro", "brand_sim_seqmatch", "brand_sim_ngram",
        "suspicious_tld", "keywords",
    }
    assert set(features.keys()) == expected_keys


def test_domain_length_via_extract(extractor):
    features = extractor.extract("google.com")
    assert features["length"] == 10


def test_digit_counting_via_extract(extractor):
    features = extractor.extract("paypal123.com")
    assert features["digits"] == 3


def test_suspicious_tld_detected(extractor):
    features = extractor.extract("login.tk")
    assert features["suspicious_tld"] == 1


def test_safe_tld_not_flagged(extractor):
    features = extractor.extract("google.com")
    assert features["suspicious_tld"] == 0


def test_known_phishing_pattern_scores_high_brand_similarity(extractor):
    features = extractor.extract("paypal-verify-login.tk")
    assert features["brand_sim"] > 0.4
    assert features["keywords"] >= 1
    assert features["suspicious_tld"] == 1


def test_extract_features_backward_compatible_function():
    features = extract_features("paypal-verify.tk")
    assert isinstance(features, dict)
    assert features["suspicious_tld"] == 1
    assert features["keywords"] >= 1