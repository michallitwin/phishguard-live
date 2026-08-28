"""
tests/test_extractor.py
-------------------------
Unit tests for domain feature extraction.
"""

from src.features.extractor import (
    domain_length,
    count_digits,
    count_hyphens,
    has_suspicious_tld,
    keyword_count,
)


def test_domain_length():
    assert domain_length("paypal.com") == 10
    assert domain_length("mammamiaaa") == 10



def test_count_digits():
    assert count_digits("secure123.com") == 3
    assert count_digits("paypal.com") == 0


def test_count_hyphens():
    assert count_hyphens("paypal-verify-login.tk") == 2
    assert count_hyphens("paypal.com") == 0


def test_has_suspicious_tld():
    assert has_suspicious_tld("paypal-verify.tk") == 1
    assert has_suspicious_tld("paypal.com") == 0


def test_keyword_count():
    assert keyword_count("paypal-verify-login.tk") == 2  # "verify", "login"
    assert keyword_count("paypal.com") == 0