"""Unit tests for text, name, and address normalization."""

from business_entity_resolution.normalization.common import clean_text, strip_accents, tokenize
from business_entity_resolution.normalization.names import extract_name_tokens, normalize_business_name
from business_entity_resolution.normalization.addresses import extract_address_tokens, normalize_address


def test_strip_accents():
    assert strip_accents("Léarning") == "Learning"
    assert strip_accents("Café") == "Cafe"
    assert strip_accents("") == ""


def test_clean_text():
    assert clean_text("  Payne   Énterprises, Inc.!  ") == "payne enterprises inc"
    assert clean_text("B+ Retail Inc") == "b retail inc"


def test_tokenize():
    tokens = tokenize("Maure Williams Colombier Inc.")
    assert "maure" in tokens
    assert "williams" in tokens
    assert "colombier" in tokens


def test_normalize_business_name():
    # Legal terms stripped
    assert normalize_business_name("Payne Enterprises, Inc.") == "payne"
    assert normalize_business_name("Dahlia Power Reliable Scientific LLC") == "dahlia power reliable scientific"
    # Domain names
    assert normalize_business_name("www.wilfordhancock.com") == "wilfordhancock"
    # Accents
    assert normalize_business_name("LLC Moncada Léarning Center") == "moncada learning"


def test_extract_name_tokens():
    tokens = extract_name_tokens("Payne Enterprises LLC")
    assert tokens == ["payne"]


def test_normalize_address():
    norm = normalize_address("3315 Fremont Street, Peoria, IL")
    assert "3315" in norm
    assert "fremont" in norm
    assert "st" in norm

    norm_state = normalize_address("85 Wayne Avenue, Ticonderoga, New York")
    assert "ny" in norm_state


def test_extract_address_tokens():
    tokens = extract_address_tokens("6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu")
    assert "6" in tokens or "29" in tokens
    assert "chennai" in tokens
