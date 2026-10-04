"""Tests for Phase 6A blocking enhancements (transliteration, number words, city synonyms)."""

import pytest

from business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
    STRATEGY_COMBINED_HYBRID,
)
from business_entity_resolution.normalization.addresses import (
    CITY_SYNONYMS,
    STATE_ABBREVIATIONS,
    extract_address_tokens,
    normalize_address,
)
from business_entity_resolution.normalization.common import (
    normalize_digit_words,
    transliterate_text,
)
from business_entity_resolution.normalization.names import (
    extract_name_tokens,
    normalize_business_name,
)


def test_transliteration_devanagari():
    """Verify Indic/Devanagari scripts are transliterated to Latin."""
    raw = "Royal सूर्य मैनेजमेंट"
    trans = transliterate_text(raw)
    assert "sury" in trans.lower() or "surya" in trans.lower()

    norm = normalize_business_name(raw)
    assert "royal" in norm
    assert "sury" in norm or "surya" in norm


def test_number_word_normalization():
    """Verify ordinal and cardinal words are normalized and leading zeros stripped."""
    text1 = "711 9th Street, Etowah, TN"
    text2 = "Ninth St, Etwah, Tennessee"
    text3 = "0550 Melvin Rd"

    norm1 = normalize_address(text1)
    norm2 = normalize_address(text2)
    norm3 = normalize_address(text3)

    assert "9" in norm1.split()
    assert "9" in norm2.split()
    assert "tn" in norm1.split()
    assert "tn" in norm2.split()
    assert "550" in norm3.split()
    assert "0550" not in norm3.split()


def test_city_synonyms():
    """Verify city and locality synonyms expand or standardize correctly."""
    addr_bombay = "201/D, Aditya, Andheri W, Bombay, Maharashtra"
    addr_mumbai = "201/D, Aditya, Andheri W, Mumbai, Maharashtra"

    norm_b = normalize_address(addr_bombay)
    norm_m = normalize_address(addr_mumbai)
    assert "mumbai" in norm_b
    assert "mumbai" in norm_m

    toks_b = set(extract_address_tokens(addr_bombay))
    toks_m = set(extract_address_tokens(addr_mumbai))
    assert "mumbai" in toks_b
    assert "bombay" in toks_m or "mumbai" in toks_m


def test_state_abbreviations_comprehensive():
    """Verify both US and Indian states map to standard 2-letter codes."""
    assert "tn" in normalize_address("Etowah, Tennessee").split()
    assert "pa" in normalize_address("Telford, Pennsylvania").split()
    assert "rj" in normalize_address("Jaipur, Rajasthan").split()
    assert "dl" in normalize_address("East Delhi, Delhi").split()


def test_candidate_generator_recovers_phase5_failures():
    """Verify CandidateGenerator successfully retrieves Phase 5 failure archetypes."""
    generator = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=50)

    # Index targets
    records = [
        ("S2-001", "LUTHER LUTHER AND-LOURENCO", "0550 MELVIN RD, TELFORD, PA", "US"),
        ("S3-002", "Royal सूर्य मैनेजमेंट", "605, Mumbai City, Mumbai, MH", "IN"),
        ("S3-003", "Oncology (Medicine)", "Ninth St, Etwah, Tennessee", "US"),
    ]
    generator.fit(records)

    # Query 1: Number zero stripping ('550' vs '0550') and state match ('PA')
    cands_1 = generator.generate_candidates(
        name="Luther and Lourenco Inc",
        address="550 Melvin Road, Franconia Township, PA",
        country="US",
        k=10,
    )
    assert "S2-001" in cands_1

    # Query 2: Cross-script Devanagari transliteration ('Royal Surya' vs 'Royal सूर्य')
    cands_2 = generator.generate_candidates(
        name="Royal Surya Management",
        address="605, 6Th Floor, Hallmark Business Plaza, Bandra (East), Mumbai, Maharashtra",
        country="IN",
        k=10,
    )
    assert "S3-002" in cands_2

    # Query 3: '9th' vs 'Ninth', 'TN' vs 'Tennessee'
    cands_3 = generator.generate_candidates(
        name="Oncology Medicine Inc",
        address="711 9th Street, Etowah, TN",
        country="US",
        k=10,
    )
    assert "S3-003" in cands_3
