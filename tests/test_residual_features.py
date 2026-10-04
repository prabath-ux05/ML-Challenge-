"""Tests for Phase 6B residual features (character address sim, fuzzy tokens, locality, missingness interactions)."""

import numpy as np
import pytest

from business_entity_resolution.features.address_features import (
    ADDRESS_FEATURE_NAMES,
    extract_address_features,
)
from business_entity_resolution.features.pair_features import (
    ALL_FEATURE_NAMES,
    INTERACTION_FEATURE_NAMES,
    extract_pair_features,
)


def test_character_level_address_and_typo_tolerance():
    """Verify character-level and 3-gram features catch address typos like 'WESM' vs 'West'."""
    a1 = "53/4, Nehru Nagar West, Coimbatore, Tamil Nadu"
    a2 = "53/4, Nehru Nagar WESM, Coimbatore, Tamil Nadu"

    feats = extract_address_features(a1, a2)

    assert feats["addr_jaro_winkler"] > 0.90
    assert feats["addr_ngram_jaccard"] > 0.80
    assert feats["addr_fuzzy_token_overlap"] >= 5.0
    assert feats["addr_fuzzy_token_jaccard"] > 0.70


def test_fuzzy_address_token_and_token_set_ratio():
    """Verify fuzzy matching and subset containment when address is truncated."""
    full_addr = "431, 441 F/F Front Side, Gali No.5 Guru Ram Dass N, East Delhi, Delhi"
    truncated = "31, East Delhi, DL"

    feats = extract_address_features(full_addr, truncated)

    # Token set ratio should be very high because 'East Delhi DL' is a subset
    assert feats["addr_token_set_ratio"] > 0.70
    assert feats["addr_fuzzy_token_overlap"] >= 2.0


def test_locality_landmark_overlap():
    """Verify landmark and locality tokens are extracted and matched."""
    a1 = "Gat No 229 Village Ranjetaluka Bhor Village Ranje Dist, Pune, Maharashtra"
    a2 = "Plot 4-3 Gat No 229 Village Ranjetaluka Bhor Village Ranje Dist, Pune, MH"

    feats = extract_address_features(a1, a2)

    assert feats["addr_locality_overlap_count"] >= 3.0
    assert feats["addr_locality_jaccard"] > 0.50


def test_address_missingness_interaction_and_scoring():
    """Verify stronger name evidence and confidence weighting when address is omitted."""
    # S1 has address, Candidate has missing address ("" or "__")
    s1 = ("Grubb Best Bancshares LLC", "6704 Centerville Court, Whitsett, NC", "US")
    cand_missing = ("Grubb Best", "__", "US")

    feats = extract_pair_features(
        s1_name=s1[0],
        s1_address=s1[1],
        s1_country=s1[2],
        cand_name=cand_missing[0],
        cand_address=cand_missing[1],
        cand_country=cand_missing[2],
        target_source="S3",
    )

    assert feats["addr_either_missing"] == 1.0
    assert feats["addr_confidence_weight"] == 0.5
    # Name evidence should be boosted via interaction features
    assert feats["name_jaro_x_addr_missing"] > 0.85
    assert feats["name_high_conf_when_addr_missing"] == 1.0
    assert feats["addr_valid_match_score"] == 0.0

    # When address is present on both
    cand_present = ("Grubb Best", "6704 Centerville Court, Whitsett, NC", "US")
    feats_pres = extract_pair_features(
        s1_name=s1[0],
        s1_address=s1[1],
        s1_country=s1[2],
        cand_name=cand_present[0],
        cand_address=cand_present[1],
        cand_country=cand_present[2],
        target_source="S3",
    )
    assert feats_pres["addr_either_missing"] == 0.0
    assert feats_pres["addr_confidence_weight"] == 1.0
    assert feats_pres["name_jaro_x_addr_missing"] == 0.0
    assert feats_pres["addr_valid_match_score"] == 1.0
