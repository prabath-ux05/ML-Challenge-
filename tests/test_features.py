"""Unit and integration tests for pairwise feature engineering."""

import numpy as np
import pytest

from business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
    STRATEGY_COMBINED_HYBRID,
)
from business_entity_resolution.features.address_features import (
    ADDRESS_FEATURE_NAMES,
    extract_address_features,
)
from business_entity_resolution.features.country_features import (
    COUNTRY_FEATURE_NAMES,
    extract_country_features,
)
from business_entity_resolution.features.name_features import (
    NAME_FEATURE_NAMES,
    extract_name_features,
)
from business_entity_resolution.features.pair_features import (
    ALL_FEATURE_NAMES,
    PairFeatureExtractor,
    extract_pair_features,
)


def test_name_features_exact_and_similar():
    """Verify name features for identical, similar, and divergent names."""
    # Identical after normalization
    res_exact = extract_name_features("Acme Corp LLC", "Acme Corporation Inc")
    assert res_exact["name_exact_match"] == 1.0
    assert res_exact["name_token_jaccard"] > 0.0
    assert res_exact["name_levenshtein_sim"] == 1.0

    # Similar with typo
    res_typo = extract_name_features("Payne Enterprises", "Payne Enterpires")
    assert res_typo["name_exact_match"] == 0.0
    assert res_typo["name_raw_levenshtein_sim"] > 0.85
    assert res_typo["name_raw_jaro_winkler"] > 0.90
    assert res_typo["name_ngram_cosine"] > 0.70

    # Completely different
    res_diff = extract_name_features("Acme Corp", "Zeta Logistics")
    assert res_diff["name_exact_match"] == 0.0
    assert res_diff["name_token_jaccard"] == 0.0
    assert res_diff["name_levenshtein_sim"] < 0.30

    # Verify all feature names exist and are floats
    for fname in NAME_FEATURE_NAMES:
        assert fname in res_exact
        assert isinstance(res_exact[fname], float)
        assert np.isfinite(res_exact[fname])


def test_name_features_empty():
    """Verify graceful handling of empty or whitespace business names."""
    res_empty = extract_name_features("", "   ")
    assert res_empty["name_exact_match"] == 0.0
    assert res_empty["name_len_s1"] == 0.0
    assert res_empty["name_levenshtein_sim"] == 0.0
    for fname in NAME_FEATURE_NAMES:
        assert np.isfinite(res_empty[fname])


def test_address_features_missing_and_numeric():
    """Verify address features including missing flags and PIN / numeric extraction."""
    # Missing addresses
    res_miss = extract_address_features("NaN", "")
    assert res_miss["addr_s1_missing"] == 1.0
    assert res_miss["addr_s2_missing"] == 1.0
    assert res_miss["addr_both_missing"] == 1.0
    assert res_miss["addr_exact_match"] == 0.0

    # Address with shared PIN code and street number
    a1 = "3315 Fremont St, Peoria, IL 61604"
    a2 = "3315 Fremont Street Suite 100, Peoria, Illinois 61604"
    res_addr = extract_address_features(a1, a2)
    assert res_addr["addr_s1_missing"] == 0.0
    assert res_addr["addr_has_shared_pincode"] == 1.0  # 61604
    assert res_addr["addr_has_shared_street_number"] == 1.0  # 3315
    assert res_addr["addr_num_jaccard"] > 0.0
    assert res_addr["addr_token_jaccard"] > 0.50

    # Verify all feature names exist and are floats
    for fname in ADDRESS_FEATURE_NAMES:
        assert fname in res_addr
        assert isinstance(res_addr[fname], float)
        assert np.isfinite(res_addr[fname])


def test_country_features():
    """Verify country matching, locale indicator, and missing features."""
    res_us = extract_country_features("US", "US")
    assert res_us["country_exact_match"] == 1.0
    assert res_us["country_is_us"] == 1.0
    assert res_us["country_is_india"] == 0.0
    assert res_us["country_either_missing"] == 0.0

    res_india = extract_country_features("India", "india")
    assert res_india["country_exact_match"] == 1.0
    assert res_india["country_is_india"] == 1.0

    res_diff = extract_country_features("US", "India")
    assert res_diff["country_exact_match"] == 0.0

    res_miss = extract_country_features("", "US")
    assert res_miss["country_s1_missing"] == 1.0
    assert res_miss["country_either_missing"] == 1.0


def test_extract_pair_features_completeness():
    """Verify that extract_pair_features outputs all features with valid numeric values."""
    s1 = ("Acme Tools", "123 Main St, Austin, TX 78701", "US")
    s2 = ("Acme Tools LLC", "123 Main Street, Austin, TX 78701", "US")

    blocking_meta = {
        "cand_rank": 1.0,
        "cand_rank_reciprocal": 1.0,
        "cand_blocking_score": 1020.0,
        "cand_score_ratio_to_top1": 1.0,
        "total_candidates_for_s1": 50.0,
        "log_total_candidates": np.log(51.0),
        "block_exact_name": 1.0,
        "block_name_token": 1.0,
        "block_address_token": 1.0,
        "block_tfidf_ngram": 1.0,
        "block_evidence_channels_count": 4.0,
    }

    feats = extract_pair_features(
        s1_name=s1[0],
        s1_address=s1[1],
        s1_country=s1[2],
        cand_name=s2[0],
        cand_address=s2[1],
        cand_country=s2[2],
        target_source="S2",
        blocking_meta=blocking_meta,
    )

    assert len(feats) == len(ALL_FEATURE_NAMES)
    for fname in ALL_FEATURE_NAMES:
        assert fname in feats, f"Missing feature: {fname}"
        val = feats[fname]
        assert isinstance(val, (int, float, np.floating))
        assert np.isfinite(val), f"Non-finite value for feature {fname}: {val}"

    assert feats["source_pair_is_s2"] == 1.0
    assert feats["source_pair_is_s3"] == 0.0
    assert feats["block_exact_name"] == 1.0
    assert feats["cand_rank"] == 1.0


def test_pair_feature_extractor_batch():
    """Verify batch DataFrame extraction and absence of data leakage."""
    extractor = PairFeatureExtractor()

    pairs = [
        (
            ("S1-1", "Dahlia Power LLC", "Peoria, IL", "US"),
            ("S2-1", "Dahlia Power Inc", "Peoria, IL", "US"),
            "S2",
            {"cand_rank": 1.0, "cand_blocking_score": 50.0},
        ),
        (
            ("S1-2", "Global Energy Corp", "Houston, TX", "US"),
            ("S2-2", "Solar Dynamics", "Dallas, TX", "US"),
            "S2",
            {"cand_rank": 2.0, "cand_blocking_score": 10.0},
        ),
    ]

    labels = [1, 0]
    df = extractor.extract_batch_dataframe(pairs, include_identifiers=True, labels=labels)

    assert len(df) == 2
    assert "source1_id" in df.columns
    assert "candidate_id" in df.columns
    assert "target_source" in df.columns
    assert "is_match" in df.columns

    # Verify all feature columns are numeric float32
    for col in ALL_FEATURE_NAMES:
        assert col in df.columns
        assert df[col].dtype == np.float32
        assert not df[col].isna().any()

    # Verify labels match
    assert list(df["is_match"]) == [1, 0]


def test_end_to_end_dataset_builder_integration():
    """Verify end-to-end integration between CandidateGenerator and PairFeatureExtractor."""
    candidates_pool = [
        ("S2-1", "Alpha Logistics Inc", "100 Broadway, NY 10001", "US"),
        ("S2-2", "Alpha Supply Chain", "100 Broadway, NY 10001", "US"),
        ("S2-3", "Beta Distribution", "500 West St, NY 10002", "US"),
    ]

    s1_queries = [
        ("S1-1", "Alpha Logistics", "100 Broadway, New York, NY 10001", "US"),
    ]

    gt_map = {
        "S1-1": {"S2-1"},
    }

    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=10)
    gen.fit(candidates_pool)

    extractor = PairFeatureExtractor()
    df = extractor.build_candidate_pair_dataset(
        s1_queries=s1_queries,
        candidate_records=candidates_pool,
        candidate_generator=gen,
        target_source="S2",
        gt_map=gt_map,
        k=5,
    )

    assert len(df) > 0
    assert "source1_id" in df.columns
    assert "candidate_id" in df.columns
    assert "is_match" in df.columns

    # Check that S2-1 has is_match == 1 and non-matches have is_match == 0
    s2_1_rows = df[df["candidate_id"] == "S2-1"]
    assert len(s2_1_rows) == 1
    assert s2_1_rows.iloc[0]["is_match"] == 1

    # Check ranking and score columns
    assert "cand_rank" in df.columns
    assert "cand_blocking_score" in df.columns
    assert "block_evidence_channels_count" in df.columns
    assert s2_1_rows.iloc[0]["cand_rank"] == 1.0
