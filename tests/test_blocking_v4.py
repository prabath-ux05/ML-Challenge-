"""Unit tests for Phase 9 targeted V4 blocker improvements."""

import pytest
from business_entity_resolution.blocking.candidate_generator import (
    CandidateGenerator,
    STRATEGY_COMBINED_HYBRID,
)


def test_v4_acronym_short_code_blocking():
    """Verify acronym and short-code candidates are retrieved via V4 acronym indexing."""
    records = [
        ("C1", "SC", "Plot 42, Sector 18, Gurgaon", "India"),
        ("C2", "Unrelated Firm Ltd", "100 Other Road, Mumbai", "India"),
    ]
    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=50)
    gen.fit(records)

    # Query with full multi-word name
    cands = gen.generate_candidates(
        name="Shree Constructions Private Limited",
        address="Sector 18, Gurgaon",
        country="India",
        k=10,
    )
    assert "C1" in cands


def test_v4_multi_address_token_boosting():
    """Verify that candidates with multi-token address co-occurrence outrank single-word distractors."""
    records = [
        ("C_distractor", "Seven Tech Enterprises", "Random Location Road, Delhi", "India"),
        ("C_true_match", "ಕನ್ನಡ ಹೆಸರು Infotech", "No 360 Dr Rajkumar Road Rajajinagar Bangalore", "India"),
    ]
    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=50)
    gen.fit(records)

    # Query has 'Seven' and same address as C_true_match
    scored, evidence = gen._query_combined_hybrid_scored(
        name="Seven Infotech Pvt Ltd",
        address="No. 360A, Dr. Rajkumar Road, Rajajinagar, Bangalore",
        country="India",
        limit=10,
    )
    cands = [cid for cid, _ in scored]
    assert "C_true_match" in cands
    # C_true_match should outrank distractor due to address co-occurrence
    assert cands.index("C_true_match") < cands.index("C_distractor")


def test_v4_pure_ngram_noise_capping():
    """Verify that pure unanchored n-gram distractors are capped to prevent crowding out."""
    records = [
        ("C_pure_ngram", "International Global Corporation", "Completely Different Address, Texas", "US"),
        ("C_address_anchor", "Totally Different Name LLC", "123 Main Street, Suite 400, Austin", "US"),
    ]
    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=50)
    gen.fit(records)

    scored, evidence = gen._query_combined_hybrid_scored(
        name="International Corp",
        address="123 Main Street, Suite 400, Austin",
        country="US",
        limit=10,
    )
    # Check that pure ngram score on C_pure_ngram is dampened compared to strong address anchor
    score_dict = dict(scored)
    assert score_dict["C_address_anchor"] > score_dict["C_pure_ngram"]


def test_v4_1_compound_concatenated_name_blocking():
    """Verify that concatenated domain names (e.g. Footanklecom) match multi-token queries."""
    records = [
        ("C_domain", "Footanklecom", "2407 Wrod Ave, Kansas City, Kansas", "US"),
        ("C_other", "General Associates", "100 Other Road, New York", "US"),
    ]
    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=50)
    gen.fit(records)

    cands = gen.generate_candidates(
        name="Foot & Ankle Associates Inc",
        address="2407 Wood Avenue, Kansas City, KS",
        country="US",
        k=10,
    )
    assert "C_domain" in cands
