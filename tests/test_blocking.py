"""Unit tests for candidate blocking strategies and evaluation."""

import pytest
from business_entity_resolution.blocking.candidate_generator import (
    ALL_STRATEGIES,
    CandidateGenerator,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_COMBINED_HYBRID,
    STRATEGY_COUNTRY_NAME,
    STRATEGY_NORMALIZED_EXACT,
    STRATEGY_TFIDF_NGRAM,
    STRATEGY_TOKEN_NAME,
)
from business_entity_resolution.blocking.exact import ExactNameBlocker
from business_entity_resolution.blocking.token import TokenNameBlocker
from business_entity_resolution.blocking.address import AddressTokenBlocker


def test_exact_name_blocker():
    records = [
        ("S2-1", "Acme Corporation"),
        ("S2-2", "Beta Solutions Inc"),
    ]
    blocker = ExactNameBlocker()
    blocker.build_index(records)

    cands = blocker.get_candidates("Acme Inc")
    assert cands == ["S2-1"]

    cands_empty = blocker.get_candidates("Unknown Corp")
    assert cands_empty == []


def test_token_name_blocker():
    records = [
        ("S2-1", "Dahlia Power Reliable Scientific LLC"),
        ("S2-2", "Alpha Logistics"),
    ]
    blocker = TokenNameBlocker()
    blocker.build_index(records)

    cands = blocker.get_candidates("Dahlia Energy")
    assert "S2-1" in cands


def test_address_token_blocker():
    records = [
        ("S2-1", "3315 Fremont Street, Peoria, IL"),
        ("S2-2", "100 Broadway, New York, NY"),
    ]
    blocker = AddressTokenBlocker()
    blocker.build_index(records)

    cands = blocker.get_candidates("3315 Fremont St, Peoria, Illinois")
    assert "S2-1" in cands


def test_candidate_generator_all_strategies():
    records = [
        ("S2-1", "Payne Enterprises Inc", "3315 Fremont Street, Peoria, IL", "US"),
        ("S2-2", "Raj Investments LLP", "6 CIT Colony, Chennai, TN", "India"),
        ("S2-3", "एसएस फूड प्राइवेट लिमिटेड", "AF-0684, Ghaziabad, 9487203", "India"),
    ]

    for strategy in ALL_STRATEGIES:
        gen = CandidateGenerator(strategy=strategy, max_candidates=10)
        gen.fit(records)

        # Test query
        cands = gen.generate_candidates("Payne Enterprises", "3315 Fremont St", "US")
        assert isinstance(cands, list)

        # Hybrid strategy should catch all 3 queries
        if strategy == STRATEGY_COMBINED_HYBRID:
            assert "S2-1" in cands

            # Transliterated Hindi query with English address tokens
            cands_indic = gen.generate_candidates("Ss Food Private Limited", "AF-0684, Ghaziabad", "India")
            assert "S2-3" in cands_indic


def test_hybrid_union_semantics_and_no_hard_intersection():
    """Verify that hybrid blocking uses TRUE UNION semantics across all 4 channels.

    Ensures no true match is discarded due to an incorrect hard intersection between name and address.
    """
    records = [
        # Entity 1: Matches name tokens only, completely different address
        ("S2-A", "Vortex Aerospace Dynamics", "100 Industrial Parkway, Austin, TX", "US"),
        # Entity 2: Matches address tokens only, completely different name
        ("S2-B", "Apex Holdings Limited", "742 Evergreen Terrace, Springfield, OR", "US"),
        # Entity 3: Matches via character 3-gram fuzzy similarity (OCR/typo: 'Vortx Aerspace')
        ("S2-C", "Vortx Aerspace Tech", "99 Random Way, Dallas, TX", "US"),
        # Entity 4: Matches BOTH name token and address token (should rank #1)
        ("S2-D", "Vortex Dynamics Group", "742 Evergreen Terrace, Springfield, OR", "US"),
    ]

    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=10)
    gen.fit(records)

    # Query has name 'Vortex Technologies' and address '742 Evergreen Terrace'
    cands = gen.generate_candidates(
        name="Vortex Technologies",
        address="742 Evergreen Terrace",
        country="US",
    )

    # True UNION: All 4 entities must be retrieved because each matches at least 1 channel!
    assert "S2-A" in cands, "Entity matching name-only must NOT be lost by hard intersection with address"
    assert "S2-B" in cands, "Entity matching address-only must NOT be lost by hard intersection with name"
    assert "S2-C" in cands, "Entity matching fuzzy TF-IDF n-grams must NOT be lost"
    assert "S2-D" in cands, "Entity matching both name and address must be retrieved"

    # Multi-attribute reinforcement: S2-D matches both name + address, so it should rank highest!
    assert cands[0] == "S2-D"


def test_post_union_capping_k():
    """Verify that candidate capping budget K is applied strictly AFTER the union."""
    records = [
        ("E-1", "Alpha Beta Gamma Corp", "100 Main St, Austin, TX", "US"),
        ("E-2", "Alpha Logistics Inc", "200 First Ave, Dallas, TX", "US"),
        ("E-3", "Gamma Enterprises", "300 Second St, Houston, TX", "US"),
        ("E-4", "Beta Holdings", "400 Third Blvd, San Antonio, TX", "US"),
    ]

    gen = CandidateGenerator(strategy=STRATEGY_COMBINED_HYBRID, max_candidates=10)
    gen.fit(records)

    # Sweep K = 1, 2, 3, 4
    for k in [1, 2, 3, 4]:
        cands = gen.generate_candidates(
            name="Alpha Beta Gamma",
            address="500 Elm St",
            country="US",
            k=k,
        )
        assert len(cands) == k
        # E-1 matches all three tokens ('Alpha', 'Beta', 'Gamma'), so it must always be rank 1
        assert cands[0] == "E-1"

