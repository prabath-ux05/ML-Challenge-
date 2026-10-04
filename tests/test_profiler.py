"""Unit tests for dataset profiler."""

import pandas as pd
from business_entity_resolution.data.profiler import (
    compute_overlap_stats,
    generate_markdown_report,
    profile_ground_truth_dataset,
    profile_source_dataset,
    write_profiling_report,
    GroundTruthExactStats,
)


def test_profile_source_dataset():
    df = pd.DataFrame({
        "entity_id": ["E-1", "E-2", "E-3", "E-1"],  # 1 duplicate
        "business_name": ["Alpha LLC", "Beta Inc", "", "Alpha LLC"],
        "business_address": ["123 Main", "456 Oak", "789 Pine", "123 Main"],
        "country": ["US", "US", "India", "US"],
    })

    profile, hashes = profile_source_dataset(df, source_name="Test Source", collect_hashes=True)

    assert profile.total_rows == 4
    assert profile.unique_ids == 3
    assert profile.duplicate_id_count == 1
    assert profile.missing_counts["business_name"] == 1
    assert profile.unique_names == 2
    assert profile.country_distribution["US"] == 3
    assert profile.country_distribution["India"] == 1
    assert hashes is not None
    assert len(hashes["name_hashes"]) == 2


def test_profile_ground_truth_dataset():
    df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": [
            "S2-10,S3-20",  # 2 matches (both S2 & S3)
            "S2-11",        # 1 match (S2 only)
            "",             # 0 matches
        ],
    })

    gt_profile = profile_ground_truth_dataset(df, s1_total_count=3)
    assert gt_profile.total_rows == 3
    assert gt_profile.unique_s1_ids == 3
    assert gt_profile.total_matches == 3
    assert gt_profile.zero_match_count == 1
    assert gt_profile.one_match_count == 1
    assert gt_profile.multi_match_count == 1
    assert gt_profile.s2_match_count == 2
    assert gt_profile.s3_match_count == 1
    assert gt_profile.matching_both_s2_s3_count == 1
    assert gt_profile.matching_s2_only_count == 1
    assert gt_profile.matching_neither_count == 1


def test_compute_overlap_stats():
    h_a = {
        "name_hashes": {hash("apple"), hash("banana")},
        "address_hashes": {hash("123 st")},
        "pair_hashes": {hash((hash("apple"), hash("123 st")))},
    }
    h_b = {
        "name_hashes": {hash("banana"), hash("cherry")},
        "address_hashes": {hash("123 st"), hash("456 ave")},
        "pair_hashes": {hash((hash("banana"), hash("123 st")))},
    }

    overlap = compute_overlap_stats("A", h_a, "B", h_b)
    assert overlap.name_overlap_count == 1
    assert overlap.address_overlap_count == 1
    assert overlap.pair_overlap_count == 0


def test_generate_and_write_report(tmp_path):
    df_s = pd.DataFrame({
        "entity_id": ["E-1"],
        "business_name": ["Foo"],
        "business_address": ["Bar"],
        "country": ["US"],
    })
    p_s, h_s = profile_source_dataset(df_s, source_name="Source 1")
    df_gt = pd.DataFrame({
        "source1_entity_id": ["E-1"],
        "matched_entity_ids": ["S2-1"],
    })
    p_gt = profile_ground_truth_dataset(df_gt)
    overlap = [compute_overlap_stats("S1", h_s, "S2", h_s)]
    exact_gt = GroundTruthExactStats(total_true_pairs=1)

    md = generate_markdown_report({"s1": p_s}, p_gt, overlap, exact_gt)
    assert "# Business Entity Resolution: Dataset Profiling Report" in md

    report_file = tmp_path / "test_report.md"
    written = write_profiling_report(md, report_file)
    assert written.exists()
