"""Unit and integration tests for production pipeline and submission validation."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from business_entity_resolution.pipeline.runner import PipelineConfig, ProductionPipelineRunner
from business_entity_resolution.pipeline.validator import (
    CANDIDATE_PAIRS_HEADER,
    MATCHING_RESULTS_HEADER,
    SubmissionValidator,
)
from business_entity_resolution.utils.io import write_tsv


def test_submission_validator_valid_case():
    """Verify SubmissionValidator passes on a completely valid submission with zero and multi matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        m_file = tmp_path / "matching_results.tsv"
        c_file = tmp_path / "candidate_pairs.tsv"

        # Candidate pairs
        df_c = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1"},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2"},
            {"source1_entity_id": "S1-3", "candidate_entity_id": "S2-3"},
        ])
        write_tsv(df_c, c_file)

        # Matching results
        df_m = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2"},
            {"source1_entity_id": "S1-3", "matched_entity_ids": ""},  # Zero match allowed
        ])
        write_tsv(df_m, m_file)

        expected_s1 = {"S1-1", "S1-2", "S1-3"}
        valid_cands = {"S2-1", "S2-2", "S2-3", "S3-1"}

        report = SubmissionValidator.validate_matching_results(
            matching_results_path=m_file,
            candidate_pairs_path=c_file,
            expected_s1_ids=expected_s1,
            valid_candidate_ids=valid_cands,
        )

        assert report.is_valid is True
        assert len(report.errors) == 0
        assert report.num_s1_entities == 3
        assert report.num_matched_s1 == 2
        assert report.num_zero_match_s1 == 1
        assert report.total_matches_produced == 3


def test_submission_validator_detects_errors():
    """Verify SubmissionValidator detects missing S1, duplicate S1, self-matches, and invalid candidate IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        m_file = tmp_path / "matching_results.tsv"
        c_file = tmp_path / "candidate_pairs.tsv"

        # Candidate pairs
        df_c = pd.DataFrame([
            {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1"},
            {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2"},
        ])
        write_tsv(df_c, c_file)

        # Invalid matching results:
        # - S1-1 is duplicated
        # - S1-2 matches S1-2 (self-match) and S2-999 (not in candidates)
        # - S1-3 is missing entirely
        df_m = pd.DataFrame([
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S2-1"},  # duplicate in list
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1"},       # duplicate S1 row
            {"source1_entity_id": "S1-2", "matched_entity_ids": "S1-2,S2-999"},  # self-match + unlisted candidate
        ])
        write_tsv(df_m, m_file)

        expected_s1 = {"S1-1", "S1-2", "S1-3"}
        valid_cands = {"S2-1", "S2-2"}

        report = SubmissionValidator.validate_matching_results(
            matching_results_path=m_file,
            candidate_pairs_path=c_file,
            expected_s1_ids=expected_s1,
            valid_candidate_ids=valid_cands,
        )

        assert report.is_valid is False
        assert len(report.errors) >= 4

        err_str = " ".join(report.errors)
        assert "Duplicate S1 entity" in err_str
        assert "Duplicate match IDs" in err_str
        assert "Self-match detected" in err_str
        assert "Missing" in err_str or "not present in candidate_pairs" in err_str


def test_end_to_end_production_runner_mini_flow():
    """Verify end-to-end execution of ProductionPipelineRunner on synthetic inputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        s1_file = tmp_path / "s1.tsv"
        s2_file = tmp_path / "s2.tsv"
        s3_file = tmp_path / "s3.tsv"
        out_dir = tmp_path / "output"

        # Synthetic inputs
        df_s1 = pd.DataFrame([
            {"entity_id": "S1-10", "business_name": "Apex Logistics Inc", "business_address": "100 Broadway, NY", "country": "US"},
            {"entity_id": "S1-20", "business_name": "Zenith Healthcare", "business_address": "500 Main St, Dallas, TX", "country": "US"},
        ])
        df_s2 = pd.DataFrame([
            {"entity_id": "S2-10", "business_name": "Apex Logistics LLC", "business_address": "100 Broadway, New York, NY", "country": "US"},
            {"entity_id": "S2-99", "business_name": "Random Corp", "business_address": "999 Other St, Miami, FL", "country": "US"},
        ])
        df_s3 = pd.DataFrame([
            {"entity_id": "S3-10", "business_name": "Apex Logistics International", "business_address": "100 Broadway, Suite 2, NY", "country": "US"},
            {"entity_id": "S3-20", "business_name": "Zenith Healthcare Clinics", "business_address": "500 Main Street, Dallas, TX", "country": "US"},
        ])

        write_tsv(df_s1, s1_file)
        write_tsv(df_s2, s2_file)
        write_tsv(df_s3, s3_file)

        config = PipelineConfig(
            s1_path=s1_file,
            s2_path=s2_file,
            s3_path=s3_file,
            output_dir=out_dir,
            models_dir=Path("models"),
            k=10,
            threshold_s2=0.20,
            threshold_s3=0.80,
            enable_qwen=False,  # Strictly OFF
        )

        runner = ProductionPipelineRunner(config)
        summary = runner.run()

        # Check outputs exist
        assert summary["is_valid"] is True
        assert (out_dir / "candidate_pairs.tsv").exists()
        assert (out_dir / "matching_results.tsv").exists()
        assert (out_dir / "pipeline_manifest.json").exists()

        df_res = pd.read_csv(out_dir / "matching_results.tsv", sep="\t", dtype=str)
        assert list(df_res.columns) == MATCHING_RESULTS_HEADER
        assert set(df_res["source1_entity_id"]) == {"S1-10", "S1-20"}
        assert len(df_res) == 2

        # Check candidate pairs
        df_c = pd.read_csv(out_dir / "candidate_pairs.tsv", sep="\t", dtype=str)
        assert list(df_c.columns) == CANDIDATE_PAIRS_HEADER
        assert len(df_c) > 0
