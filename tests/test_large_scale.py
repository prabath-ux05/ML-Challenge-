"""Unit tests for Phase 8 large-scale validator and country distribution auditing."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from business_entity_resolution.matching.evaluate_large_scale import LargeScaleValidator


def test_validator_initialization():
    """Verify validator instantiates cleanly with default configuration."""
    validator = LargeScaleValidator(random_seed=42)
    assert validator.random_seed == 42
    assert validator.config is not None


def test_error_categorization_logic():
    """Test false negative and false positive categorization logic."""
    validator = LargeScaleValidator()

    val_s1_ids = {"S1_1", "S1_2", "S1_3"}
    s1_dict = {
        "S1_1": ("S1_1", "Acme Retail Store", "123 Main St", "US"),
        "S1_2": ("S1_2", "Tata Consultancy", "", "India"),
        "S1_3": ("S1_3", "Global Tech Inc", "456 Market St", "US"),
    }
    cand_dict = {
        "C_1": ("C_1", "Acme Retail", "123 Main St", "US"),
        "C_2": ("C_2", "TCS", "", "India"),
        "C_3": ("C_3", "Other Firm", "123 Main St", "US"),
    }
    gt_map = {
        "S1_1": {"C_1"},
        "S1_2": {"C_2"},
        "S1_3": set(),
    }

    df_eval = pd.DataFrame([
        {"source1_id": "S1_1", "candidate_id": "C_1", "proba": 0.85, "is_match": 1},
        {"source1_id": "S1_2", "candidate_id": "C_2", "proba": 0.15, "is_match": 1},  # FN (th=0.20)
        {"source1_id": "S1_3", "candidate_id": "C_3", "proba": 0.90, "is_match": 0},  # FP (th=0.20)
    ])

    errors = validator._categorize_errors(
        df_eval_s2=df_eval,
        df_eval_s3=df_eval,
        val_s1_ids=val_s1_ids,
        s1_dict=s1_dict,
        s2_dict=cand_dict,
        s3_dict=cand_dict,
        gt_s2_map=gt_map,
        gt_s3_map=gt_map,
        th_s2=0.20,
        th_s3=0.80,
    )

    assert "S2" in errors
    assert "S3" in errors
    assert "fn" in errors["S2"]
    assert "fp" in errors["S2"]
    assert len(errors["S2"]["examples"]) > 0
