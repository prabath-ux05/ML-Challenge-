"""Unit tests for Qwen selective semantic reranker and model compliance."""

import pytest
import pandas as pd
import numpy as np

from business_entity_resolution.matching.qwen_reranker import (
    AmbiguityWindow,
    QwenPromptBuilder,
    QwenReranker,
    RerankDecision,
    verify_qwen_model_compliance,
    QWEN_MODEL_SPECS,
)


def test_verify_qwen_model_compliance_valid():
    """Verify that compliant Qwen models pass challenge constraints (<=8B params, Apache 2.0)."""
    # 7B model
    res_7b = verify_qwen_model_compliance("Qwen/Qwen2.5-7B-Instruct")
    assert res_7b["is_challenge_compliant"] is True
    assert res_7b["parameters_billion"] <= 8.0
    assert res_7b["license"] in ("Apache 2.0", "MIT")

    # 1.5B model
    res_15b = verify_qwen_model_compliance("Qwen/Qwen2.5-1.5B-Instruct")
    assert res_15b["is_challenge_compliant"] is True
    assert res_15b["parameters_billion"] <= 8.0

    # 0.5B model
    res_05b = verify_qwen_model_compliance("Qwen/Qwen2.5-0.5B-Instruct")
    assert res_05b["is_challenge_compliant"] is True
    assert res_05b["parameters_billion"] <= 8.0


def test_verify_qwen_model_compliance_invalid():
    """Verify that unknown models raise ValueError."""
    with pytest.raises(ValueError, match="not recognized in verified Qwen registry"):
        verify_qwen_model_compliance("NonExistent/Model-99B")


def test_ambiguity_window_bounds_and_filtering():
    """Test AmbiguityWindow boundary calculation and containment."""
    w_narrow = AmbiguityWindow(name="Narrow", lower_delta=0.05, upper_delta=0.05)
    low, high = w_narrow.get_bounds(0.20)
    assert low == 0.15
    assert high == 0.25

    assert w_narrow.is_ambiguous(0.20, 0.20) is True
    assert w_narrow.is_ambiguous(0.16, 0.20) is True
    assert w_narrow.is_ambiguous(0.14, 0.20) is False
    assert w_narrow.is_ambiguous(0.26, 0.20) is False

    # Edge clipping test
    w_wide = AmbiguityWindow(name="Wide", lower_delta=0.30, upper_delta=0.30)
    low_clip, high_clip = w_wide.get_bounds(0.10)
    assert low_clip == 0.0
    assert high_clip == 0.40


def test_qwen_prompt_builder():
    """Test prompt builder formatting."""
    prompt = QwenPromptBuilder.build_prompt(
        s1_name="Tata Consultancy Services",
        s1_addr="TCS House, Raveline Street, Fort, Mumbai",
        s1_country="India",
        cand_name="TCS Ltd",
        cand_addr="Fort, Mumbai",
        cand_country="India",
        lgbm_prob=0.35,
    )
    assert "Tata Consultancy Services" in prompt
    assert "TCS Ltd" in prompt
    assert "Statistical Match Probability (LightGBM): 0.3500" in prompt
    assert "Acronyms & Short Codes" in prompt
    assert "Output JSON:" in prompt


def test_qwen_reranker_unambiguous_passthrough():
    """Test that candidate pairs outside the ambiguity window are NOT sent to Qwen."""
    reranker = QwenReranker(backend="semantic_rules")
    w = AmbiguityWindow(name="Medium", lower_delta=0.10, upper_delta=0.10)

    # Threshold = 0.20, Window = [0.10, 0.30]
    # Score 0.05 is far below window -> direct negative
    dec_neg = reranker.rerank_pair(
        s1_tuple=("S1_1", "ABC Corp", "123 Main St", "US"),
        cand_tuple=("S2_1", "XYZ Ltd", "999 Other St", "US"),
        lgbm_prob=0.05,
        threshold=0.20,
        window=w,
    )
    assert dec_neg.qwen_called is False
    assert dec_neg.is_ambiguous is False
    assert dec_neg.final_prediction == 0
    assert dec_neg.final_proba == 0.05

    # Score 0.85 is far above window -> direct positive
    dec_pos = reranker.rerank_pair(
        s1_tuple=("S1_2", "Acme Tech", "10 Innovation Way", "US"),
        cand_tuple=("S2_2", "Acme Tech Inc", "10 Innovation Way", "US"),
        lgbm_prob=0.85,
        threshold=0.20,
        window=w,
    )
    assert dec_pos.qwen_called is False
    assert dec_pos.is_ambiguous is False
    assert dec_pos.final_prediction == 1
    assert dec_pos.final_proba == 0.85


def test_qwen_reranker_acronym_recovery():
    """Test that ambiguous acronym candidate pairs are evaluated and recovered."""
    reranker = QwenReranker(backend="semantic_rules")
    w = AmbiguityWindow(name="Wide", lower_delta=0.15, upper_delta=0.10)

    # LightGBM prob = 0.16 (below threshold 0.20, so LightGBM alone would predict 0)
    dec = reranker.rerank_pair(
        s1_tuple=("S1_10", "Shree Constructions Private Limited", "Plot 42, Sector 18, Gurgaon", "India"),
        cand_tuple=("S2_10", "SC", "Plot 42, Sector 18, Gurgaon", "India"),
        lgbm_prob=0.16,
        threshold=0.20,
        window=w,
    )
    assert dec.qwen_called is True
    assert dec.is_ambiguous is True
    assert dec.qwen_prediction is True
    assert dec.final_prediction == 1
    assert "Acronym match" in dec.reasoning


def test_qwen_reranker_transliteration_recovery():
    """Test that ambiguous transliteration pairs are evaluated and recovered."""
    reranker = QwenReranker(backend="semantic_rules")
    w = AmbiguityWindow(name="Medium", lower_delta=0.10, upper_delta=0.10)

    # LightGBM prob = 0.18 (below threshold 0.20)
    # Devanagari "श्री गणेश एंटरप्राइजेज" vs English "Shree Ganesh Enterprises"
    dec = reranker.rerank_pair(
        s1_tuple=("S1_11", "श्री गणेश एंटरप्राइजेज", "Shop 5, Station Road, Pune", "India"),
        cand_tuple=("S2_11", "Shree Ganesh Enterprises", "Shop 5, Station Road, Pune", "India"),
        lgbm_prob=0.18,
        threshold=0.20,
        window=w,
    )
    assert dec.qwen_called is True
    assert dec.qwen_prediction is True
    assert dec.final_prediction == 1
    assert "transliteration match" in dec.reasoning.lower()


def test_qwen_reranker_candidate_dataframe():
    """Test batch reranking on a candidate DataFrame."""
    reranker = QwenReranker(backend="semantic_rules")
    w = AmbiguityWindow(name="TestWindow", lower_delta=0.10, upper_delta=0.10)

    df_cands = pd.DataFrame([
        {"source1_id": "S1_1", "candidate_id": "C_1", "proba": 0.05, "is_match": 0},
        {"source1_id": "S1_2", "candidate_id": "C_2", "proba": 0.18, "is_match": 1},
        {"source1_id": "S1_3", "candidate_id": "C_3", "proba": 0.90, "is_match": 1},
    ])

    s1_lookup = {
        "S1_1": ("S1_1", "Alpha Corp", "100 Broadway", "US"),
        "S1_2": ("S1_2", "Bharat Heavy Electricals", "BHEL Complex, Ranipur, Haridwar", "India"),
        "S1_3": ("S1_3", "Gamma Global", "55 Market St", "US"),
    }
    cand_lookup = {
        "C_1": ("C_1", "Zeta Limited", "900 Unknown Way", "US"),
        "C_2": ("C_2", "BHEL", "BHEL Complex, Ranipur, Haridwar", "India"),
        "C_3": ("C_3", "Gamma Global Inc", "55 Market St", "US"),
    }

    df_out, summary = reranker.rerank_candidate_dataframe(
        df_candidates=df_cands,
        s1_lookup=s1_lookup,
        cand_lookup=cand_lookup,
        threshold=0.20,
        window=w,
    )

    assert len(df_out) == 3
    assert "qwen_called" in df_out.columns
    assert "final_proba" in df_out.columns
    assert "final_prediction" in df_out.columns

    # Only row 1 (proba=0.18) is in [0.10, 0.30]
    assert bool(df_out.loc[0, "qwen_called"]) is False
    assert bool(df_out.loc[1, "qwen_called"]) is True
    assert bool(df_out.loc[2, "qwen_called"]) is False

    assert summary["qwen_called_count"] == 1
    assert summary["total_pairs"] == 3
    assert summary["qwen_called_pct"] == pytest.approx(33.33, rel=1e-2)
    assert summary["est_cost_usd"] > 0.0
