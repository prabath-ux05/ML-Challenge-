"""Comprehensive tests for ML matching, metrics, grouped splitting, and decision logic."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from business_entity_resolution.evaluation.metrics import (
    compute_entity_level_metrics,
    compute_f_beta,
    compute_pairwise_metrics,
)
from business_entity_resolution.evaluation.validation import (
    stratified_grouped_s1_split,
)
from business_entity_resolution.matching.decision import (
    ThresholdOptimizer,
    predict_matches_for_entities,
)
from business_entity_resolution.matching.model import EntityMatchClassifier


def test_compute_f_beta():
    """Verify F-beta and specifically F0.5 precision weighting."""
    # When P=0 or R=0, F=0
    assert compute_f_beta(0.0, 1.0, beta=0.5) == 0.0
    assert compute_f_beta(1.0, 0.0, beta=0.5) == 0.0

    # When P=1 and R=1, F=1
    assert compute_f_beta(1.0, 1.0, beta=0.5) == 1.0

    # For P=0.8, R=0.4, beta=0.5:
    # 5 * 0.8 * 0.4 / (0.8 + 4 * 0.4) = 1.6 / 2.4 = 2/3 ≈ 0.6667
    f05 = compute_f_beta(0.8, 0.4, beta=0.5)
    assert abs(f05 - (2.0 / 3.0)) < 1e-5

    # Notice F0.5 rewards precision more than recall:
    f05_high_p = compute_f_beta(0.8, 0.4, beta=0.5)
    f05_high_r = compute_f_beta(0.4, 0.8, beta=0.5)
    assert f05_high_p > f05_high_r


def test_compute_pairwise_metrics():
    """Test pairwise confusion matrix and metrics calculation."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.2, 0.3])

    # Threshold 0.5:
    # predictions: [1, 1, 1, 0, 0]
    # TP: (idx 0, 1) = 2
    # FP: (idx 2) = 1
    # FN: (idx 4) = 1
    # TN: (idx 3) = 1
    metrics = compute_pairwise_metrics(y_true, y_scores, threshold=0.5, beta=0.5)
    assert metrics.tp == 2
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.tn == 1
    assert metrics.precision == round(2 / 3, 4)
    assert metrics.recall == round(2 / 3, 4)


def test_stratified_grouped_s1_split():
    """Verify grouped train/validation split maintains strictly disjoint S1 IDs."""
    queries = [
        ("S1-001", "Acme Inc", "123 Main St", "US"),
        ("S1-002", "Beta LLC", "456 Oak Rd", "US"),
        ("S1-003", "Gamma Corp", "789 Pine Ave", "IN"),
        ("S1-004", "Delta Ltd", "101 Elm Dr", "US"),
        ("S1-005", "Epsilon Co", "202 Cedar St", "IN"),
        ("S1-006", "Zeta Tech", "303 Birch Ln", "US"),
        ("S1-007", "Eta Systems", "404 Walnut St", "US"),
        ("S1-008", "Theta Global", "505 Maple Ave", "IN"),
        ("S1-009", "Iota Digital", "606 Spruce Rd", "US"),
        ("S1-010", "Kappa Media", "707 Ash Blvd", "IN"),
    ]

    # Cardinality distribution:
    # S1-001: 0 matches (zero)
    # S1-002: 1 match (single)
    # S1-003: 2 matches (multi)
    # ...
    gt_s2 = {
        "S1-001": set(),
        "S1-002": {"S2-002"},
        "S1-003": {"S2-003A", "S2-003B"},
        "S1-004": set(),
        "S1-005": {"S2-005"},
        "S1-006": {"S2-006A", "S2-006B"},
        "S1-007": set(),
        "S1-008": {"S2-008"},
        "S1-009": {"S2-009A", "S2-009B"},
        "S1-010": {"S2-010"},
    }
    gt_s3 = {q[0]: set() for q in queries}

    split = stratified_grouped_s1_split(
        s1_queries=queries,
        gt_s2_map=gt_s2,
        gt_s3_map=gt_s3,
        train_ratio=0.8,
        random_seed=42,
    )

    # Assert strictly disjoint: zero leakage
    assert len(split.train_s1_ids & split.val_s1_ids) == 0
    assert len(split.train_s1_ids) + len(split.val_s1_ids) == 10
    assert split.num_train == 7
    assert split.num_val == 3


def test_entity_level_metrics_and_subgroups():
    """Verify entity-level evaluation and zero/single/multi subgroup logic."""
    all_s1_ids = ["S1-ZERO", "S1-SINGLE", "S1-MULTI"]
    gt_map = {
        "S1-ZERO": set(),
        "S1-SINGLE": {"S2-TRUE1"},
        "S1-MULTI": {"S2-TRUE2A", "S2-TRUE2B"},
    }

    df_pairs = pd.DataFrame([
        # S1-ZERO: candidate predicted with low prob (correct rejection)
        {"source1_id": "S1-ZERO", "candidate_id": "S2-FALSE1", "proba": 0.2, "is_match": 0},
        # S1-SINGLE: candidate predicted with high prob (correct single match)
        {"source1_id": "S1-SINGLE", "candidate_id": "S2-TRUE1", "proba": 0.85, "is_match": 1},
        {"source1_id": "S1-SINGLE", "candidate_id": "S2-FALSE2", "proba": 0.15, "is_match": 0},
        # S1-MULTI: 1 true match predicted, 1 missed
        {"source1_id": "S1-MULTI", "candidate_id": "S2-TRUE2A", "proba": 0.9, "is_match": 1},
        {"source1_id": "S1-MULTI", "candidate_id": "S2-TRUE2B", "proba": 0.3, "is_match": 1},
    ])

    report = compute_entity_level_metrics(
        df_pairs=df_pairs,
        gt_map=gt_map,
        all_s1_ids=all_s1_ids,
        target_source="S2",
        threshold=0.5,
    )

    assert report.num_queries == 3
    # S1-ZERO had 0 predicted matches above 0.5 -> clean rejection!
    assert report.zero_match_subgroup.entity_accuracy == 1.0
    assert report.zero_match_subgroup.pred_pairs_count == 0

    # S1-SINGLE had exact match
    assert report.single_match_subgroup.entity_accuracy == 1.0
    assert report.single_match_subgroup.recall == 1.0

    # Overall macro F0.5 should be positive
    assert report.entity_macro_f05 > 0.0


def test_entity_match_classifier_fit_predict_save(tmp_path: Path):
    """Test LightGBM classifier training, prediction, importance, and serialization."""
    np.random.seed(42)
    n_samples = 120
    feature_names = [f"feat_{i}" for i in range(10)]
    X = pd.DataFrame(np.random.randn(n_samples, 10).astype(np.float32), columns=feature_names)
    # Target correlated with feat_0
    y = (X["feat_0"] > 0.3).astype(int).values

    clf = EntityMatchClassifier(
        n_estimators=30,
        learning_rate=0.1,
        num_leaves=15,
        feature_names=feature_names,
        random_state=42,
    )

    clf.fit(X, y)
    assert clf.is_fitted_

    probas = clf.predict_proba(X)
    assert len(probas) == n_samples
    assert np.all((probas >= 0.0) & (probas <= 1.0))

    preds = clf.predict(X, threshold=0.5)
    assert set(preds).issubset({0, 1})

    # Feature importances
    df_imp = clf.get_feature_importances(importance_type="gain")
    assert len(df_imp) == 10
    assert "normalized_pct" in df_imp.columns
    # feat_0 should have substantial importance
    assert df_imp.iloc[0]["feature"] == "feat_0"

    # Save and Load
    model_file = tmp_path / "test_model"
    clf.save(model_file)

    loaded_clf = EntityMatchClassifier.load(model_file)
    assert loaded_clf.is_fitted_
    loaded_probas = loaded_clf.predict_proba(X)
    np.testing.assert_allclose(probas, loaded_probas, atol=1e-5)


def test_threshold_optimizer():
    """Test threshold sweeping from 0.10 to 0.90."""
    all_s1_ids = ["S1-A", "S1-B"]
    gt_map = {"S1-A": {"S2-1"}, "S1-B": set()}

    df_pairs = pd.DataFrame([
        {"source1_id": "S1-A", "candidate_id": "S2-1", "proba": 0.85, "is_match": 1},
        {"source1_id": "S1-A", "candidate_id": "S2-2", "proba": 0.35, "is_match": 0},
        {"source1_id": "S1-B", "candidate_id": "S2-3", "proba": 0.25, "is_match": 0},
    ])

    optimizer = ThresholdOptimizer(thresholds=[0.10, 0.30, 0.50, 0.70, 0.90])
    df_sweep, best_th, best_rep = optimizer.sweep_thresholds(
        df_pairs=df_pairs,
        gt_map=gt_map,
        all_s1_ids=all_s1_ids,
        target_source="S2",
    )

    assert len(df_sweep) == 5
    assert best_th in [0.10, 0.30, 0.50, 0.70, 0.90]
    assert best_rep is not None
    assert best_rep.entity_macro_f05 >= 0.0

    # Match prediction extraction
    matches = predict_matches_for_entities(df_pairs, threshold=0.5)
    assert "S1-A" in matches
    assert len(matches["S1-A"]) == 1
    assert matches["S1-A"][0][0] == "S2-1"
    assert "S1-B" not in matches


def test_model_tuner_config_instantiation():
    """Verify ModelTuner and hyperparameter configurations are valid."""
    from business_entity_resolution.matching.tune import HYPERPARAM_CONFIGS, ModelTuner

    assert "Config_A_Baseline" in HYPERPARAM_CONFIGS
    assert "Config_D_PrecisionTuned" in HYPERPARAM_CONFIGS

    for name, params in HYPERPARAM_CONFIGS.items():
        assert "num_leaves" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert "n_estimators" in params
        assert "min_child_samples" in params
        assert "colsample_bytree" in params
        assert "subsample" in params

    tuner = ModelTuner()
    assert len(tuner.fine_thresholds) == 37
    assert tuner.fine_thresholds[0] == 0.05
    assert tuner.fine_thresholds[-1] == 0.95
