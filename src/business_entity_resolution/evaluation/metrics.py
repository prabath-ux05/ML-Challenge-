"""Comprehensive evaluation metrics for business entity resolution.

Focuses on Precision, Recall, F0.5, Macro F0.5, and match-cardinality subgroup metrics
(zero-match, single-match, multi-match).
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Calculate F-beta score from precision and recall.

    F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    For beta=0.5:
    F_0.5 = 1.25 * (P * R) / (0.25 * P + R) = 5 * P * R / (P + 4 * R)

    Args:
        precision: Precision value in [0.0, 1.0].
        recall: Recall value in [0.0, 1.0].
        beta: Weight of precision relative to recall (default 0.5).

    Returns:
        F_beta score in [0.0, 1.0].
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta_sq) * (precision * recall) / denom)


@dataclass
class PairwiseMetrics:
    """Pairwise classification metrics on candidate pairs."""

    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_pairs: int
    precision: float
    recall: float
    f1: float
    f05: float
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    beta: float = 0.5,
) -> PairwiseMetrics:
    """Compute pairwise classification metrics at a given decision threshold.

    Args:
        y_true: Binary ground-truth array (0 or 1).
        y_scores: Predicted probabilities or continuous scores.
        threshold: Decision threshold for positive prediction.
        beta: Beta value for F-beta score (default 0.5).

    Returns:
        PairwiseMetrics dataclass.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_pred_arr = (np.asarray(y_scores) >= threshold).astype(np.int32)

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    total = len(y_true_arr)

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    f05 = compute_f_beta(prec, rec, beta=beta)
    acc = float((tp + tn) / total) if total > 0 else 1.0

    return PairwiseMetrics(
        threshold=round(float(threshold), 4),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        total_pairs=total,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        f05=round(f05, 4),
        accuracy=round(acc, 4),
    )


@dataclass
class SubgroupEntityMetrics:
    """Entity-level metrics for a specific match cardinality subgroup."""

    subgroup_name: str
    num_entities: int
    true_pairs_count: int
    pred_pairs_count: int
    correct_entities_exact: int
    entity_accuracy: float
    precision: float
    recall: float
    f05: float
    macro_f05: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntityEvaluationReport:
    """Full entity-level evaluation report across all queries and subgroups."""

    target_source: str
    threshold: float
    num_queries: int
    total_true_pairs: int
    total_pred_pairs: int
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f05: float
    pairwise_f1: float
    # Entity-level macro averages
    entity_macro_precision: float
    entity_macro_recall: float
    entity_macro_f05: float
    # Subgroup breakdowns
    zero_match_subgroup: SubgroupEntityMetrics
    single_match_subgroup: SubgroupEntityMetrics
    multi_match_subgroup: SubgroupEntityMetrics

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


def compute_entity_level_metrics(
    df_pairs: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    all_s1_ids: Iterable[str],
    target_source: str = "S2",
    threshold: float = 0.5,
    beta: float = 0.5,
) -> EntityEvaluationReport:
    """Compute entity-level evaluation metrics including macro F0.5 and subgroup metrics.

    Args:
        df_pairs: DataFrame with 'source1_id', 'candidate_id', 'score' or 'proba', and optionally 'is_match'.
        gt_map: Mapping {s1_id: set_of_true_matching_candidate_ids}.
        all_s1_ids: All evaluated S1 query IDs (including those with 0 candidates retrieved).
        target_source: 'S2' or 'S3'.
        threshold: Decision probability threshold.
        beta: F-beta weight (default 0.5).

    Returns:
        EntityEvaluationReport.
    """
    score_col = "proba" if "proba" in df_pairs.columns else "score"
    if score_col not in df_pairs.columns:
        raise ValueError(f"df_pairs must contain 'score' or 'proba' column, found {df_pairs.columns.tolist()}")

    # Group predicted matches above threshold by source1_id
    pred_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in all_s1_ids}
    df_filtered = df_pairs[df_pairs[score_col] >= threshold]

    for s1_id, cand_id in zip(df_filtered["source1_id"], df_filtered["candidate_id"]):
        if s1_id in pred_matches:
            pred_matches[s1_id].add(cand_id)

    # Subgroup tracking lists
    zero_entities = []
    single_entities = []
    multi_entities = []

    all_entity_precisions = []
    all_entity_recalls = []
    all_entity_f05s = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    subgroup_stats: Dict[str, Dict[str, Any]] = {
        "zero": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "f05_list": [], "p_list": [], "r_list": []},
        "single": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "f05_list": [], "p_list": [], "r_list": []},
        "multi": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "f05_list": [], "p_list": [], "r_list": []},
    }

    for s1_id in all_s1_ids:
        true_set = gt_map.get(s1_id, set())
        pred_set = pred_matches.get(s1_id, set())

        # Determine subgroup
        n_true = len(true_set)
        if n_true == 0:
            sg = "zero"
            zero_entities.append(s1_id)
        elif n_true == 1:
            sg = "single"
            single_entities.append(s1_id)
        else:
            sg = "multi"
            multi_entities.append(s1_id)

        # Intersect
        tp_i = len(true_set & pred_set)
        fp_i = len(pred_set - true_set)
        fn_i = len(true_set - pred_set)

        total_tp += tp_i
        total_fp += fp_i
        total_fn += fn_i

        subgroup_stats[sg]["tp"] += tp_i
        subgroup_stats[sg]["fp"] += fp_i
        subgroup_stats[sg]["fn"] += fn_i

        # Exact match check
        if true_set == pred_set:
            subgroup_stats[sg]["exact"] += 1

        # Per-entity precision, recall, F0.5
        if n_true == 0 and len(pred_set) == 0:
            # Correct zero-prediction
            p_e = 1.0
            r_e = 1.0
            f05_e = 1.0
        elif n_true == 0 and len(pred_set) > 0:
            # False alarms on true zero
            p_e = 0.0
            r_e = 1.0  # Or 0.0; by convention precision is 0
            f05_e = 0.0
        elif n_true > 0 and len(pred_set) == 0:
            # Missed all true matches
            p_e = 0.0
            r_e = 0.0
            f05_e = 0.0
        else:
            p_e = tp_i / len(pred_set)
            r_e = tp_i / n_true
            f05_e = compute_f_beta(p_e, r_e, beta=beta)

        all_entity_precisions.append(p_e)
        all_entity_recalls.append(r_e)
        all_entity_f05s.append(f05_e)

        subgroup_stats[sg]["p_list"].append(p_e)
        subgroup_stats[sg]["r_list"].append(r_e)
        subgroup_stats[sg]["f05_list"].append(f05_e)

    # Compute overall pairwise metrics
    pairwise_prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else (1.0 if total_fn == 0 else 0.0)
    pairwise_rec = float(total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 1.0
    pairwise_f1 = float(2 * total_tp / (2 * total_tp + total_fp + total_fn)) if (2 * total_tp + total_fp + total_fn) > 0 else 0.0
    pairwise_f05 = compute_f_beta(pairwise_prec, pairwise_rec, beta=beta)

    # Compute subgroup metrics
    def build_sg_metrics(name: str, count: int, key: str) -> SubgroupEntityMetrics:
        st = subgroup_stats[key]
        tp_s, fp_s, fn_s = st["tp"], st["fp"], st["fn"]
        p_s = float(tp_s / (tp_s + fp_s)) if (tp_s + fp_s) > 0 else (1.0 if (key == "zero" and fp_s == 0) else 0.0)
        r_s = float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else (1.0 if key == "zero" else 0.0)
        f05_s = compute_f_beta(p_s, r_s, beta=beta) if key != "zero" else (1.0 if fp_s == 0 else 0.0)
        exact_acc = float(st["exact"] / count) if count > 0 else 1.0
        m_f05 = float(np.mean(st["f05_list"])) if st["f05_list"] else 0.0

        return SubgroupEntityMetrics(
            subgroup_name=name,
            num_entities=count,
            true_pairs_count=tp_s + fn_s,
            pred_pairs_count=tp_s + fp_s,
            correct_entities_exact=st["exact"],
            entity_accuracy=round(exact_acc, 4),
            precision=round(p_s, 4),
            recall=round(r_s, 4),
            f05=round(f05_s, 4),
            macro_f05=round(m_f05, 4),
        )

    sg_zero = build_sg_metrics("zero_match", len(zero_entities), "zero")
    sg_single = build_sg_metrics("single_match", len(single_entities), "single")
    sg_multi = build_sg_metrics("multi_match", len(multi_entities), "multi")

    num_queries = len(all_s1_ids)
    entity_macro_prec = float(np.mean(all_entity_precisions)) if all_entity_precisions else 0.0
    entity_macro_rec = float(np.mean(all_entity_recalls)) if all_entity_recalls else 0.0
    entity_macro_f05 = float(np.mean(all_entity_f05s)) if all_entity_f05s else 0.0

    return EntityEvaluationReport(
        target_source=target_source,
        threshold=round(float(threshold), 4),
        num_queries=num_queries,
        total_true_pairs=total_tp + total_fn,
        total_pred_pairs=total_tp + total_fp,
        pairwise_precision=round(pairwise_prec, 4),
        pairwise_recall=round(pairwise_rec, 4),
        pairwise_f05=round(pairwise_f05, 4),
        pairwise_f1=round(pairwise_f1, 4),
        entity_macro_precision=round(entity_macro_prec, 4),
        entity_macro_recall=round(entity_macro_rec, 4),
        entity_macro_f05=round(entity_macro_f05, 4),
        zero_match_subgroup=sg_zero,
        single_match_subgroup=sg_single,
        multi_match_subgroup=sg_multi,
    )
