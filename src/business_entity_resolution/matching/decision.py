"""Decision logic and threshold tuning for entity resolution matching."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.evaluation.metrics import (
    EntityEvaluationReport,
    compute_entity_level_metrics,
    compute_pairwise_metrics,
)
from business_entity_resolution.utils.logging import get_logger

logger = get_logger("matching.decision")


@dataclass
class ThresholdSweepRow:
    """Summary row for a single threshold evaluation."""

    threshold: float
    target_source: str
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f05: float
    pairwise_f1: float
    entity_macro_precision: float
    entity_macro_recall: float
    entity_macro_f05: float
    # Subgroup F0.5
    zero_match_accuracy: float
    single_match_f05: float
    multi_match_f05: float
    pred_pairs_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ThresholdOptimizer:
    """Sweeps decision thresholds to optimize macro F0.5 across entity resolution predictions."""

    def __init__(
        self,
        thresholds: Optional[List[float]] = None,
        beta: float = 0.5,
    ):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            # Default grid: 0.10 to 0.90 in steps of 0.05
            self.thresholds = [round(t, 2) for t in np.linspace(0.10, 0.90, 17)]
        self.beta = beta

    def sweep_thresholds(
        self,
        df_pairs: pd.DataFrame,
        gt_map: Dict[str, Set[str]],
        all_s1_ids: Iterable[str],
        target_source: str = "S2",
    ) -> Tuple[pd.DataFrame, float, EntityEvaluationReport]:
        """Evaluate full grid of thresholds and identify optimal threshold for macro F0.5.

        Args:
            df_pairs: DataFrame with candidate predictions ('source1_id', 'candidate_id', 'proba').
            gt_map: True ground-truth match mapping {s1_id: set_of_cand_ids}.
            all_s1_ids: List/set of all evaluated S1 IDs.
            target_source: 'S2' or 'S3'.

        Returns:
            Tuple of:
            - DataFrame of sweep results across all thresholds.
            - Best threshold value maximizing entity_macro_f05.
            - Best EntityEvaluationReport at that optimal threshold.
        """
        all_s1_list = list(all_s1_ids)
        rows: List[Dict[str, Any]] = []
        reports: Dict[float, EntityEvaluationReport] = {}

        best_thresh = self.thresholds[0]
        best_macro_f05 = -1.0
        best_report: Optional[EntityEvaluationReport] = None

        for thresh in self.thresholds:
            report = compute_entity_level_metrics(
                df_pairs=df_pairs,
                gt_map=gt_map,
                all_s1_ids=all_s1_list,
                target_source=target_source,
                threshold=thresh,
                beta=self.beta,
            )
            reports[thresh] = report

            row = ThresholdSweepRow(
                threshold=thresh,
                target_source=target_source,
                pairwise_precision=report.pairwise_precision,
                pairwise_recall=report.pairwise_recall,
                pairwise_f05=report.pairwise_f05,
                pairwise_f1=report.pairwise_f1,
                entity_macro_precision=report.entity_macro_precision,
                entity_macro_recall=report.entity_macro_recall,
                entity_macro_f05=report.entity_macro_f05,
                zero_match_accuracy=report.zero_match_subgroup.entity_accuracy,
                single_match_f05=report.single_match_subgroup.f05,
                multi_match_f05=report.multi_match_subgroup.f05,
                pred_pairs_count=report.total_pred_pairs,
            )
            rows.append(row.to_dict())

            # Optimize for macro F0.5
            if report.entity_macro_f05 > best_macro_f05:
                best_macro_f05 = report.entity_macro_f05
                best_thresh = thresh
                best_report = report

        df_sweep = pd.DataFrame(rows)
        logger.info(
            "Threshold sweep (%s) optimal threshold: %.2f (Macro F0.5 = %.4f, Pairwise F0.5 = %.4f)",
            target_source,
            best_thresh,
            best_macro_f05,
            best_report.pairwise_f05 if best_report else 0.0,
        )
        return df_sweep, best_thresh, best_report


def predict_matches_for_entities(
    df_pairs: pd.DataFrame,
    threshold: float = 0.5,
    max_matches_per_entity: Optional[int] = None,
    score_col: str = "proba",
) -> Dict[str, List[Tuple[str, float]]]:
    """Extract filtered and ranked matches for each S1 entity.

    Args:
        df_pairs: DataFrame with 'source1_id', 'candidate_id', and probability/score.
        threshold: Minimum probability threshold.
        max_matches_per_entity: Optional ceiling on matches per entity.
        score_col: Column name containing probability or score.

    Returns:
        Mapping {s1_id: [(candidate_id, probability), ...]}.
    """
    filtered = df_pairs[df_pairs[score_col] >= threshold]
    matches: Dict[str, List[Tuple[str, float]]] = {}

    for row in filtered.itertuples():
        s1_id = getattr(row, "source1_id")
        cand_id = getattr(row, "candidate_id")
        score = float(getattr(row, score_col))
        if s1_id not in matches:
            matches[s1_id] = []
        matches[s1_id].append((cand_id, score))

    # Sort each entity's candidates by score descending
    for s1_id in matches:
        matches[s1_id].sort(key=lambda x: x[1], reverse=True)
        if max_matches_per_entity is not None:
            matches[s1_id] = matches[s1_id][:max_matches_per_entity]

    return matches
