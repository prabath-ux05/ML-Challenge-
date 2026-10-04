"""Grouped and stratified train/validation splitting by S1 entity ID."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.utils.logging import get_logger

logger = get_logger("evaluation.validation")


@dataclass
class GroupedSplitResult:
    """Result of a grouped train/validation split."""

    train_s1_ids: Set[str]
    val_s1_ids: Set[str]
    train_queries: List[Tuple[str, str, str, str]]
    val_queries: List[Tuple[str, str, str, str]]
    num_train: int
    num_val: int
    train_ratio: float
    # Subgroup counts
    train_zero_count: int
    train_single_count: int
    train_multi_count: int
    val_zero_count: int
    val_single_count: int
    val_multi_count: int


def stratified_grouped_s1_split(
    s1_queries: List[Tuple[str, str, str, str]],
    gt_s2_map: Dict[str, Set[str]],
    gt_s3_map: Dict[str, Set[str]],
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> GroupedSplitResult:
    """Split S1 queries into train and validation sets grouped by S1 entity_id.

    Stratified by total ground-truth match cardinality (zero, single, multi) to guarantee
    both splits have identical distributions and absolutely zero data leakage.

    Args:
        s1_queries: List of (s1_id, business_name, business_address, country).
        gt_s2_map: Mapping {s1_id: set_of_true_s2_match_ids}.
        gt_s3_map: Mapping {s1_id: set_of_true_s3_match_ids}.
        train_ratio: Fraction of entities assigned to training (default 0.8).
        random_seed: Seed for reproducibility.

    Returns:
        GroupedSplitResult with isolated train and validation query sets.
    """
    rng = np.random.RandomState(random_seed)

    query_lookup = {q[0]: q for q in s1_queries}

    zero_ids: List[str] = []
    single_ids: List[str] = []
    multi_ids: List[str] = []

    for s1_tuple in s1_queries:
        s1_id = s1_tuple[0]
        n_m = len(gt_s2_map.get(s1_id, set())) + len(gt_s3_map.get(s1_id, set()))
        if n_m == 0:
            zero_ids.append(s1_id)
        elif n_m == 1:
            single_ids.append(s1_id)
        else:
            multi_ids.append(s1_id)

    # Shuffle each stratum independently
    rng.shuffle(zero_ids)
    rng.shuffle(single_ids)
    rng.shuffle(multi_ids)

    def split_list(items: List[str], ratio: float) -> Tuple[List[str], List[str]]:
        split_idx = int(round(len(items) * ratio))
        return items[:split_idx], items[split_idx:]

    train_zero, val_zero = split_list(zero_ids, train_ratio)
    train_single, val_single = split_list(single_ids, train_ratio)
    train_multi, val_multi = split_list(multi_ids, train_ratio)

    train_ids = set(train_zero + train_single + train_multi)
    val_ids = set(val_zero + val_single + val_multi)

    # Integrity verification: strictly disjoint
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"CRITICAL: Found {len(overlap)} overlapping S1 IDs in train and validation sets!")

    train_queries = [query_lookup[sid] for sid in train_ids]
    val_queries = [query_lookup[sid] for sid in val_ids]

    logger.info(
        "Grouped split completed: %d Train (Z=%d, S=%d, M=%d) | %d Val (Z=%d, S=%d, M=%d)",
        len(train_ids),
        len(train_zero),
        len(train_single),
        len(train_multi),
        len(val_ids),
        len(val_zero),
        len(val_single),
        len(val_multi),
    )

    return GroupedSplitResult(
        train_s1_ids=train_ids,
        val_s1_ids=val_ids,
        train_queries=train_queries,
        val_queries=val_queries,
        num_train=len(train_ids),
        num_val=len(val_ids),
        train_ratio=train_ratio,
        train_zero_count=len(train_zero),
        train_single_count=len(train_single),
        train_multi_count=len(train_multi),
        val_zero_count=len(val_zero),
        val_single_count=len(val_single),
        val_multi_count=len(val_multi),
    )


def split_pairwise_dataframe_by_s1_groups(
    df: pd.DataFrame,
    train_s1_ids: Set[str],
    val_s1_ids: Set[str],
    s1_col: str = "source1_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split an existing pairwise candidate DataFrame by pre-computed S1 group IDs.

    Args:
        df: Pairwise DataFrame containing `s1_col`.
        train_s1_ids: Set of S1 entity IDs designated for training.
        val_s1_ids: Set of S1 entity IDs designated for validation.
        s1_col: Column name identifying the S1 entity ID.

    Returns:
        Tuple of (train_df, val_df).
    """
    train_mask = df[s1_col].isin(train_s1_ids)
    val_mask = df[s1_col].isin(val_s1_ids)

    train_df = df[train_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)

    return train_df, val_df
