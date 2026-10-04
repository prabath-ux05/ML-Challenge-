"""Comprehensive pairwise feature extraction for business entity resolution."""

import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from business_entity_resolution.features.address_features import (
    ADDRESS_FEATURE_NAMES,
    extract_address_features,
)
from business_entity_resolution.features.country_features import (
    COUNTRY_FEATURE_NAMES,
    extract_country_features,
)
from business_entity_resolution.features.name_features import (
    NAME_FEATURE_NAMES,
    extract_name_features,
)

METADATA_FEATURE_NAMES: List[str] = [
    "source_pair_is_s2",
    "source_pair_is_s3",
    "block_exact_name",
    "block_name_token",
    "block_address_token",
    "block_tfidf_ngram",
    "block_evidence_channels_count",
    "cand_rank",
    "cand_rank_reciprocal",
    "cand_blocking_score",
    "cand_score_ratio_to_top1",
    "total_candidates_for_s1",
    "log_total_candidates",
]

INTERACTION_FEATURE_NAMES: List[str] = [
    "name_jaccard_x_addr_missing",
    "name_jaro_x_addr_missing",
    "name_exact_x_addr_missing",
    "name_high_conf_when_addr_missing",
    "addr_valid_match_score",
]

ALL_FEATURE_NAMES: List[str] = (
    NAME_FEATURE_NAMES
    + ADDRESS_FEATURE_NAMES
    + COUNTRY_FEATURE_NAMES
    + METADATA_FEATURE_NAMES
    + INTERACTION_FEATURE_NAMES
)


def extract_pair_features(
    s1_name: str,
    s1_address: str,
    s1_country: str,
    cand_name: str,
    cand_address: str,
    cand_country: str,
    target_source: str = "S2",
    blocking_meta: Optional[Dict[str, Any]] = None,
    ngram_idf: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract complete pairwise feature dictionary between Source 1 entity and a candidate entity.

    All features are strictly numeric floats suitable for LightGBM/XGBoost.

    Args:
        s1_name: S1 entity business name.
        s1_address: S1 entity business address.
        s1_country: S1 entity country.
        cand_name: Candidate entity business name.
        cand_address: Candidate entity business address.
        cand_country: Candidate entity country.
        target_source: 'S2' or 'S3'.
        blocking_meta: Optional metadata dict from CandidateGenerator.
        ngram_idf: Optional mapping of character 3-grams to IDF.

    Returns:
        Dictionary mapping all feature names in ALL_FEATURE_NAMES to numeric floats.
    """
    # 1. Name features
    name_feats = extract_name_features(s1_name, cand_name, ngram_idf=ngram_idf)

    # 2. Address features
    addr_feats = extract_address_features(s1_address, cand_address)

    # 3. Country features
    country_feats = extract_country_features(s1_country, cand_country)

    # 4. Source pair indicators
    is_s2 = 1.0 if target_source.upper() == "S2" else 0.0
    is_s3 = 1.0 if target_source.upper() == "S3" else 0.0

    # 5. Blocking metadata features (with neutral defaults if not provided)
    meta = blocking_meta or {}
    b_exact = float(meta.get("block_exact_name", 0.0))
    b_ntok = float(meta.get("block_name_token", 0.0))
    b_atok = float(meta.get("block_address_token", 0.0))
    b_tfidf = float(meta.get("block_tfidf_ngram", 0.0))
    b_count = float(meta.get("block_evidence_channels_count", b_exact + b_ntok + b_atok + b_tfidf))

    rank = float(meta.get("cand_rank", 1.0))
    rank_recip = float(meta.get("cand_rank_reciprocal", 1.0 / rank if rank > 0 else 1.0))
    score = float(meta.get("cand_blocking_score", 0.0))
    score_ratio = float(meta.get("cand_score_ratio_to_top1", 1.0 if score > 0 else 0.0))
    tot_cands = float(meta.get("total_candidates_for_s1", 1.0))
    log_tot_cands = float(meta.get("log_total_candidates", math.log(1.0 + tot_cands)))

    meta_feats = {
        "source_pair_is_s2": is_s2,
        "source_pair_is_s3": is_s3,
        "block_exact_name": b_exact,
        "block_name_token": b_ntok,
        "block_address_token": b_atok,
        "block_tfidf_ngram": b_tfidf,
        "block_evidence_channels_count": b_count,
        "cand_rank": rank,
        "cand_rank_reciprocal": rank_recip,
        "cand_blocking_score": score,
        "cand_score_ratio_to_top1": score_ratio,
        "total_candidates_for_s1": tot_cands,
        "log_total_candidates": log_tot_cands,
    }

    # 6. Address missingness-aware interaction features
    addr_either_miss = addr_feats["addr_either_missing"]
    name_jaccard = name_feats["name_token_jaccard"]
    name_jaro = name_feats["name_raw_jaro_winkler"]
    name_exact = name_feats["name_exact_match"]

    name_jaccard_x_miss = name_jaccard * addr_either_miss
    name_jaro_x_miss = name_jaro * addr_either_miss
    name_exact_x_miss = name_exact * addr_either_miss
    contain_min = name_feats.get("name_token_containment_min", 0.0)
    name_high_conf = 1.0 if (addr_either_miss > 0.5 and (name_jaro >= 0.85 or name_exact > 0.5 or contain_min >= 0.99)) else 0.0
    addr_valid_score = (1.0 - addr_either_miss) * addr_feats["addr_token_jaccard"]

    interaction_feats = {
        "name_jaccard_x_addr_missing": name_jaccard_x_miss,
        "name_jaro_x_addr_missing": name_jaro_x_miss,
        "name_exact_x_addr_missing": name_exact_x_miss,
        "name_high_conf_when_addr_missing": name_high_conf,
        "addr_valid_match_score": addr_valid_score,
    }

    # Combined dictionary
    feats: Dict[str, float] = {}
    feats.update(name_feats)
    feats.update(addr_feats)
    feats.update(country_feats)
    feats.update(meta_feats)
    feats.update(interaction_feats)

    return feats


class PairFeatureExtractor:
    """State-of-the-art pairwise feature extraction engine for business entity resolution."""

    def __init__(self, ngram_idf: Optional[Dict[str, float]] = None):
        """Initialize the extractor with optional global character 3-gram IDF table."""
        self.ngram_idf = ngram_idf or {}

    @property
    def feature_names(self) -> List[str]:
        """List of all extracted numeric feature column names."""
        return list(ALL_FEATURE_NAMES)

    def extract_pair(
        self,
        s1_tuple: Tuple[str, str, str, str],
        cand_tuple: Tuple[str, str, str, str],
        target_source: str = "S2",
        blocking_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Extract feature dictionary for a single entity pair.

        Args:
            s1_tuple: (s1_id, name, address, country).
            cand_tuple: (cand_id, name, address, country).
            target_source: 'S2' or 'S3'.
            blocking_meta: Optional blocking metadata.

        Returns:
            Dictionary of numeric features.
        """
        _, s1_name, s1_addr, s1_country = s1_tuple
        _, cand_name, cand_addr, cand_country = cand_tuple

        return extract_pair_features(
            s1_name=s1_name,
            s1_address=s1_addr,
            s1_country=s1_country,
            cand_name=cand_name,
            cand_address=cand_addr,
            cand_country=cand_country,
            target_source=target_source,
            blocking_meta=blocking_meta,
            ngram_idf=self.ngram_idf,
        )

    def extract_batch_dataframe(
        self,
        pairs: Iterable[Tuple[Tuple[str, str, str, str], Tuple[str, str, str, str], str, Optional[Dict[str, Any]]]],
        include_identifiers: bool = True,
        labels: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Extract a batch of candidate pairs into a pandas DataFrame.

        Args:
            pairs: Iterable of (s1_tuple, cand_tuple, target_source, blocking_meta).
            include_identifiers: Whether to prepend 'source1_id', 'candidate_id', 'target_source'.
            labels: Optional ground-truth match labels (0 or 1) to append as 'is_match'.

        Returns:
            DataFrame containing extracted numeric features.
        """
        rows = []
        id_rows = []

        for item in pairs:
            s1_tuple, cand_tuple, target_src, b_meta = item
            feats = self.extract_pair(
                s1_tuple=s1_tuple,
                cand_tuple=cand_tuple,
                target_source=target_src,
                blocking_meta=b_meta,
            )
            rows.append(feats)
            if include_identifiers:
                id_rows.append({
                    "source1_id": s1_tuple[0],
                    "candidate_id": cand_tuple[0],
                    "target_source": target_src,
                })

        df_feats = pd.DataFrame(rows, columns=ALL_FEATURE_NAMES).astype(np.float32)

        if include_identifiers and id_rows:
            df_ids = pd.DataFrame(id_rows)
            df = pd.concat([df_ids, df_feats], axis=1)
        else:
            df = df_feats

        if labels is not None:
            df["is_match"] = np.array(labels, dtype=np.int32)

        return df

    def build_candidate_pair_dataset(
        self,
        s1_queries: List[Tuple[str, str, str, str]],
        candidate_records: List[Tuple[str, str, str, str]],
        candidate_generator: Any,
        target_source: str,
        gt_map: Optional[Dict[str, Set[str]]] = None,
        k: int = 100,
    ) -> pd.DataFrame:
        """End-to-end dataset builder: blocks candidates and extracts pairwise features.

        Guarantees zero data leakage: ground-truth matching labels are strictly isolated in
        the 'is_match' target column and never touch feature calculations.

        Args:
            s1_queries: List of (s1_id, name, address, country).
            candidate_records: List of candidate records indexed by candidate_generator.
            candidate_generator: Fitted CandidateGenerator instance.
            target_source: 'S2' or 'S3'.
            gt_map: Optional ground-truth map {s1_id: {true_matched_cand_ids}}.
            k: Candidate budget per S1 query.

        Returns:
            DataFrame with identifiers, all numeric features, and 'is_match' label (if gt_map provided).
        """
        # Build candidate lookup dictionary for fast entity fetching
        cand_dict: Dict[str, Tuple[str, str, str, str]] = {
            r[0]: r for r in candidate_records
        }

        pair_items = []
        labels = []

        for s1_tuple in s1_queries:
            s1_id, s1_name, s1_addr, s1_country = s1_tuple
            # Retrieve candidates with rich metadata
            meta_candidates = candidate_generator.generate_candidates_with_metadata(
                name=s1_name,
                address=s1_addr,
                country=s1_country,
                k=k,
            )

            true_matches = gt_map.get(s1_id, set()) if gt_map else set()

            for cand_meta in meta_candidates:
                cand_id = cand_meta["entity_id"]
                if cand_id not in cand_dict:
                    continue
                cand_tuple = cand_dict[cand_id]

                pair_items.append((s1_tuple, cand_tuple, target_source, cand_meta))

                if gt_map is not None:
                    is_match = 1 if cand_id in true_matches else 0
                    labels.append(is_match)

        return self.extract_batch_dataframe(
            pairs=pair_items,
            include_identifiers=True,
            labels=labels if gt_map is not None else None,
        )
