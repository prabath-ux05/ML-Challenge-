"""String similarity, token overlap, fuzzy matching, and numeric/PIN code features for addresses."""

import re
from typing import Dict, List, Set
from rapidfuzz import distance, fuzz

from business_entity_resolution.normalization.addresses import (
    STATE_ABBREVIATIONS,
    STREET_ABBREVIATIONS,
    extract_address_tokens,
    normalize_address,
)

ADDRESS_FEATURE_NAMES: List[str] = [
    "addr_s1_missing",
    "addr_s2_missing",
    "addr_both_missing",
    "addr_either_missing",
    "addr_exact_match",
    "addr_raw_exact_match",
    "addr_len_s1",
    "addr_len_s2",
    "addr_len_diff",
    "addr_len_ratio",
    "addr_token_count_s1",
    "addr_token_count_s2",
    "addr_token_count_diff",
    "addr_token_overlap_count",
    "addr_token_jaccard",
    "addr_token_dice",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_levenshtein_sim",
    "addr_jaro_winkler",
    "addr_ngram_jaccard",
    "addr_fuzzy_token_overlap",
    "addr_fuzzy_token_jaccard",
    "addr_locality_overlap_count",
    "addr_locality_jaccard",
    "addr_confidence_weight",
    "addr_num_count_s1",
    "addr_num_count_s2",
    "addr_num_overlap_count",
    "addr_num_jaccard",
    "addr_num_exact_match",
    "addr_has_shared_pincode",
    "addr_has_shared_street_number",
]

GENERIC_STREET_WORDS = set(STREET_ABBREVIATIONS.keys()).union(set(STREET_ABBREVIATIONS.values()))
STATE_CODES = set(STATE_ABBREVIATIONS.values())


def _is_missing(addr: str) -> bool:
    """Check if address is empty or null placeholder."""
    if not addr:
        return True
    s = addr.strip().lower()
    return s in ("", "nan", "null", "none", "unknown", "n/a", "__")


def _extract_numbers(addr: str) -> Set[str]:
    """Extract all numeric token sequences from an address."""
    if _is_missing(addr):
        return set()
    return set(re.findall(r"\b\d+\b", addr))


def _extract_char_3grams(text: str) -> Set[str]:
    """Extract character 3-grams from cleaned text."""
    if not text:
        return set()
    s = f" {text.strip().lower()} "
    if len(s) < 3:
        return {s}
    return {s[i : i + 3] for i in range(len(s) - 2)}


def _extract_locality_tokens(tokens: Set[str]) -> Set[str]:
    """Extract non-numeric, non-generic landmark and locality tokens."""
    localities = set()
    for t in tokens:
        if (
            len(t) >= 3
            and not t.isdigit()
            and t not in GENERIC_STREET_WORDS
            and t not in STATE_CODES
            and t not in ("near", "opp", "opposite", "road", "street", "lane")
        ):
            localities.add(t)
    return localities


def extract_address_features(
    addr1: str,
    addr2: str,
) -> Dict[str, float]:
    """Extract string, token, fuzzy, locality, missingness, and numeric PIN features between two addresses.

    Args:
        addr1: Business address from Source 1.
        addr2: Business address from Candidate source (Source 2 or Source 3).

    Returns:
        Dictionary mapping feature name to numeric float value.
    """
    miss1 = 1.0 if _is_missing(addr1) else 0.0
    miss2 = 1.0 if _is_missing(addr2) else 0.0
    both_miss = 1.0 if (miss1 == 1.0 and miss2 == 1.0) else 0.0
    either_miss = 1.0 if (miss1 == 1.0 or miss2 == 1.0) else 0.0

    # Confidence weight for address evidence
    conf_weight = 1.0 if not either_miss else (0.5 if not both_miss else 0.0)

    a1_clean = (addr1 or "").strip()
    a2_clean = (addr2 or "").strip()

    # Raw exact match
    raw_exact = 1.0 if (not either_miss and a1_clean.lower() == a2_clean.lower()) else 0.0

    # Normalized addresses
    norm1 = normalize_address(a1_clean) if not miss1 else ""
    norm2 = normalize_address(a2_clean) if not miss2 else ""

    exact_match = 1.0 if (norm1 and norm1 == norm2) else 0.0

    # Length features
    len1 = float(len(norm1))
    len2 = float(len(norm2))
    len_diff = abs(len1 - len2)
    max_len = max(len1, len2, 1.0)
    min_len = min(len1, len2)
    len_ratio = min_len / max_len if not either_miss else 0.0

    # Token features
    toks1 = set(extract_address_tokens(a1_clean)) if not miss1 else set()
    toks2 = set(extract_address_tokens(a2_clean)) if not miss2 else set()
    cnt1 = float(len(toks1))
    cnt2 = float(len(toks2))
    cnt_diff = abs(cnt1 - cnt2)

    shared_toks = toks1.intersection(toks2)
    overlap_cnt = float(len(shared_toks))
    union_cnt = float(len(toks1.union(toks2)))
    jaccard = (overlap_cnt / union_cnt) if union_cnt > 0 else 0.0
    dice = (2.0 * overlap_cnt / (cnt1 + cnt2)) if (cnt1 + cnt2) > 0 else 0.0

    # Character and token similarities
    if norm1 and norm2:
        lev_sim = float(distance.Levenshtein.normalized_similarity(norm1, norm2))
        jw_sim = float(distance.JaroWinkler.similarity(norm1, norm2))
        tok_sort = float(fuzz.token_sort_ratio(norm1, norm2)) / 100.0
        tok_set = float(fuzz.token_set_ratio(norm1, norm2)) / 100.0

        # Character 3-gram Jaccard on address string
        ng1 = _extract_char_3grams(norm1)
        ng2 = _extract_char_3grams(norm2)
        ng_union = len(ng1.union(ng2))
        addr_ng_jaccard = float(len(ng1.intersection(ng2)) / ng_union) if ng_union > 0 else 0.0
    else:
        lev_sim = 0.0
        jw_sim = 0.0
        tok_sort = 0.0
        tok_set = 0.0
        addr_ng_jaccard = 0.0

    # Fuzzy address token matching (typo tolerance across tokens)
    fuzzy_overlap = 0.0
    if toks1 and toks2:
        toks2_list = list(toks2)
        matched_in_2 = set()
        for t1 in toks1:
            best_sim = 0.0
            best_idx = -1
            for idx, t2 in enumerate(toks2_list):
                if idx in matched_in_2:
                    continue
                sim = distance.Levenshtein.normalized_similarity(t1, t2)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx
            if best_sim >= 0.80:
                fuzzy_overlap += 1.0
                if best_idx >= 0:
                    matched_in_2.add(best_idx)
        fuzzy_jaccard = fuzzy_overlap / union_cnt if union_cnt > 0 else 0.0
    else:
        fuzzy_jaccard = 0.0

    # Locality / landmark token overlap
    loc1 = _extract_locality_tokens(toks1)
    loc2 = _extract_locality_tokens(toks2)
    loc_overlap = float(len(loc1.intersection(loc2)))
    loc_union = float(len(loc1.union(loc2)))
    loc_jaccard = loc_overlap / loc_union if loc_union > 0 else 0.0

    # Numeric and PIN code features
    nums1 = _extract_numbers(a1_clean)
    nums2 = _extract_numbers(a2_clean)
    num_cnt1 = float(len(nums1))
    num_cnt2 = float(len(nums2))

    shared_nums = nums1.intersection(nums2)
    num_overlap = float(len(shared_nums))
    num_union = float(len(nums1.union(nums2)))
    num_jaccard = (num_overlap / num_union) if num_union > 0 else 0.0
    num_exact = 1.0 if (nums1 and nums1 == nums2) else 0.0

    # Postal codes / PIN codes (5 digits in US, 6 digits in India)
    has_pincode = 1.0 if any(len(n) in (5, 6) for n in shared_nums) else 0.0
    # Street numbers (1-4 digits)
    has_street_num = 1.0 if any(len(n) <= 4 for n in shared_nums) else 0.0

    return {
        "addr_s1_missing": miss1,
        "addr_s2_missing": miss2,
        "addr_both_missing": both_miss,
        "addr_either_missing": either_miss,
        "addr_exact_match": exact_match,
        "addr_raw_exact_match": raw_exact,
        "addr_len_s1": len1,
        "addr_len_s2": len2,
        "addr_len_diff": len_diff,
        "addr_len_ratio": len_ratio,
        "addr_token_count_s1": cnt1,
        "addr_token_count_s2": cnt2,
        "addr_token_count_diff": cnt_diff,
        "addr_token_overlap_count": overlap_cnt,
        "addr_token_jaccard": jaccard,
        "addr_token_dice": dice,
        "addr_token_sort_ratio": tok_sort,
        "addr_token_set_ratio": tok_set,
        "addr_levenshtein_sim": lev_sim,
        "addr_jaro_winkler": jw_sim,
        "addr_ngram_jaccard": addr_ng_jaccard,
        "addr_fuzzy_token_overlap": fuzzy_overlap,
        "addr_fuzzy_token_jaccard": fuzzy_jaccard,
        "addr_locality_overlap_count": loc_overlap,
        "addr_locality_jaccard": loc_jaccard,
        "addr_confidence_weight": conf_weight,
        "addr_num_count_s1": num_cnt1,
        "addr_num_count_s2": num_cnt2,
        "addr_num_overlap_count": num_overlap,
        "addr_num_jaccard": num_jaccard,
        "addr_num_exact_match": num_exact,
        "addr_has_shared_pincode": has_pincode,
        "addr_has_shared_street_number": has_street_num,
    }
