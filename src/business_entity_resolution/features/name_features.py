"""String similarity, edit distance, token, and n-gram features for business names."""

from collections import Counter
import math
from typing import Dict, List, Optional
from rapidfuzz import distance, fuzz

from business_entity_resolution.blocking.candidate_generator import extract_char_ngrams
from business_entity_resolution.normalization.names import (
    extract_name_tokens,
    normalize_business_name,
)

NAME_FEATURE_NAMES: List[str] = [
    "name_exact_match",
    "name_raw_exact_match",
    "name_len_s1",
    "name_len_s2",
    "name_len_diff",
    "name_len_ratio",
    "name_token_count_s1",
    "name_token_count_s2",
    "name_token_count_diff",
    "name_token_overlap_count",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_containment_min",
    "name_token_containment_max",
    "name_levenshtein_sim",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_raw_levenshtein_sim",
    "name_raw_jaro_winkler",
    "name_raw_token_sort_ratio",
    "name_ngram_jaccard",
    "name_ngram_overlap_count",
    "name_ngram_cosine",
    "name_tfidf_sim",
]


def extract_name_features(
    name1: str,
    name2: str,
    ngram_idf: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Extract string, token, edit distance, and n-gram similarity features between two business names.

    Args:
        name1: Business name from Source 1.
        name2: Business name from Candidate source (Source 2 or Source 3).
        ngram_idf: Optional mapping of character 3-gram to inverse document frequency.

    Returns:
        Dictionary mapping feature name to numeric float value.
    """
    n1_clean = (name1 or "").strip()
    n2_clean = (name2 or "").strip()

    # Raw string exact match
    raw_exact = 1.0 if (n1_clean and n1_clean.lower() == n2_clean.lower()) else 0.0

    # Normalized names
    norm1 = normalize_business_name(n1_clean)
    norm2 = normalize_business_name(n2_clean)

    exact_match = 1.0 if (norm1 and norm1 == norm2) else 0.0

    # Length features
    len1 = float(len(norm1))
    len2 = float(len(norm2))
    len_diff = abs(len1 - len2)
    max_len = max(len1, len2, 1.0)
    min_len = min(len1, len2)
    len_ratio = min_len / max_len

    # Token features
    toks1 = set(extract_name_tokens(n1_clean))
    toks2 = set(extract_name_tokens(n2_clean))
    cnt1 = float(len(toks1))
    cnt2 = float(len(toks2))
    cnt_diff = abs(cnt1 - cnt2)

    shared_toks = toks1.intersection(toks2)
    overlap_cnt = float(len(shared_toks))
    union_cnt = float(len(toks1.union(toks2)))
    jaccard = (overlap_cnt / union_cnt) if union_cnt > 0 else 0.0
    dice = (2.0 * overlap_cnt / (cnt1 + cnt2)) if (cnt1 + cnt2) > 0 else 0.0

    c1 = (overlap_cnt / cnt1) if cnt1 > 0 else 0.0
    c2 = (overlap_cnt / cnt2) if cnt2 > 0 else 0.0
    containment_min = min(c1, c2)
    containment_max = max(c1, c2)

    # Character edit similarities
    if norm1 and norm2:
        lev_sim = float(distance.Levenshtein.normalized_similarity(norm1, norm2))
        jw_sim = float(distance.JaroWinkler.similarity(norm1, norm2))
        tok_sort = float(fuzz.token_sort_ratio(norm1, norm2)) / 100.0
        tok_set = float(fuzz.token_set_ratio(norm1, norm2)) / 100.0
    else:
        lev_sim = 0.0
        jw_sim = 0.0
        tok_sort = 0.0
        tok_set = 0.0

    # Raw string edit similarities (robust against suffix stripping divergence)
    if n1_clean and n2_clean:
        raw_lev_sim = float(distance.Levenshtein.normalized_similarity(n1_clean.lower(), n2_clean.lower()))
        raw_jw_sim = float(distance.JaroWinkler.similarity(n1_clean.lower(), n2_clean.lower()))
        raw_tok_sort = float(fuzz.token_sort_ratio(n1_clean.lower(), n2_clean.lower())) / 100.0
    else:
        raw_lev_sim = 0.0
        raw_jw_sim = 0.0
        raw_tok_sort = 0.0

    # N-gram similarity (character 3-grams)
    ng1 = extract_char_ngrams(n1_clean, n=3)
    ng2 = extract_char_ngrams(n2_clean, n=3)
    set_ng1 = set(ng1)
    set_ng2 = set(ng2)
    shared_ng = set_ng1.intersection(set_ng2)
    union_ng = set_ng1.union(set_ng2)
    ng_jaccard = (len(shared_ng) / len(union_ng)) if union_ng else 0.0
    ng_overlap = float(len(shared_ng))

    # Frequency-based cosine similarity on n-grams
    if ng1 and ng2:
        c_ng1 = Counter(ng1)
        c_ng2 = Counter(ng2)
        dot_product = sum(c_ng1[g] * c_ng2[g] for g in shared_ng)
        norm_a = math.sqrt(sum(v * v for v in c_ng1.values()))
        norm_b = math.sqrt(sum(v * v for v in c_ng2.values()))
        ng_cosine = (dot_product / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0

        # IDF-weighted cosine similarity
        if ngram_idf:
            dot_idf = sum(c_ng1[g] * c_ng2[g] * (ngram_idf.get(g, 1.0) ** 2) for g in shared_ng)
            norm_idf_a = math.sqrt(sum((v * ngram_idf.get(g, 1.0)) ** 2 for g, v in c_ng1.items()))
            norm_idf_b = math.sqrt(sum((v * ngram_idf.get(g, 1.0)) ** 2 for g, v in c_ng2.items()))
            tfidf_sim = (dot_idf / (norm_idf_a * norm_idf_b)) if (norm_idf_a > 0 and norm_idf_b > 0) else 0.0
        else:
            tfidf_sim = ng_cosine
    else:
        ng_cosine = 0.0
        tfidf_sim = 0.0

    return {
        "name_exact_match": exact_match,
        "name_raw_exact_match": raw_exact,
        "name_len_s1": len1,
        "name_len_s2": len2,
        "name_len_diff": len_diff,
        "name_len_ratio": len_ratio,
        "name_token_count_s1": cnt1,
        "name_token_count_s2": cnt2,
        "name_token_count_diff": cnt_diff,
        "name_token_overlap_count": overlap_cnt,
        "name_token_jaccard": jaccard,
        "name_token_dice": dice,
        "name_token_containment_min": containment_min,
        "name_token_containment_max": containment_max,
        "name_levenshtein_sim": lev_sim,
        "name_jaro_winkler": jw_sim,
        "name_token_sort_ratio": tok_sort,
        "name_token_set_ratio": tok_set,
        "name_raw_levenshtein_sim": raw_lev_sim,
        "name_raw_jaro_winkler": raw_jw_sim,
        "name_raw_token_sort_ratio": raw_tok_sort,
        "name_ngram_jaccard": ng_jaccard,
        "name_ngram_overlap_count": ng_overlap,
        "name_ngram_cosine": ng_cosine,
        "name_tfidf_sim": tfidf_sim,
    }
