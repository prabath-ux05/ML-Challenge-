"""Candidate generator supporting 6 blocking strategies for entity resolution."""

from collections import defaultdict
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from business_entity_resolution.normalization.addresses import extract_address_tokens, normalize_address
from business_entity_resolution.normalization.names import LEGAL_TERMS, extract_name_tokens, normalize_business_name
from business_entity_resolution.normalization.common import clean_text, transliterate_text


# Strategy name constants
STRATEGY_NORMALIZED_EXACT = "normalized_name_exact"
STRATEGY_TOKEN_NAME = "token_name"
STRATEGY_COUNTRY_NAME = "country_name"
STRATEGY_ADDRESS_TOKEN = "address_token"
STRATEGY_TFIDF_NGRAM = "tfidf_ngram"
STRATEGY_COMBINED_HYBRID = "combined_hybrid"

ALL_STRATEGIES = [
    STRATEGY_NORMALIZED_EXACT,
    STRATEGY_TOKEN_NAME,
    STRATEGY_COUNTRY_NAME,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_TFIDF_NGRAM,
    STRATEGY_COMBINED_HYBRID,
]


def extract_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Extract character n-grams from cleaned text with word boundaries."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    words = cleaned.split()
    ngrams = []
    for w in words:
        padded = f" {w} "
        if len(padded) < n:
            ngrams.append(padded)
        else:
            for i in range(len(padded) - n + 1):
                ngrams.append(padded[i : i + n])
    return ngrams


class CandidateGenerator:
    """Multi-strategy candidate generation and indexing engine."""

    def __init__(
        self,
        strategy: str = STRATEGY_COMBINED_HYBRID,
        max_candidates: int = 200,
        max_token_freq: int = 30_000,
    ):
        """Initialize the candidate generator.

        Args:
            strategy: One of ALL_STRATEGIES.
            max_candidates: Maximum candidates returned per reference entity.
            max_token_freq: Inverted index pruning threshold for frequent tokens.
        """
        if strategy not in ALL_STRATEGIES:
            raise ValueError(f"Unknown strategy '{strategy}'. Expected one of {ALL_STRATEGIES}")

        self.strategy = strategy
        self.max_candidates = max_candidates
        self.max_token_freq = max_token_freq

        # Indices
        self.exact_index: Dict[str, List[str]] = defaultdict(list)
        self.country_exact_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.name_token_index: Dict[str, List[str]] = defaultdict(list)
        self.country_token_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.address_token_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.raw_address_token_index: Dict[str, List[str]] = defaultdict(list)
        self.ngram_index: Dict[str, List[str]] = defaultdict(list)
        self.ngram_idf: Dict[str, float] = {}
        self.country_ngram_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.country_ngram_idf: Dict[Tuple[str, str], float] = {}
        self.acronym_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.short_name_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.compound_name_index: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        self.num_indexed_entities = 0

    def fit(self, records: Iterable[Tuple[str, str, str, str]]):
        """Build index from an iterable of (entity_id, business_name, business_address, country).

        Args:
            records: Iterable yielding (id, name, address, country).
        """
        self.clear()
        count = 0

        ngram_df: Dict[str, int] = defaultdict(int)
        country_ngram_df: Dict[Tuple[str, str], int] = defaultdict(int)

        for entity_id, name, address, country in records:
            count += 1
            norm_name = normalize_business_name(name, transliterate=True)
            raw_norm_name = normalize_business_name(name, transliterate=False)
            name_tokens = set(extract_name_tokens(name))
            addr_tokens = set(extract_address_tokens(address))
            clean_country = country.strip() if country else ""

            # 1. Exact Name
            if self.strategy in (STRATEGY_NORMALIZED_EXACT, STRATEGY_COMBINED_HYBRID):
                if norm_name:
                    self.exact_index[norm_name].append(entity_id)
                    self.country_exact_index[(clean_country, norm_name)].append(entity_id)
                if raw_norm_name and raw_norm_name != norm_name:
                    self.exact_index[raw_norm_name].append(entity_id)
                    self.country_exact_index[(clean_country, raw_norm_name)].append(entity_id)

            # 2. Token Name
            if self.strategy in (STRATEGY_TOKEN_NAME, STRATEGY_COMBINED_HYBRID):
                for t in name_tokens:
                    self.name_token_index[t].append(entity_id)

            # 3. Country + Name
            if self.strategy in (STRATEGY_COUNTRY_NAME, STRATEGY_COMBINED_HYBRID):
                for t in name_tokens:
                    self.country_token_index[(clean_country, t)].append(entity_id)

            # 4. Address Token
            if self.strategy in (STRATEGY_ADDRESS_TOKEN, STRATEGY_COMBINED_HYBRID):
                for at in addr_tokens:
                    self.address_token_index[(clean_country, at)].append(entity_id)
                    self.raw_address_token_index[at].append(entity_id)

            # 5. TF-IDF N-gram
            if self.strategy in (STRATEGY_TFIDF_NGRAM, STRATEGY_COMBINED_HYBRID):
                ngrams = set(extract_char_ngrams(name, n=3))
                trans_name = transliterate_text(name)
                if trans_name != name:
                    ngrams.update(extract_char_ngrams(trans_name, n=3))
                for ng in ngrams:
                    self.ngram_index[ng].append(entity_id)
                    self.country_ngram_index[(clean_country, ng)].append(entity_id)
                    ngram_df[ng] += 1
                    country_ngram_df[(clean_country, ng)] += 1

            # 6. Acronym / Initial Indexing
            if self.strategy in (STRATEGY_TOKEN_NAME, STRATEGY_COMBINED_HYBRID):
                inits = self._get_initials(name)
                if len(inits) >= 2:
                    self.acronym_index[(clean_country, inits)].append(entity_id)
                    self.acronym_index[("", inits)].append(entity_id)
                clean_short = re.sub(r"[^\w]", "", name).upper()
                if 2 <= len(clean_short) <= 5:
                    self.short_name_index[(clean_country, clean_short)].append(entity_id)
                    self.short_name_index[("", clean_short)].append(entity_id)
                clean_compound = re.sub(r"[^\w]", "", name or "").lower()
                if 4 <= len(clean_compound) <= 35:
                    self.compound_name_index[(clean_country, clean_compound)].append(entity_id)
                    self.compound_name_index[("", clean_compound)].append(entity_id)

        self.num_indexed_entities = count

        # Prune high-frequency tokens to maintain performance and selectivity
        self._prune_frequent_tokens(ngram_df, country_ngram_df)

    def _prune_frequent_tokens(
        self,
        ngram_df: Dict[str, int],
        country_ngram_df: Optional[Dict[Tuple[str, str], int]] = None,
    ):
        """Prune inverted lists exceeding frequency threshold and compute IDFs."""
        if self.name_token_index:
            self.name_token_index = defaultdict(
                list,
                {k: v for k, v in self.name_token_index.items() if len(v) <= self.max_token_freq},
            )

        if self.country_token_index:
            self.country_token_index = defaultdict(
                list,
                {k: v for k, v in self.country_token_index.items() if len(v) <= self.max_token_freq},
            )

        if self.address_token_index:
            self.address_token_index = defaultdict(
                list,
                {k: v for k, v in self.address_token_index.items() if len(v) <= self.max_token_freq},
            )

        if self.raw_address_token_index:
            self.raw_address_token_index = defaultdict(
                list,
                {k: v for k, v in self.raw_address_token_index.items() if len(v) <= self.max_token_freq},
            )

        N = max(self.num_indexed_entities, 1)

        if self.ngram_index:
            pruned_ngram = {}
            for ng, postings in self.ngram_index.items():
                df = ngram_df.get(ng, len(postings))
                if df <= self.max_token_freq:
                    pruned_ngram[ng] = postings
                    self.ngram_idf[ng] = math.log((1 + N) / (1 + df)) + 1.0
            self.ngram_index = defaultdict(list, pruned_ngram)

        if self.country_ngram_index:
            c_df = country_ngram_df or {}
            pruned_c_ngram = {}
            for k, postings in self.country_ngram_index.items():
                df = c_df.get(k, len(postings))
                if df <= self.max_token_freq:
                    pruned_c_ngram[k] = postings
                    self.country_ngram_idf[k] = math.log((1 + N) / (1 + df)) + 1.0
            self.country_ngram_index = defaultdict(list, pruned_c_ngram)

    def clear(self):
        """Clear all indexed data structures."""
        self.exact_index.clear()
        self.country_exact_index.clear()
        self.name_token_index.clear()
        self.country_token_index.clear()
        self.address_token_index.clear()
        self.raw_address_token_index.clear()
        self.ngram_index.clear()
        self.ngram_idf.clear()
        self.country_ngram_index.clear()
        self.country_ngram_idf.clear()
        self.acronym_index.clear()
        self.short_name_index.clear()
        self.compound_name_index.clear()
        self.num_indexed_entities = 0

    @staticmethod
    def _get_initials(name: str) -> str:
        """Extract multi-word uppercase initials excluding common legal stopwords."""
        clean = re.sub(r"[^\w\s]", "", name or "")
        words = [
            w for w in clean.split()
            if w.lower() not in LEGAL_TERMS
            and w.lower() not in ("pvt", "ltd", "inc", "corp", "llc", "limited", "private", "co", "llp")
        ]
        if len(words) >= 2:
            return "".join(w[0].upper() for w in words if w)
        return ""

    def generate_candidates(
        self,
        name: str,
        address: str,
        country: str,
        k: Optional[int] = None,
    ) -> List[str]:
        """Generate candidate entity IDs for a query entity."""
        limit = k if k is not None else self.max_candidates
        if self.strategy == STRATEGY_NORMALIZED_EXACT:
            return self._query_normalized_exact(name, limit=limit)

        elif self.strategy == STRATEGY_TOKEN_NAME:
            return self._query_token_name(name, limit=limit)

        elif self.strategy == STRATEGY_COUNTRY_NAME:
            return self._query_country_name(name, country, limit=limit)

        elif self.strategy == STRATEGY_ADDRESS_TOKEN:
            return self._query_address_token(address, country, limit=limit)

        elif self.strategy == STRATEGY_TFIDF_NGRAM:
            return self._query_tfidf_ngram(name, limit=limit)

        elif self.strategy == STRATEGY_COMBINED_HYBRID:
            return self._query_combined_hybrid(name, address, country, limit=limit)

        return []

    def _query_normalized_exact(self, name: str, limit: Optional[int] = None) -> List[str]:
        cap = limit if limit is not None else self.max_candidates
        norm = normalize_business_name(name)
        if not norm:
            return []
        return self.exact_index.get(norm, [])[:cap]

    def _query_token_name(self, name: str, limit: Optional[int] = None) -> List[str]:
        cap = limit if limit is not None else self.max_candidates
        tokens = extract_name_tokens(name)
        if not tokens:
            return []
        valid_lists = [self.name_token_index[t] for t in tokens if t in self.name_token_index]
        if not valid_lists:
            return []
        valid_lists.sort(key=len)

        candidates: Set[str] = set()
        for plist in valid_lists:
            candidates.update(plist)
            if len(candidates) >= cap:
                break
        return list(candidates)[:cap]

    def _query_country_name(self, name: str, country: str, limit: Optional[int] = None) -> List[str]:
        cap = limit if limit is not None else self.max_candidates
        tokens = extract_name_tokens(name)
        clean_country = country.strip() if country else ""
        if not tokens:
            return []

        valid_lists = [
            self.country_token_index[(clean_country, t)]
            for t in tokens
            if (clean_country, t) in self.country_token_index
        ]
        if not valid_lists:
            return []
        valid_lists.sort(key=len)

        candidates: Set[str] = set()
        for plist in valid_lists:
            candidates.update(plist)
            if len(candidates) >= cap:
                break
        return list(candidates)[:cap]

    def _query_address_token(self, address: str, country: str, limit: Optional[int] = None) -> List[str]:
        cap = limit if limit is not None else self.max_candidates
        tokens = extract_address_tokens(address)
        clean_country = country.strip() if country else ""
        if not tokens:
            return []

        valid_lists = [
            self.address_token_index[(clean_country, t)]
            for t in tokens
            if (clean_country, t) in self.address_token_index
        ]
        if not valid_lists:
            return []
        valid_lists.sort(key=len)

        candidates: Set[str] = set()
        for plist in valid_lists:
            candidates.update(plist)
            if len(candidates) >= cap:
                break
        return list(candidates)[:cap]

    def _query_tfidf_ngram(self, name: str, limit: Optional[int] = None) -> List[str]:
        cap = limit if limit is not None else self.max_candidates
        ngrams = extract_char_ngrams(name, n=3)
        if not ngrams:
            return []

        valid_ngrams = [ng for ng in ngrams if ng in self.ngram_index]
        if not valid_ngrams:
            return []

        # Prioritize top 6 most discriminative (highest IDF) n-grams
        valid_ngrams.sort(key=lambda x: self.ngram_idf.get(x, 1.0), reverse=True)
        top_ngrams = valid_ngrams[:6]

        scores: Dict[str, float] = defaultdict(float)
        for ng in top_ngrams:
            idf = self.ngram_idf.get(ng, 1.0)
            postings = self.ngram_index[ng]
            for eid in postings[:1000]:
                scores[eid] += idf

        if not scores:
            return []

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:cap]
        return [eid for eid, _ in top_k]

    def _query_combined_hybrid_scored(
        self,
        name: str,
        address: str,
        country: str,
        limit: Optional[int] = None,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, bool]]]:
        """Hybrid union returning ranked candidates with scores and channel evidence."""
        budget = limit if limit is not None else self.max_candidates
        scores: Dict[str, float] = defaultdict(float)
        evidence: Dict[str, Dict[str, bool]] = defaultdict(lambda: {
            "exact": False,
            "name_tok": False,
            "addr_tok": False,
            "tfidf": False,
        })
        clean_country = country.strip() if country else ""

        # 1. Normalized-name exact match (High precision anchor: +1000.0)
        norm_name = normalize_business_name(name, transliterate=True)
        raw_norm_name = normalize_business_name(name, transliterate=False)
        for cand_key in set(filter(None, [norm_name, raw_norm_name])):
            if clean_country and (clean_country, cand_key) in self.country_exact_index:
                for eid in self.country_exact_index[(clean_country, cand_key)]:
                    scores[eid] += 1000.0
                    evidence[eid]["exact"] = True
            elif cand_key in self.exact_index:
                for eid in self.exact_index[cand_key]:
                    scores[eid] += 1000.0
                    evidence[eid]["exact"] = True

        # 2. Token-name (Country-partitioned with unconstrained fallback: +10.0)
        name_tokens = extract_name_tokens(name)
        if name_tokens:
            valid_tokens = [
                t for t in name_tokens
                if (clean_country, t) in self.country_token_index or t in self.name_token_index
            ]
            valid_tokens.sort(
                key=lambda t: len(self.country_token_index.get((clean_country, t), []))
                if (clean_country, t) in self.country_token_index
                else len(self.name_token_index.get(t, []))
            )
            for t in valid_tokens[:5]:
                plist = (
                    self.country_token_index[(clean_country, t)]
                    if (clean_country, t) in self.country_token_index
                    else self.name_token_index[t]
                )
                for eid in plist[:500]:
                    scores[eid] += 10.0
                    evidence[eid]["name_tok"] = True

        # 2b. Acronyms & Short Codes (+60.0)
        q_inits = self._get_initials(name)
        if len(q_inits) >= 2:
            matched_eids = self.short_name_index.get((clean_country, q_inits), []) or self.short_name_index.get(("", q_inits), [])
            for eid in matched_eids[:100]:
                scores[eid] += 60.0
                evidence[eid]["name_tok"] = True

        clean_q_name = re.sub(r"[^\w]", "", name).upper()
        if 2 <= len(clean_q_name) <= 5:
            matched_eids = self.acronym_index.get((clean_country, clean_q_name), []) or self.acronym_index.get(("", clean_q_name), [])
            for eid in matched_eids[:100]:
                scores[eid] += 60.0
                evidence[eid]["name_tok"] = True

        # 2c. Compound / Concatenated Name Matching (+50.0)
        # Recovers domain-name / token-concatenated variants like Foot & Ankle -> Footanklecom, Parasnath Trust -> Parasnathtrust.com
        if name_tokens and len(name_tokens) >= 2:
            joined_toks = "".join(name_tokens).lower()
            if 4 <= len(joined_toks) <= 35:
                matched_compound = (
                    self.compound_name_index.get((clean_country, joined_toks), []) or
                    self.compound_name_index.get((clean_country, joined_toks + "com"), []) or
                    self.compound_name_index.get(("", joined_toks), []) or
                    self.compound_name_index.get(("", joined_toks + "com"), [])
                )
                for eid in matched_compound[:100]:
                    scores[eid] += 50.0
                    evidence[eid]["name_tok"] = True

        # 3. Address-token (PIN codes +15.0, numbers +10.0, text tokens +4.0) with co-occurrence boost
        addr_tokens = extract_address_tokens(address)
        addr_hit_count: Dict[str, int] = defaultdict(int)
        addr_has_number: Dict[str, bool] = defaultdict(bool)
        addr_has_text: Dict[str, bool] = defaultdict(bool)

        if addr_tokens:
            valid_addrs = [
                at for at in addr_tokens
                if (clean_country, at) in self.address_token_index or at in self.raw_address_token_index
            ]
            valid_addrs.sort(
                key=lambda at: len(self.address_token_index.get((clean_country, at), []))
                if (clean_country, at) in self.address_token_index
                else len(self.raw_address_token_index.get(at, []))
            )
            for at in valid_addrs[:6]:
                is_num = at.isdigit() or any(ch.isdigit() for ch in at)
                if at.isdigit() and len(at) in (5, 6):
                    w = 15.0
                elif is_num:
                    w = 10.0
                else:
                    w = 4.0
                plist = (
                    self.address_token_index[(clean_country, at)]
                    if (clean_country, at) in self.address_token_index
                    else self.raw_address_token_index[at]
                )
                for eid in plist[:500]:
                    scores[eid] += w
                    evidence[eid]["addr_tok"] = True
                    addr_hit_count[eid] += 1
                    if is_num:
                        addr_has_number[eid] = True
                    else:
                        addr_has_text[eid] = True

            # Boost address co-occurrence
            for eid, hits in addr_hit_count.items():
                if hits >= 2:
                    scores[eid] += 20.0 * (hits - 1)
                if hits >= 3:
                    scores[eid] += 30.0
                if addr_has_number[eid] and addr_has_text[eid]:
                    scores[eid] += 25.0

        # 4. Country-constrained TF-IDF character 3-grams with noise dampening on pure distractors
        ngrams = extract_char_ngrams(name, n=3)
        trans_name = transliterate_text(name)
        if trans_name != name:
            ngrams = list(dict.fromkeys(ngrams + extract_char_ngrams(trans_name, n=3)))
        if ngrams:
            valid_ngs = [
                ng for ng in ngrams
                if (clean_country, ng) in self.country_ngram_index or ng in self.ngram_index
            ]
            valid_ngs.sort(
                key=lambda ng: len(self.country_ngram_index.get((clean_country, ng), []))
                if (clean_country, ng) in self.country_ngram_index
                else len(self.ngram_index.get(ng, []))
            )
            ngram_gains: Dict[str, float] = defaultdict(float)
            for ng in valid_ngs[:6]:
                idf = self.country_ngram_idf.get(
                    (clean_country, ng),
                    self.ngram_idf.get(ng, 1.0)
                )
                plist = (
                    self.country_ngram_index[(clean_country, ng)]
                    if (clean_country, ng) in self.country_ngram_index
                    else self.ngram_index[ng]
                )
                for eid in plist[:500]:
                    ngram_gains[eid] += 2.0 * idf
                    evidence[eid]["tfidf"] = True

            for eid, gain in ngram_gains.items():
                # If candidate has NO other evidence (no exact, no name token, no addr token):
                # cap pure ngram contribution to 18.0 to prevent flooding out real candidates
                if not evidence[eid]["exact"] and not evidence[eid]["name_tok"] and not evidence[eid]["addr_tok"]:
                    scores[eid] += min(gain, 18.0)
                else:
                    scores[eid] += gain

        if not scores:
            return [], {}

        top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:budget]
        return top_k, evidence

    def _query_combined_hybrid(
        self,
        name: str,
        address: str,
        country: str,
        limit: Optional[int] = None,
    ) -> List[str]:
        """Hybrid union: exact name + country-constrained TF-IDF + token name + address token.

        Accumulates weighted scores across all 4 channels with true UNION semantics.
        Candidate budget capping is strictly applied AFTER the full union.
        """
        budget = limit if limit is not None else self.max_candidates
        top_k, _ = self._query_combined_hybrid_scored(name, address, country, limit=budget)
        return [eid for eid, _ in top_k]

    def generate_candidates_with_metadata(
        self,
        name: str,
        address: str,
        country: str,
        k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidate entity IDs with rank, score, and blocking channel evidence.

        Args:
            name: Query entity name.
            address: Query entity address.
            country: Query entity country.
            k: Candidate budget cap.

        Returns:
            List of metadata dicts for each candidate entity.
        """
        budget = k if k is not None else self.max_candidates
        top_k, evidence = self._query_combined_hybrid_scored(name, address, country, limit=budget)
        if not top_k:
            return []

        top1_score = top_k[0][1] if top_k else 1.0
        total_cands = float(len(top_k))

        results = []
        for rank_idx, (eid, score) in enumerate(top_k):
            rank = float(rank_idx + 1)
            ev = evidence.get(eid, {})
            exact = 1.0 if ev.get("exact") else 0.0
            name_tok = 1.0 if ev.get("name_tok") else 0.0
            addr_tok = 1.0 if ev.get("addr_tok") else 0.0
            tfidf = 1.0 if ev.get("tfidf") else 0.0
            channel_count = exact + name_tok + addr_tok + tfidf

            results.append({
                "entity_id": eid,
                "cand_rank": rank,
                "cand_rank_reciprocal": 1.0 / rank,
                "cand_blocking_score": float(score),
                "cand_score_ratio_to_top1": (float(score) / top1_score) if top1_score > 0 else 0.0,
                "total_candidates_for_s1": total_cands,
                "log_total_candidates": math.log(1.0 + total_cands),
                "block_exact_name": exact,
                "block_name_token": name_tok,
                "block_address_token": addr_tok,
                "block_tfidf_ngram": tfidf,
                "block_evidence_channels_count": channel_count,
            })
        return results
