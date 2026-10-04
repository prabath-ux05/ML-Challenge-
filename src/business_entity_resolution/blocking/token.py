"""Inverted index token-based blocking for business names."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
from business_entity_resolution.normalization.names import extract_name_tokens


class TokenNameBlocker:
    """Inverted index on normalized business name tokens."""

    def __init__(self, max_token_frequency: Optional[int] = 50_000, max_candidates: int = 500):
        """Initialize TokenNameBlocker.

        Args:
            max_token_frequency: Max posting list length. Tokens appearing more often are pruned.
            max_candidates: Maximum candidates returned per query to prevent memory explosion.
        """
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.max_token_frequency = max_token_frequency
        self.max_candidates = max_candidates

    def build_index(self, records: Iterable[Tuple[str, str]]):
        """Build inverted index from an iterable of (entity_id, business_name)."""
        self.index.clear()
        for entity_id, name in records:
            tokens = set(extract_name_tokens(name))
            for t in tokens:
                self.index[t].append(entity_id)

        # Prune ultra-frequent tokens if specified
        if self.max_token_frequency is not None:
            pruned = {
                t: postings
                for t, postings in self.index.items()
                if len(postings) <= self.max_token_frequency
            }
            self.index = defaultdict(list, pruned)

    def get_candidates(self, name: str) -> List[str]:
        """Retrieve candidate IDs sharing at least one informative name token."""
        tokens = extract_name_tokens(name)
        if not tokens:
            return []

        # Sort tokens by inverted list length (rarest first) for efficiency
        valid_lists = [self.index[t] for t in tokens if t in self.index]
        if not valid_lists:
            return []

        valid_lists.sort(key=len)

        candidates: Set[str] = set()
        for plist in valid_lists:
            candidates.update(plist)
            if len(candidates) >= self.max_candidates:
                break

        return list(candidates)[:self.max_candidates]
