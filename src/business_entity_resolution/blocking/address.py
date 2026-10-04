"""Inverted index address token blocking."""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
from business_entity_resolution.normalization.addresses import extract_address_tokens


class AddressTokenBlocker:
    """Inverted index on normalized address tokens (street numbers, localities, postal codes)."""

    def __init__(self, max_token_frequency: Optional[int] = 50_000, max_candidates: int = 500):
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.max_token_frequency = max_token_frequency
        self.max_candidates = max_candidates

    def build_index(self, records: Iterable[Tuple[str, str]]):
        """Build inverted index from an iterable of (entity_id, business_address)."""
        self.index.clear()
        for entity_id, address in records:
            tokens = set(extract_address_tokens(address))
            for t in tokens:
                self.index[t].append(entity_id)

        if self.max_token_frequency is not None:
            pruned = {
                t: postings
                for t, postings in self.index.items()
                if len(postings) <= self.max_token_frequency
            }
            self.index = defaultdict(list, pruned)

    def get_candidates(self, address: str) -> List[str]:
        """Retrieve candidate IDs sharing informative address tokens."""
        tokens = extract_address_tokens(address)
        if not tokens:
            return []

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
