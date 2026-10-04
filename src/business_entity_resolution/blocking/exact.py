"""Exact blocking index implementations."""

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple
import pandas as pd
from business_entity_resolution.normalization.names import normalize_business_name


class ExactNameBlocker:
    """Exact blocking on normalized business name."""

    def __init__(self):
        self.index: Dict[str, List[str]] = defaultdict(list)

    def build_index(self, records: Iterable[Tuple[str, str]]):
        """Build exact block index from an iterable of (entity_id, business_name)."""
        self.index.clear()
        for entity_id, name in records:
            key = normalize_business_name(name)
            if key:
                self.index[key].append(entity_id)

    def get_candidates(self, name: str) -> List[str]:
        """Retrieve candidate IDs for a query name."""
        key = normalize_business_name(name)
        if not key:
            return []
        return self.index.get(key, [])
