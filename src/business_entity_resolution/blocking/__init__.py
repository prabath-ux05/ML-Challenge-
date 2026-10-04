"""Candidate pair generation and indexing modules."""

from business_entity_resolution.blocking.candidate_generator import (
    ALL_STRATEGIES,
    CandidateGenerator,
    STRATEGY_ADDRESS_TOKEN,
    STRATEGY_COMBINED_HYBRID,
    STRATEGY_COUNTRY_NAME,
    STRATEGY_NORMALIZED_EXACT,
    STRATEGY_TFIDF_NGRAM,
    STRATEGY_TOKEN_NAME,
)
from business_entity_resolution.blocking.exact import ExactNameBlocker
from business_entity_resolution.blocking.token import TokenNameBlocker
from business_entity_resolution.blocking.address import AddressTokenBlocker
from business_entity_resolution.blocking.evaluator import (
    BlockingEvaluator,
    BlockingStrategyResult,
    SubgroupMetrics,
    format_blocking_report,
    format_blocking_v2_report,
    run_blocking_analysis,
    run_blocking_analysis_v2,
)

__all__ = [
    "ALL_STRATEGIES",
    "CandidateGenerator",
    "STRATEGY_NORMALIZED_EXACT",
    "STRATEGY_TOKEN_NAME",
    "STRATEGY_COUNTRY_NAME",
    "STRATEGY_ADDRESS_TOKEN",
    "STRATEGY_TFIDF_NGRAM",
    "STRATEGY_COMBINED_HYBRID",
    "ExactNameBlocker",
    "TokenNameBlocker",
    "AddressTokenBlocker",
    "BlockingEvaluator",
    "BlockingStrategyResult",
    "SubgroupMetrics",
    "format_blocking_report",
    "format_blocking_v2_report",
    "run_blocking_analysis",
    "run_blocking_analysis_v2",
]
