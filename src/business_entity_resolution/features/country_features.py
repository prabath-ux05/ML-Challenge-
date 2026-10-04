"""Country equality, categorical indicators, and missingness features."""

from typing import Dict, List

COUNTRY_FEATURE_NAMES: List[str] = [
    "country_exact_match",
    "country_is_us",
    "country_is_india",
    "country_s1_missing",
    "country_s2_missing",
    "country_either_missing",
]


def extract_country_features(
    country1: str,
    country2: str,
) -> Dict[str, float]:
    """Extract country equality, locale indicators, and missingness flags.

    Args:
        country1: Country string from Source 1.
        country2: Country string from Candidate source (Source 2 or Source 3).

    Returns:
        Dictionary mapping feature name to numeric float value.
    """
    c1 = (country1 or "").strip()
    c2 = (country2 or "").strip()

    miss1 = 1.0 if not c1 else 0.0
    miss2 = 1.0 if not c2 else 0.0
    either_miss = 1.0 if (miss1 == 1.0 or miss2 == 1.0) else 0.0

    match = 1.0 if (c1 and c2 and c1.upper() == c2.upper()) else 0.0
    is_us = 1.0 if c1.upper() == "US" else 0.0
    is_india = 1.0 if c1.lower() == "india" else 0.0

    return {
        "country_exact_match": match,
        "country_is_us": is_us,
        "country_is_india": is_india,
        "country_s1_missing": miss1,
        "country_s2_missing": miss2,
        "country_either_missing": either_miss,
    }
