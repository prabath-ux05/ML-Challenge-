"""Business name normalization and tokenization utilities with transliteration and number normalization."""

import re
from typing import List, Set
from business_entity_resolution.normalization.common import (
    clean_text,
    normalize_digit_words,
    tokenize,
    transliterate_text,
)

# Common legal entities and noise suffixes in English and Indic languages
LEGAL_TERMS = {
    # English terms
    "inc", "incorporated", "llc", "corp", "corporation", "ltd", "limited",
    "co", "company", "pvt", "private", "llp", "gmbh", "sa", "sarl", "pllc",
    "plc", "group", "holdings", "enterprises", "enterprise", "ventures",
    "services", "solutions", "technologies", "international", "global",
    "center", "centre",

    # Indic terms (Hindi / Tamil transliterated or native)
    "प्राइवेट", "लिमिटेड", "एलएलपी", "எல்எல்பி", "பிரைவேட்", "லிமிடெட்",
}

# Regex to strip common domain patterns like .com, www.
DOMAIN_PATTERN = re.compile(r"^(?:https?://)?(?:www\.)?([a-z0-9\-_]+)(?:\.com|\.org|\.net|\.in|\.co|\.io|\.biz)?", re.IGNORECASE)


def normalize_business_name(name: str, transliterate: bool = True) -> str:
    """Normalize business name by transliterating, cleaning text, handling domain names, and stripping legal suffixes.

    Args:
        name: Raw business name string.
        transliterate: Whether to transliterate non-ASCII scripts to Latin.

    Returns:
        Normalized business name string.
    """
    if not name:
        return ""

    cleaned = name.strip()

    # Transliterate Indic / non-ASCII characters to Latin
    if transliterate:
        cleaned = transliterate_text(cleaned)

    # Strip web prefix/suffix if name looks like a website domain
    match = DOMAIN_PATTERN.match(cleaned)
    if match and ("." in cleaned or "www" in cleaned.lower()):
        cleaned = match.group(1)

    # Standard clean
    cleaned = clean_text(cleaned)

    # Normalize digit words
    cleaned = normalize_digit_words(cleaned)

    tokens = cleaned.split()

    # Filter out standalone legal terms
    filtered = [t for t in tokens if t not in LEGAL_TERMS]

    # If all tokens were filtered (e.g. name was literally just "Enterprises Inc"), retain original cleaned
    if not filtered:
        return cleaned

    return " ".join(filtered)


def extract_name_tokens(name: str, min_length: int = 2) -> List[str]:
    """Extract informative alphanumeric tokens from a business name, filtering common stopwords."""
    if not name:
        return []
    normalized = normalize_business_name(name, transliterate=True)
    tokens = tokenize(normalized, min_length=min_length)
    return [t for t in tokens if t not in LEGAL_TERMS]
