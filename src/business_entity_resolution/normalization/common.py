"""Common string manipulation, transliteration, and text sanitization utilities."""

import re
import unicodedata
from typing import List, Set

try:
    from anyascii import anyascii
except ImportError:  # pragma: no cover
    def anyascii(text: str) -> str:
        return text

# Regex to strip accents while keeping base characters
# Also regex for splitting tokens
TOKEN_PATTERN = re.compile(r"[\w\u0900-\u097F\u0B80-\u0BFF]+", re.UNICODE)
WHITESPACE_PATTERN = re.compile(r"\s+")
PUNCT_PATTERN = re.compile(r"[^\w\s\u0900-\u097F\u0B80-\u0BFF]", re.UNICODE)

# Cardinal and ordinal number words mapping to digits
NUMBER_WORD_MAP = {
    # Ordinals
    "first": "1", "1st": "1",
    "second": "2", "2nd": "2",
    "third": "3", "3rd": "3",
    "fourth": "4", "4th": "4",
    "fifth": "5", "5th": "5",
    "sixth": "6", "6th": "6",
    "seventh": "7", "7th": "7",
    "eighth": "8", "8th": "8",
    "ninth": "9", "9th": "9",
    "tenth": "10", "10th": "10",
    "eleventh": "11", "11th": "11",
    "twelfth": "12", "12th": "12",
    "thirteenth": "13", "13th": "13",
    "fourteenth": "14", "14th": "14",
    "fifteenth": "15", "15th": "15",
    "sixteenth": "16", "16th": "16",
    "seventeenth": "17", "17th": "17",
    "eighteenth": "18", "18th": "18",
    "nineteenth": "19", "19th": "19",
    "twentieth": "20", "20th": "20",
    # Cardinals
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def strip_accents(text: str) -> str:
    """Normalize unicode and strip combining diacritical marks (e.g., É -> E)."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def transliterate_text(text: str) -> str:
    """Transliterate unicode/Indic scripts (e.g. Devanagari, Tamil) to ASCII Latin."""
    if not text:
        return ""
    return anyascii(text)


def normalize_digit_words(text: str) -> str:
    """Convert ordinal/cardinal number words to digits and strip leading zeros on integers."""
    if not text:
        return ""
    words = text.split()
    normalized = []
    for w in words:
        w_lower = w.lower()
        if w_lower in NUMBER_WORD_MAP:
            normalized.append(NUMBER_WORD_MAP[w_lower])
        elif w.isdigit() and len(w) > 1 and w.startswith("0"):
            # Strip leading zeros on street numbers: '0550' -> '550'
            normalized.append(str(int(w)))
        else:
            normalized.append(w)
    return " ".join(normalized)


def clean_text(text: str, transliterate: bool = True) -> str:
    """Lowercases, optionally transliterates, strips accents, removes punctuation, and collapses whitespace."""
    if not text:
        return ""
    if transliterate:
        text = transliterate_text(text)
    text = strip_accents(text).lower()
    text = PUNCT_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def tokenize(text: str, min_length: int = 2) -> List[str]:
    """Tokenize cleaned text into a list of alphanumeric/unicode tokens."""
    if not text:
        return []
    cleaned = clean_text(text)
    tokens = TOKEN_PATTERN.findall(cleaned)
    return [t for t in tokens if len(t) >= min_length]
