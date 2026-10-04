"""Address normalization, street abbreviation, locality synonym, and token extraction utilities."""

import re
from typing import List, Set
from business_entity_resolution.normalization.common import (
    clean_text,
    normalize_digit_words,
    tokenize,
    transliterate_text,
)

# Standard street designator replacements
STREET_ABBREVIATIONS = {
    "street": "st",
    "saint": "st",
    "avenue": "ave",
    "road": "rd",
    "drive": "dr",
    "lane": "ln",
    "boulevard": "blvd",
    "court": "ct",
    "terrace": "ter",
    "circle": "cir",
    "highway": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "square": "sq",
    "apartment": "apt",
    "suite": "ste",
    "building": "bldg",
    "floor": "fl",
}

# Comprehensive state mappings (all US states + Indian states and Union Territories)
STATE_ABBREVIATIONS = {
    # US States
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",

    # Indian States & Union Territories
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "telangana": "ts", "tripura": "tr", "uttarakhand": "uk", "uttaranchal": "uk",
    "uttar pradesh": "up", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}

# Major city and locality synonyms (historic / colloquial variants)
CITY_SYNONYMS = {
    "bombay": "mumbai",
    "mumbai": "mumbai",
    "madras": "chennai",
    "chennai": "chennai",
    "calcutta": "kolkata",
    "kolkata": "kolkata",
    "bangalore": "bengaluru",
    "bengaluru": "bengaluru",
    "baroda": "vadodara",
    "vadodara": "vadodara",
    "gurgaon": "gurugram",
    "gurugram": "gurugram",
    "allahabad": "prayagraj",
    "prayagraj": "prayagraj",
    "benares": "varanasi",
    "banaras": "varanasi",
    "kashi": "varanasi",
    "varanasi": "varanasi",
    "cochin": "kochi",
    "kochi": "kochi",
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "calicut": "kozhikode",
    "kozhikode": "kozhikode",
    "pondicherry": "puducherry",
    "puducherry": "puducherry",
    "poona": "pune",
    "pune": "pune",
    "nyc": "new york",
    "new york city": "new york",
}

PIN_CODE_PATTERN = re.compile(r"\b\d{5,6}\b")


def normalize_address(address: str) -> str:
    """Normalize an address string with transliteration, number normalization, street suffixes, state abbreviations, and city synonyms."""
    if not address:
        return ""

    # 1. Transliterate any Indic/non-ASCII characters
    transliterated = transliterate_text(address)

    # 2. Clean text
    cleaned = clean_text(transliterated)

    # 3. Normalize ordinal and cardinal number words & strip leading zeros
    cleaned = normalize_digit_words(cleaned)

    # 4. Standardize street abbreviations and city synonyms
    tokens = cleaned.split()
    normalized_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in STREET_ABBREVIATIONS:
            normalized_tokens.append(STREET_ABBREVIATIONS[token_lower])
        elif token_lower in CITY_SYNONYMS:
            normalized_tokens.append(CITY_SYNONYMS[token_lower])
        else:
            normalized_tokens.append(token_lower)

    result = " ".join(normalized_tokens)

    # 5. Standardize state names to standard 2-letter codes
    for full_name, abbr in STATE_ABBREVIATIONS.items():
        result = re.sub(r"\b" + full_name + r"\b", abbr, result)

    return result


def extract_address_tokens(address: str, min_length: int = 2) -> List[str]:
    """Extract informative address tokens (street numbers, PIN codes, localities, city synonyms)."""
    if not address:
        return []

    normalized = normalize_address(address)
    tokens = tokenize(normalized, min_length=min_length)

    # Extract any 5 or 6 digit PIN/ZIP codes directly
    raw_pins = PIN_CODE_PATTERN.findall(address)
    pin_tokens = [p for p in raw_pins]

    # Filter pure generic noise words
    noise = {"near", "opposite", "opp", "block", "floor", "unit", "null", "none"}
    clean_tokens = [t for t in tokens if t not in noise]

    # Add PIN tokens and city synonyms if present
    all_tokens = list(dict.fromkeys(clean_tokens + pin_tokens))

    # Also include original city synonym token if mapped (e.g. if 'mumbai', also include 'bombay')
    synonym_expansions = []
    for t in all_tokens:
        for k_syn, v_syn in CITY_SYNONYMS.items():
            if t == v_syn and k_syn != v_syn and k_syn not in all_tokens:
                synonym_expansions.append(k_syn)

    return all_tokens + synonym_expansions
