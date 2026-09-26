"""Text normalisation for business names and addresses.

Everything here is country-agnostic by design: country only switches a few
abbreviation tables (e.g. French "bd" -> boulevard), never filters records.
"""
import re
from functools import lru_cache

from unidecode import unidecode

# ---------------------------------------------------------------------------
# Name canonicalisation: every variant maps to one canonical token.
# ---------------------------------------------------------------------------
NAME_CANON = {
    "corporation": "corp", "corpn": "corp", "corp": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "and": "and", "et": "and", "n": "and",
    "international": "intl", "intl": "intl", "internatl": "intl",
    "manufacturing": "mfg", "mfg": "mfg", "manufacturers": "mfg",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc", "servs": "svc",
    "associates": "assoc", "assoc": "assoc", "associated": "assoc",
    "brothers": "bros", "bros": "bros", "bro": "bros",
    "national": "natl", "natl": "natl",
    "management": "mgmt", "mgmt": "mgmt",
    "group": "grp", "grp": "grp", "groupe": "grp",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent", "entp": "ent", "entreprise": "ent",
    "industries": "ind", "industry": "ind", "inds": "ind", "ind": "ind", "industrie": "ind", "industries.": "ind",
    "technologies": "tech", "technology": "tech", "tech": "tech", "techs": "tech", "technologie": "tech",
    "solutions": "sol", "solution": "sol", "soln": "sol", "solns": "sol",
    "systems": "sys", "system": "sys", "sys": "sys", "systemes": "sys",
    "laboratories": "labs", "laboratory": "labs", "labs": "labs", "lab": "labs",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharma": "pharma",
    "department": "dept", "dept": "dept",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "restaurant": "rest", "restaurants": "rest",
    "hospital": "hosp", "hosp": "hosp", "hopital": "hosp",
    "limited liability company": "llc",
    "saint": "st", "sainte": "ste",
    "mister": "mr", "doctor": "dr",
    "traders": "traders", "trading": "trading",
    "agency": "agcy", "agencies": "agcy",
    "university": "univ", "univ": "univ",
    "insurance": "ins", "ins": "ins",
    "financial": "fin", "finance": "fin",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg",
    # French
    "societe": "soc", "soc": "soc",
    "association": "assn", "assn": "assn",
    "boulangerie": "boul", "patisserie": "pat",
    "pharmacie": "pharm", "librairie": "lib",
    "lycee": "lycee", "ecole": "ecole", "college": "college",
    "maison": "maison", "atelier": "atelier",
    "cabinet": "cab", "cab": "cab",
    "immobilier": "immo", "immo": "immo",
    "securite": "sec", "securite": "sec",
    "transport": "trans", "transports": "trans", "trans": "trans",
    "informatique": "info", "info": "info",
    "conseil": "conseil", "consultants": "consult", "consultant": "consult", "consulting": "consult",
    "garage": "garage",
    "menuiserie": "menuiserie", "plomberie": "plomberie",
    "coiffure": "coiffure",
    # Additional common terms
    "foundation": "fdn", "fdn": "fdn", "fondation": "fdn",
    "global": "global", "worldwide": "global",
    "electric": "elec", "electrical": "elec", "electronics": "elec", "elec": "elec",
    "communications": "comm", "communication": "comm", "comm": "comm",
    "construction": "const", "const": "const",
    "engineering": "engg", "engg": "engg", "engr": "engg",
    "general": "genl", "genl": "genl", "gen": "genl",
    "education": "edu", "edu": "edu", "educational": "edu",
    "medical": "med", "med": "med", "medicale": "med",
    "properties": "prop", "property": "prop", "prop": "prop",
    "development": "dev", "dev": "dev", "developpement": "dev",
    "marketing": "mktg", "mktg": "mktg",
    "products": "prod", "product": "prod", "prod": "prod", "produits": "prod",
    "supply": "supply", "supplies": "supply",
    "distribution": "dist", "distributing": "dist", "distributors": "dist",
    "foods": "food", "food": "food",
    "automobiles": "auto", "automotive": "auto", "auto": "auto",
}

# Legal-form / generic tokens removed to build the "core" name.
LEGAL_TOKENS = {
    "corp", "inc", "ltd", "pvt", "co", "llc", "llp", "lp", "plc", "pllc", "pc",
    "sa", "sas", "sarl", "eurl", "sasu", "sci", "snc", "scs", "selarl", "scop",
    "gmbh", "ag", "bv", "nv", "opc", "the", "and", "of", "le", "la", "les", "de",
    "du", "des", "l", "d", "et", "soc", "assn",
    # honorifics / prefixes seen in the data ("M/s ABC Traders", "Mr RKK Products", "Sri ...")
    "ms", "mr", "mrs", "sri", "shri", "shree", "smt", "null",
}

# Multi-word phrases collapsed before tokenising.
NAME_PHRASES = [
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\blimited\s+liability\s+(company|partnership)\b", "llc"),
    (r"\bl\s*l\s*c\b", "llc"),
    (r"\bl\s*l\s*p\b", "llp"),
    (r"\bp\s*l\s*c\b", "plc"),
    (r"\bs\s*a\s*r\s*l\b", "sarl"),
    (r"\bs\s*a\s*s\b", "sas"),
    (r"\bpvt\s*\.?\s*ltd\b", "pvt ltd"),
    (r"\bpte\s*\.?\s*ltd\b", "pvt ltd"),
]

# ---------------------------------------------------------------------------
# Address canonicalisation.
# ---------------------------------------------------------------------------
ADDR_CANON = {
    "street": "st", "str": "st", "st": "st", "strt": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "aven": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "expressway": "expy", "expy": "expy",
    "terrace": "ter", "ter": "ter",
    "circle": "cir", "cir": "cir",
    "suite": "ste", "ste": "ste",
    "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl", "flr": "fl",
    "building": "bldg", "bldg": "bldg", "bldng": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp",
    "behind": "bhd", "beside": "bsd",
    "sector": "sec", "sec": "sec", "sect": "sec",
    "phase": "ph", "ph": "ph",
    "extension": "extn", "extn": "extn", "ext": "extn",
    "nagar": "ngr", "ngr": "ngr",
    "colony": "col", "col": "col",
    "marg": "marg",
    "chowk": "chk",
    "cross": "crs", "crs": "crs",
    "main": "main", "mn": "main",
    "layout": "lyt",
    "industrial": "indl", "indl": "indl",
    "estate": "est", "est": "est",
    "area": "area",
    "post": "po", "po": "po",
    "district": "dist", "dist": "dist", "distt": "dist",
    "taluk": "tq", "taluka": "tq", "tq": "tq",
    "village": "vill", "vill": "vill", "vil": "vill",
    "mount": "mt", "mt": "mt",
    "fort": "ft", "ft": "ft",
    "number": "no", "no": "no", "num": "no", "nr.": "no",
    "rue": "rue", "r": "rue",
    "chemin": "ch", "ch": "ch",
    "impasse": "imp", "imp": "imp",
    "allee": "all", "all": "all",
    "faubourg": "fbg", "fbg": "fbg",
    "route": "rte", "rte": "rte",
    "quai": "qu", "qu": "qu",
    "cedex": "cedex",
    "saint": "st", "sainte": "ste",
    "bis": "bis", "ter.": "ter",
    "center": "ctr", "centre": "ctr", "ctr": "ctr",
    "hospital": "hosp",
    "station": "stn", "stn": "stn",
    "railway": "rly", "rly": "rly",
    "market": "mkt", "mkt": "mkt",
    "complex": "cmplx", "cmplx": "cmplx",
    "tower": "twr", "twr": "twr",
    "plaza": "plz", "plz": "plz",
    "junction": "jn", "jn": "jn", "jct": "jn",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
}

# Cities whose official name changed / have common transliteration variants.
CITY_CANON = {
    "bombay": "mumbai", "bangalore": "bengaluru", "bengalooru": "bengaluru",
    "madras": "chennai", "calcutta": "kolkata", "gurgaon": "gurugram",
    "pondicherry": "puducherry", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "poona": "pune", "mysore": "mysuru",
    "banaras": "varanasi", "benares": "varanasi", "allahabad": "prayagraj",
    "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi",
    "vizag": "visakhapatnam", "vishakhapatnam": "visakhapatnam", "simla": "shimla",
    "orissa": "odisha", "cawnpore": "kanpur", "trichy": "tiruchirappalli",
    "tiruchirapalli": "tiruchirappalli", "calicut": "kozhikode",
    "philly": "philadelphia",
    # French city variants
    "marseilles": "marseille", "lyons": "lyon",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "district of columbia": "dc",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "chattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml",
    "mizoram": "mz", "nagaland": "nl", "odisha": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "tamilnadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb",
    "delhi": "dl", "new delhi": "dl", "jammu and kashmir": "jk", "puducherry": "py",
    "chandigarh": "ch", "ladakh": "la",
}
_STATE_PATTERNS = sorted({**US_STATES, **IN_STATES}.items(), key=lambda kv: -len(kv[0]))
_STATE_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k, _ in _STATE_PATTERNS) + r")\b")
_STATE_MAP = dict(_STATE_PATTERNS)

_LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent to|in front of|close to|landmark|pres de|a cote de|face a|devant)\b[^,;]*",
    re.I,
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
_POSTAL6 = re.compile(r"(?<!\d)(\d{3})\s?(\d{3})(?!\d)")
_POSTAL5 = re.compile(r"(?<!\d)(\d{5})(?:-\d{4})?(?!\d)")
_DBA_RE = re.compile(r"\b(?:d\s*/?\s*b\s*/?\s*a|doing business as|t\s*/\s*a|trading as|aka|a\s*/\s*k\s*/\s*a)\b", re.I)
# French region names that can be dropped for matching (they're administrative, not address-specific)
_FR_REGIONS = {
    "ile de france", "nouvelle aquitaine", "auvergne rhone alpes", "occitanie",
    "hauts de france", "grand est", "provence alpes cote d azur", "pays de la loire",
    "bretagne", "normandie", "bourgogne franche comte", "centre val de loire",
    "corse",
}


_INDIC = re.compile(r"[\u0900-\u0DFF]")
_TRANSLIT_FIX = [  # tokens produced by unidecode on Devanagari/Kannada/Tamil/... legal words
    (re.compile(r"\b(?:pr[aiey]+[vb]h?[aeiry]*t+\w*|pr[aie]+v\w*)\b"), "private"),
    (re.compile(r"\blimi\w*\b|\blmtd\b"), "limited"),
    (re.compile(r"\bk[ae]mp[ae]?n\w*\b"), "company"),
]


def ascii_lower(s) -> str:
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = str(s)
    if _INDIC.search(s):  # transliterate, then undo unidecode's doubled letters
        t = unidecode(s).lower().replace("'", "")
        t = re.sub(r"([a-z])\1+", r"\1", t)
        for pat, rep in _TRANSLIT_FIX:
            t = pat.sub(rep, t)
        return t
    return unidecode(s).lower()


_WEB_RE = re.compile(r"(^|\s)(?:https?://)?(?:www\.)?([a-z0-9-]+)\.(?:com|net|org|in|co\.in|co|fr|us|biz|info|io)\b")
_HANDLE_RE = re.compile(r"(^|\s)@([a-z0-9_]+)")
_OCR_DIGIT = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "8": "b"}
_OCR_RE = re.compile(r"(?<=[a-z])[013458](?=[a-z])")


def _basic(s: str) -> str:
    s = s.replace("&", " and ").replace("@", " at ").replace("+", " and ")
    s = s.replace("'", "").replace("`", "")
    s = _NON_ALNUM.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _split_alnum(s: str) -> str:
    # "12b" -> "12 b", "no12" -> "no 12" : unify municipal numbering formats.
    s = re.sub(r"(\d)([a-z])", r"\1 \2", s)
    return re.sub(r"([a-z])(\d)", r"\1 \2", s)


@lru_cache(maxsize=1 << 16)
def norm_name(raw: str) -> str:
    s = ascii_lower(raw)
    s = _WEB_RE.sub(r"\1\2", s)  # "gulfbrc.com" -> "gulfbrc"
    s = _HANDLE_RE.sub(r"\1\2", s)  # "@casillasconcord" -> "casillasconcord"
    s = _OCR_RE.sub(lambda m: _OCR_DIGIT[m.group(0)], s)  # "fa1con" -> "falcon"
    s = re.sub(r"^\s*m\s*/\s*s\b", " ", s)  # "M/s ABC" -> "ABC"
    s = re.sub(r"\bpr[il]v[a-z]?te\b", "private", s)  # OCR "PRlVATE"
    s = re.sub(r"\bl[il]m[il]ted\b", "limited", s)
    s = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", s)  # "i.b.m" -> "ibm"
    s = s.replace(".", " ")
    for pat, rep in NAME_PHRASES:
        s = re.sub(pat, rep, s)
    s = _basic(s)
    return " ".join(NAME_CANON.get(t, t) for t in s.split())


@lru_cache(maxsize=1 << 16)
def name_variants(raw: str) -> tuple:
    """Full name plus DBA/trade-name parts and bracketed aliases."""
    s = ascii_lower(raw)
    parts = [s]
    if _DBA_RE.search(s):
        parts += [p for p in _DBA_RE.split(s) if p.strip()]
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        parts += [m.group(1), s[: m.start()] + s[m.end():]]
    out = []
    for p in parts:
        n = norm_name(p)
        if n and n not in out:
            out.append(n)
    return tuple(out)


@lru_cache(maxsize=1 << 16)
def core_name(normed: str) -> str:
    toks = [t for t in normed.split() if t not in LEGAL_TOKENS]
    return " ".join(toks) if toks else normed


def acronym(core: str) -> str:
    toks = core.split()
    return "".join(t[0] for t in toks if t and not t.isdigit()) if len(toks) > 1 else ""


@lru_cache(maxsize=1 << 16)
def phonetic(s: str) -> str:
    """Crude transliteration-tolerant key (Indian/French spelling variants, typos)."""
    s = re.sub(r"(ph)", "f", s)
    s = re.sub(r"([kgcbdtjp])h", r"\1", s)  # kh->k, sh stays, th->t, bh->b ...
    s = s.replace("ee", "i").replace("oo", "u").replace("ou", "u").replace("aa", "a")
    s = s.replace("w", "v").replace("z", "j").replace("q", "k").replace("ck", "k")
    s = s.replace("y", "i").replace("sh", "s").replace("x", "ks")
    s = re.sub(r"(.)\1+", r"\1", s)  # collapse doubles
    s = re.sub(r"(?<=[a-z])[aeiou]+(?=\b)", "", s)  # drop word-final vowels
    return s


def canon_country(c) -> str:
    return _WS.sub(" ", ascii_lower(c)).strip()


def _is_fr(country: str) -> bool:
    return country.startswith("fr") or "france" in country


@lru_cache(maxsize=1 << 16)
def norm_address(raw: str, country: str = "") -> dict:
    s0 = ascii_lower(raw)
    landmark = " ".join(m.group(0) for m in _LANDMARK_RE.finditer(s0))
    no_lm = _LANDMARK_RE.sub(" ", s0)

    pins = ["".join(m.groups()) for m in _POSTAL6.finditer(s0)]
    zips = [m.group(1) for m in _POSTAL5.finditer(s0)]
    postal = pins[-1] if pins else (zips[-1] if zips else "")

    def canon(text: str) -> str:
        t = _basic(text.replace(".", " "))
        t = _STATE_RE.sub(lambda m: _STATE_MAP[m.group(1)], t)
        t = _split_alnum(t)
        toks = []
        fr = _is_fr(country)
        for tok in t.split():
            if fr and tok == "st":
                toks.append("st")  # saint
                continue
            if tok in ("null", "none", "nan", "na"):
                continue
            tok = ADDR_CANON.get(tok, tok)
            toks.append(CITY_CANON.get(tok, tok))
        return " ".join(toks)

    full = canon(s0)
    core = canon(no_lm)
    # strip French region names from core (they are administrative, not address-specific)
    if _is_fr(country):
        core_low = core
        for reg in _FR_REGIONS:
            core_low = core_low.replace(reg, " ")
        core = _WS.sub(" ", core_low).strip()
    core_np = _POSTAL5.sub(" ", _POSTAL6.sub(" ", core)) if postal else core
    nums = _DIGITS.findall(core_np)
    return {
        "full": full,
        "core": core,
        "landmark": canon(landmark),
        "postal": postal,
        "nums": tuple(nums),
        "first_num": nums[0] if nums else "",
    }
