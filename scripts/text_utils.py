"""Text helpers: cleaning, tokenising, number/unit parsing, negation, colours, sizes.

Everything here is a pure function of the text: nothing is learned from data.
"""
from __future__ import annotations

import html
import re
import unicodedata

from config import TEXT_CHAR_BUDGET

# ---------------------------------------------------------------- cleaning
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: object, budget: int | None = None) -> str:
    """Strip HTML tags/entities, NFKC-normalise, collapse whitespace, truncate.

    Keeps the original case (used for display); lowercase separately for features.
    """
    if text is None or (isinstance(text, float) and text != text):  # None or NaN
        return ""
    s = str(text)
    s = _TAG_RE.sub(" ", s)          # tags like <br> or <p>
    s = html.unescape(s)              # entities like &amp; or &#39;
    s = _TAG_RE.sub(" ", s)          # tags that were themselves escaped (&lt;b&gt;)
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    if budget is not None and len(s) > budget:
        s = s[:budget].rsplit(" ", 1)[0]  # cut at a word boundary
    return s


def clean_long_text(text: object) -> str:
    """Cleaning for description / bullet points, truncated to the character budget."""
    return clean_text(text, TEXT_CHAR_BUDGET)


# ---------------------------------------------------------------- tokens
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
STOPWORDS = frozenset(
    "a an and the for of with in on to by at from or is it as be this that your you".split()
)


_APOSTROPHE_RE = re.compile(r"['\u2019](?=[a-z])")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; keeps decimals like '2.5' together.
    Apostrophes inside words are dropped: "levi's" -> "levis", "men's" -> "mens"."""
    return _TOKEN_RE.findall(_APOSTROPHE_RE.sub("", text.lower()))


def content_tokens(text: str) -> list[str]:
    """Tokens without a tiny stopword list (used for overlap features)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def normalize_query(q: str) -> str:
    """Normalisation used for near-duplicate detection (spec §4.3):
    lowercase, strip punctuation, collapse spaces, sort tokens."""
    return " ".join(sorted(tokenize(q)))


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Set of character n-grams of the lowercased, space-collapsed text."""
    s = " ".join(text.lower().split())
    return {s[i:i + n] for i in range(max(len(s) - n + 1, 0))}


# ---------------------------------------------------------------- numbers + units
# Canonical unit -> spellings. Order of alternatives matters in the regex (longest first).
UNIT_SYNONYMS = {
    "oz": ["fl oz", "fl. oz", "ounces", "ounce", "oz"],
    "ml": ["milliliters", "millilitres", "ml"],
    "l": ["liters", "litres", "liter", "litre", "l"],
    "lb": ["pounds", "pound", "lbs", "lb"],
    "kg": ["kilograms", "kilogram", "kgs", "kg"],
    "g": ["grams", "gram", "gr", "g"],
    "gb": ["gigabytes", "gigabyte", "gb"],
    "tb": ["terabytes", "terabyte", "tb"],
    "inch": ["inches", "inch", "in", '"', "''"],
    "ft": ["feet", "foot", "ft"],
    "pack": ["packs", "pack", "pk", "count", "ct", "pcs", "pieces", "piece", "pc"],
    "x": ["x"],
}
_UNIT_LOOKUP = {sp: canon for canon, sps in UNIT_SYNONYMS.items() for sp in sps}
_UNIT_ALT = "|".join(re.escape(s) for s in sorted(_UNIT_LOOKUP, key=len, reverse=True))
# a number, optional space or hyphen, optional unit that must end at a word boundary
_NUM_UNIT_RE = re.compile(
    rf"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*-?\s*({_UNIT_ALT})?(?![a-z0-9])", re.IGNORECASE
)
# "size 9", "size 10.5", "pack of 6"
_SIZE_NUM_RE = re.compile(r"\bsize\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_PACK_OF_RE = re.compile(r"\b(?:pack|set|box|case) of\s*(\d+)\b", re.IGNORECASE)


def _num(s: str) -> float:
    return float(s)


def parse_numbers_units(text: str) -> list[tuple[float, str | None]]:
    """Return (number, canonical unit or None) pairs found in text.

    '12 oz' -> (12, 'oz'); '6-pack' -> (6, 'pack'); 'size 9' -> (9, 'size');
    'pack of 4' -> (4, 'pack'); '2tb' -> (2, 'tb'); a bare '5' -> (5, None).
    """
    out: list[tuple[float, str | None]] = []
    t = text.lower()
    for m in _SIZE_NUM_RE.finditer(t):
        out.append((_num(m.group(1)), "size"))
    for m in _PACK_OF_RE.finditer(t):
        out.append((_num(m.group(1)), "pack"))
    covered = {m.start(1) for m in _SIZE_NUM_RE.finditer(t)} | {
        m.start(1) for m in _PACK_OF_RE.finditer(t)
    }
    for m in _NUM_UNIT_RE.finditer(t):
        if m.start(1) in covered:
            continue
        unit = _UNIT_LOOKUP.get(m.group(2).lower()) if m.group(2) else None
        if m.group(2) and m.group(2).lower() == "in" and re.match(r"\s*\d", t[m.end():]):
            unit = None  # "2 in 1" is not 2 inches
        out.append((_num(m.group(1)), unit))
    return out


def numbers_of(pairs: list[tuple[float, str | None]]) -> set[float]:
    return {n for n, _ in pairs}


def number_unit_features(query: str, title: str) -> dict[str, int]:
    """Agreement between numbers/units of the query and of the title.

    num_match    : some query number also appears in the title
    num_conflict : query and title both have numbers, but none is shared
    unit_match   : some (number, unit) pair of the query appears exactly in the title
    unit_conflict: a query unit appears in the title only with different numbers
    """
    q = parse_numbers_units(query)
    if not q:
        return {"num_match": 0, "num_conflict": 0, "unit_match": 0, "unit_conflict": 0}
    t = parse_numbers_units(title)
    qn, tn = numbers_of(q), numbers_of(t)
    q_units = {(n, u) for n, u in q if u}
    t_units = {(n, u) for n, u in t if u}
    unit_match = int(bool(q_units & t_units))
    unit_conflict = 0
    for n, u in q_units:
        t_nums_same_unit = {tn_ for tn_, tu in t_units if tu == u}
        if t_nums_same_unit and n not in t_nums_same_unit:
            unit_conflict = 1
    return {
        "num_match": int(bool(qn & tn)),
        "num_conflict": int(bool(tn) and not (qn & tn)),
        "unit_match": unit_match,
        "unit_conflict": unit_conflict * (1 - unit_match),
    }


# ---------------------------------------------------------------- sizes
SIZE_WORDS = {
    "xxs": "xxs", "xs": "xs", "small": "s", "medium": "m", "large": "l",
    "xl": "xl", "xxl": "xxl", "2xl": "xxl", "xxxl": "xxxl", "3xl": "xxxl",
    "4xl": "4xl", "5xl": "5xl", "x-large": "xl", "xx-large": "xxl",
    "x-small": "xs", "petite": "petite", "plus": "plus", "king": "king",
    "queen": "queen", "twin": "twin", "full": "full",
}
_SIZE_WORD_RE = re.compile(
    r"(?<![a-z0-9-])(" + "|".join(sorted(map(re.escape, SIZE_WORDS), key=len, reverse=True))
    + r")(?![a-z0-9-])"
)


def size_tokens(text: str) -> set[str]:
    """Canonical size tokens: words like 'xl', 'small', 'queen' plus 'size N' as 'size:N'."""
    t = text.lower()
    out = {SIZE_WORDS[m.group(1)] for m in _SIZE_WORD_RE.finditer(t)}
    out |= {f"size:{float(m.group(1)):g}" for m in _SIZE_NUM_RE.finditer(t)}
    return out


def size_features(query: str, title: str) -> dict[str, int]:
    """size_match: a query size token is in the title; size_conflict: both have
    size tokens of the same kind but none is shared."""
    qs = size_tokens(query)
    if not qs:
        return {"size_match": 0, "size_conflict": 0}
    ts = size_tokens(title)
    match = bool(qs & ts)
    return {"size_match": int(match), "size_conflict": int(bool(ts) and not match)}


# ---------------------------------------------------------------- colours
COLOURS = frozenset(
    """black white red blue green yellow orange purple pink brown grey gray silver gold
    beige navy teal turquoise maroon burgundy ivory cream tan khaki olive charcoal
    lavender violet magenta cyan coral mint aqua bronze copper rose clear transparent
    multicolor multicolour""".split()
)
_COLOUR_CANON = {"gray": "grey", "multicolour": "multicolor", "transparent": "clear"}


def colour_tokens(text: str) -> set[str]:
    """Colour words in text, canonicalised (gray -> grey)."""
    return {_COLOUR_CANON.get(t, t) for t in tokenize(text) if t in COLOURS}


# ---------------------------------------------------------------- negation
NEGATION_PREFIX_CUES = ("without", "no", "not", "non")  # cue comes before the term
NEGATION_SUFFIX_CUES = ("free",)                        # "sugar free", "sugar-free"
_NEG_PREFIX_RE = re.compile(r"\b(without|no|not|non)[\s-]+([a-z0-9]+)")
_NEG_SUFFIX_RE = re.compile(r"\b([a-z0-9]+)[\s-]?free\b")


def negated_terms(query: str) -> list[str]:
    """Terms the query asks NOT to have: 'shampoo without sulfate' -> ['sulfate'],
    'gluten-free pasta' -> ['gluten'], 'non toxic' -> ['toxic']."""
    q = query.lower()
    terms = [m.group(2) for m in _NEG_PREFIX_RE.finditer(q)]
    terms += [m.group(1) for m in _NEG_SUFFIX_RE.finditer(q)
              if m.group(1) not in ("for", "and", "the", "a", "hands", "tax")]
    return [t for t in terms if t not in ("free",)]


def has_negation_cue(query: str) -> bool:
    toks = tokenize(query.replace("-", " "))
    return any(t in NEGATION_PREFIX_CUES or t in NEGATION_SUFFIX_CUES for t in toks)


def negation_features(query: str, title: str) -> dict[str, int]:
    """neg_cue: query contains a negation cue.
    neg_phrase_in_title: the title states the same negation ('sugar free').
    neg_term_conflict: a negated term appears in the title WITHOUT a negation next
    to it, i.e. the product likely has the thing the shopper wants to avoid."""
    cue = has_negation_cue(query)
    if not cue:
        return {"neg_cue": 0, "neg_phrase_in_title": 0, "neg_term_conflict": 0}
    t = title.lower()
    phrase, conflict = 0, 0
    for term in negated_terms(query):
        negated_in_title = re.search(
            rf"\b{re.escape(term)}[\s-]?free\b|\b(?:no|non|without|not)[\s-]+{re.escape(term)}\b", t
        )
        if negated_in_title:
            phrase = 1
        elif re.search(rf"\b{re.escape(term)}\b", t):
            conflict = 1
    return {"neg_cue": 1, "neg_phrase_in_title": phrase, "neg_term_conflict": conflict}
