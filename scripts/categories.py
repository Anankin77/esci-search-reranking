"""Rule-based query categories for error analysis (spec Phase 6).

All categories except `substitute_heavy` use the query text only, so they can be
computed for any query. `substitute_heavy` uses labels and is ANALYSIS-ONLY.
"""
from __future__ import annotations

import pandas as pd

from features import brands_in_text
from text_utils import colour_tokens, has_negation_cue, tokenize

TEXT_CATEGORIES = ["numeric", "negation", "brand_in_query", "colour_in_query", "short", "long"]
ANALYSIS_ONLY_CATEGORIES = ["substitute_heavy"]
CATEGORY_DESCRIPTIONS = {
    "numeric": "query contains a digit (sizes, quantities, model numbers)",
    "negation": "query contains without / no / not / non / free",
    "brand_in_query": "query names a brand from the training-split brand vocabulary",
    "colour_in_query": "query contains a colour word",
    "short": "query has at most 2 tokens",
    "long": "query has at least 6 tokens",
    "substitute_heavy": "at least 50% of candidates are labelled Substitute (label-based, analysis only)",
}


def text_categories(query: str, brand_vocab: frozenset[str]) -> dict[str, bool]:
    n = len(tokenize(query))
    return {
        "numeric": any(ch.isdigit() for ch in query),
        "negation": has_negation_cue(query),
        "brand_in_query": bool(brands_in_text(query, brand_vocab)),
        "colour_in_query": bool(colour_tokens(query)),
        "short": n <= 2,
        "long": n >= 6,
    }


def query_categories(pairs: pd.DataFrame, brand_vocab: frozenset[str]) -> pd.DataFrame:
    """One row per query with a boolean column per category."""
    queries = pairs.drop_duplicates("query_id")[["query_id", "query"]]
    rows = [{"query_id": qid, **text_categories(q, brand_vocab)}
            for qid, q in zip(queries["query_id"], queries["query"])]
    out = pd.DataFrame(rows)
    # analysis-only, label-based category
    share_s = pairs.groupby("query_id")["esci_label"].apply(lambda s: (s == "S").mean())
    out["substitute_heavy"] = out["query_id"].map(share_s) >= 0.5
    return out
