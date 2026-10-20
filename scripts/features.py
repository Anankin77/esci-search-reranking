"""Feature matrix for (query, product) pairs.

Two kinds of features:
- static: pure functions of one pair's text (overlap, numbers, negation, cosine, ...).
  Nothing is learned, so they are computed once per split and cached.
- fitted: need statistics learned from a training set - BM25 IDF/avgdl per field and
  the brand vocabulary. `fit_state` learns them from the training part of a fold
  (or the full training split for the final model); `fitted_features` applies them.

Leakage guard: features are computed from `feature_input(pairs)`, which keeps only
query/product text columns. `assert_no_labels` checks inputs and outputs.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from config import (BM25_B, BM25_K1, ROOT, RunPaths, base_arg_parser, get_paths,
                    log_runtime)
from embeddings import cosine_features, semantic_available, title_bullets_text
from text_utils import (COLOURS, STOPWORDS, char_ngrams, colour_tokens, content_tokens,
                        negation_features, number_unit_features, size_features, tokenize)

LABEL_COLUMNS = ("esci_label", "gain", "grade", "label", "split")
INPUT_COLUMNS = ["query_id", "query", "product_id", "title", "description", "bullets",
                 "brand", "colour"]

# ---------------------------------------------------------------- registry
# name -> (group, formula, why it might help). FEATURES.md is generated from this.
FEATURE_DOCS: dict[str, tuple[str, str, str]] = {
    "bm25_title": ("lexical", "BM25(query, title), IDF from training-fold titles",
                   "classic keyword relevance on the most informative field"),
    "bm25_bullets": ("lexical", "BM25(query, bullet points)",
                     "attributes often appear only in bullets"),
    "bm25_description": ("lexical", "BM25(query, description)",
                         "long-form text catches extra query terms"),
    "bm25_all": ("lexical", "BM25(query, title + brand + colour + bullets)",
                 "one keyword score over all short catalog fields"),
    "jaccard_title": ("lexical", "|Q ∩ T| / |Q ∪ T| on content tokens",
                      "symmetric overlap; penalises long off-topic titles"),
    "coverage_title": ("lexical", "share of query content tokens found in the title",
                       "did the title mention everything the shopper typed?"),
    "coverage_all": ("lexical", "share of query content tokens found in title/brand/colour/bullets",
                     "same, but over all short fields"),
    "char3_jaccard_title": ("lexical", "Jaccard of character 3-gram sets of query and title",
                            "robust to typos, plurals and glued words"),
    "phrase_in_title": ("lexical", "1 if the whole lowercased query appears in the title",
                        "exact phrase match is strong evidence"),
    "first_token_in_title": ("lexical", "1 if the first query token is in the title",
                             "first token is often the brand or main noun"),
    "last_token_in_title": ("lexical", "1 if the last query token is in the title",
                            "in English queries the head noun is often last"),
    "brand_match": ("attribute", "1 if the product's brand appears in the query",
                    "shopper asked for this brand"),
    "brand_conflict": ("attribute", "1 if the query names a known brand (training vocabulary) "
                       "and the product has a different brand",
                       "wrong brand is usually a substitute, not exact"),
    "colour_match": ("attribute", "1 if a query colour is in the product colour "
                     "(title colours if colour field empty)", "colour requested and present"),
    "colour_conflict": ("attribute", "1 if the query has a colour and the product has only "
                        "other colours", "wrong colour is a common substitute"),
    "num_match": ("numeric", "1 if a query number also appears in the title",
                  "model numbers, sizes and quantities agree"),
    "num_conflict": ("numeric", "1 if query and title have numbers but share none",
                     "different size/quantity/model"),
    "unit_match": ("numeric", "1 if a query (number, unit) pair appears in the title",
                   "'12 oz' asked and offered"),
    "unit_conflict": ("numeric", "1 if a query unit appears in the title only with other numbers",
                      "'12 oz' asked but '32 oz' offered"),
    "size_match": ("numeric", "1 if a query size token (xl, size 9, queen...) is in the title",
                   "size agreement for clothes, shoes, bedding"),
    "size_conflict": ("numeric", "1 if both have size tokens but none is shared",
                      "wrong size"),
    "neg_cue": ("negation", "1 if the query has a negation cue (without/no/not/non/free)",
                "keyword matching rewards the very term the shopper wants to avoid"),
    "neg_phrase_in_title": ("negation", "1 if the title repeats the negation ('sulfate-free')",
                            "product satisfies the negative constraint"),
    "neg_term_conflict": ("negation", "1 if a negated term appears in the title without negation",
                          "product likely contains what the shopper wants to avoid"),
    "cos_title": ("semantic", "cosine(MiniLM(query), MiniLM(title))",
                  "meaning-level similarity; handles synonyms"),
    "cos_title_bullets": ("semantic", "cosine(MiniLM(query), MiniLM(title + bullets))",
                          "semantic match using attribute text too"),
    "title_len": ("quality", "number of title tokens", "very short/long titles behave differently"),
    "no_description": ("quality", "1 if description is empty", "sparse listings may be lower quality"),
    "no_bullets": ("quality", "1 if bullet points are empty", "sparse listings"),
    "no_brand": ("quality", "1 if brand is empty", "sparse listings; brand features undefined"),
    "no_colour": ("quality", "1 if colour is empty", "colour features fall back to title"),
    "query_len": ("context", "number of query tokens", "long queries are more specific"),
    "query_has_digit": ("context", "1 if the query contains a digit", "numeric queries are harder"),
    "n_candidates": ("context", "number of candidates of this query", "list size changes base rates"),
    "bm25_all_rank": ("within_query", "rank of bm25_all within the query (1 = highest)",
                      "relative keyword strength among the candidates"),
    "bm25_all_z": ("within_query", "z-score of bm25_all within the query",
                   "relative keyword strength, scale-free"),
    "cos_title_rank": ("within_query", "rank of cos_title within the query (1 = highest)",
                       "relative semantic strength among the candidates"),
    "cos_title_z": ("within_query", "z-score of cos_title within the query",
                    "relative semantic strength, scale-free"),
}
FEATURE_NAMES = list(FEATURE_DOCS)
FEATURE_GROUPS: dict[str, list[str]] = {}
for _name, (_group, _, _) in FEATURE_DOCS.items():
    FEATURE_GROUPS.setdefault(_group, []).append(_name)
SEMANTIC_DEPENDENT = ["cos_title", "cos_title_bullets", "cos_title_rank", "cos_title_z"]

# Plain-language labels for the website's contribution bars.
FEATURE_LABELS = {
    "bm25_title": "keyword match in title", "bm25_bullets": "keyword match in bullets",
    "bm25_description": "keyword match in description", "bm25_all": "keyword match (all fields)",
    "jaccard_title": "word overlap with title", "coverage_title": "query words found in title",
    "coverage_all": "query words found anywhere", "char3_jaccard_title": "spelling-level overlap",
    "phrase_in_title": "exact phrase in title", "first_token_in_title": "first query word in title",
    "last_token_in_title": "last query word in title", "brand_match": "brand matches query",
    "brand_conflict": "brand conflicts with query", "colour_match": "colour matches query",
    "colour_conflict": "colour conflicts with query", "num_match": "number matches query",
    "num_conflict": "number conflicts with query", "unit_match": "quantity+unit matches",
    "unit_conflict": "quantity+unit conflicts", "size_match": "size matches query",
    "size_conflict": "size conflicts with query", "neg_cue": "query has a negation",
    "neg_phrase_in_title": "title repeats the negation", "neg_term_conflict": "title has the excluded term",
    "cos_title": "meaning similar to title", "cos_title_bullets": "meaning similar to title+bullets",
    "title_len": "title length", "no_description": "no description", "no_bullets": "no bullet points",
    "no_brand": "no brand listed", "no_colour": "no colour listed", "query_len": "query length",
    "query_has_digit": "query has a number", "n_candidates": "number of candidates",
    "bm25_all_rank": "keyword rank among candidates", "bm25_all_z": "keyword score vs other candidates",
    "cos_title_rank": "meaning rank among candidates", "cos_title_z": "meaning score vs other candidates",
}

FITTED_FEATURES = ["bm25_title", "bm25_bullets", "bm25_description", "bm25_all",
                   "brand_conflict", "bm25_all_rank", "bm25_all_z"]
STATIC_FEATURES = [f for f in FEATURE_NAMES if f not in FITTED_FEATURES]
BM25_FIELDS = {"bm25_title": "title", "bm25_bullets": "bullets",
               "bm25_description": "description", "bm25_all": "all_fields"}


def available_features(paths: RunPaths) -> list[str]:
    """All features, minus the semantic ones if the encoder could not be loaded."""
    if semantic_available(paths):
        return FEATURE_NAMES
    return [f for f in FEATURE_NAMES if f not in SEMANTIC_DEPENDENT]


# ---------------------------------------------------------------- leakage guard
def assert_no_labels(df: pd.DataFrame) -> None:
    bad = [c for c in df.columns if c in LABEL_COLUMNS]
    assert not bad, f"label columns must never reach feature code: {bad}"


def feature_input(pairs: pd.DataFrame) -> pd.DataFrame:
    """The only columns feature code may see: query and catalog text."""
    out = pairs[INPUT_COLUMNS].copy()
    assert_no_labels(out)
    return out


def all_fields_text(df: pd.DataFrame) -> pd.Series:
    return df["title"] + " " + df["brand"] + " " + df["colour"] + " " + df["bullets"]


# ---------------------------------------------------------------- BM25
@dataclass
class BM25Stats:
    """IDF and average document length learned by rank_bm25.BM25Okapi on a corpus."""

    idf: dict[str, float]
    avgdl: float


def fit_bm25(corpus: list[str]) -> BM25Stats:
    tokens = [tokenize(d) for d in corpus]
    if not any(tokens):  # field empty everywhere: every score is 0
        return BM25Stats(idf={}, avgdl=1.0)
    bm = BM25Okapi(tokens, k1=BM25_K1, b=BM25_B)
    return BM25Stats(idf=dict(bm.idf), avgdl=float(bm.avgdl))


def bm25_score(query_tokens: list[str], doc_tf: Counter, doc_len: int, stats: BM25Stats) -> float:
    """Same formula as BM25Okapi.get_scores, but usable for documents outside the
    fitted corpus (validation/test products). Unseen terms contribute 0, as in rank_bm25."""
    norm = BM25_K1 * (1 - BM25_B + BM25_B * doc_len / stats.avgdl)
    s = 0.0
    for t in query_tokens:
        f = doc_tf.get(t, 0)
        if f:
            s += stats.idf.get(t, 0.0) * f * (BM25_K1 + 1) / (f + norm)
    return s


def bm25_column(df: pd.DataFrame, text: pd.Series, stats: BM25Stats) -> np.ndarray:
    """BM25 for every row of df, tokenising each unique query / document once."""
    q_tokens = {q: tokenize(q) for q in df["query"].unique()}
    doc_cache: dict[str, tuple[Counter, int]] = {}
    out = np.empty(len(df))
    for i, (q, d) in enumerate(zip(df["query"].to_numpy(), text.to_numpy())):
        if d not in doc_cache:
            toks = tokenize(d)
            doc_cache[d] = (Counter(toks), len(toks))
        tf, dl = doc_cache[d]
        out[i] = bm25_score(q_tokens[q], tf, dl, stats)
    return out


# ---------------------------------------------------------------- brands
BRAND_STOPLIST = frozenset({"generic", "unbranded", "unknown", "none", "na", "n a", "no brand",
                            "brand", "new", "other", "various", "visit"})
MAX_BRAND_TOKENS = 4


CORPORATE_SUFFIXES = frozenset({"inc", "llc", "ltd", "co", "corp", "corporation", "company",
                                "electronics", "official", "store", "usa", "international",
                                "group"})


def normalize_brand(brand: str) -> str:
    """Lowercase tokens without trailing corporate words: 'Samsung Electronics' -> 'samsung'."""
    toks = tokenize(brand)
    while len(toks) > 1 and toks[-1] in CORPORATE_SUFFIXES:
        toks.pop()
    return " ".join(toks)


def fit_brand_vocab(products: pd.DataFrame, min_products: int = 2) -> frozenset[str]:
    """Brands of at least `min_products` distinct training products, excluding
    placeholders, colours, stopwords, pure numbers and very short strings."""
    p = products.drop_duplicates("product_id")
    counts = p["brand"].map(normalize_brand).value_counts()
    keep = [
        b for b, n in counts.items()
        if n >= min_products and len(b) >= 3 and b not in BRAND_STOPLIST
        and b not in COLOURS and b not in STOPWORDS and not b.replace(" ", "").isdigit()
        and len(b.split()) <= MAX_BRAND_TOKENS
    ]
    return frozenset(keep)


def brands_in_text(text: str, vocab: frozenset[str]) -> set[str]:
    """Known brands appearing as contiguous tokens of the text."""
    toks = tokenize(text)
    found = set()
    for n in range(1, MAX_BRAND_TOKENS + 1):
        for i in range(len(toks) - n + 1):
            gram = " ".join(toks[i:i + n])
            if gram in vocab:
                found.add(gram)
    return found


def contains_phrase(haystack_tokens: list[str], phrase: str) -> bool:
    needle = phrase.split()
    n = len(needle)
    return n > 0 and any(haystack_tokens[i:i + n] == needle
                         for i in range(len(haystack_tokens) - n + 1))


# ---------------------------------------------------------------- fitted state
@dataclass
class FittedState:
    bm25: dict[str, BM25Stats]
    brand_vocab: frozenset[str]


def fit_state(train: pd.DataFrame) -> FittedState:
    """Learn BM25 statistics and the brand vocabulary from the training products only."""
    assert_no_labels(train)
    prod = train.drop_duplicates("product_id")
    texts = {"title": prod["title"], "bullets": prod["bullets"],
             "description": prod["description"], "all_fields": all_fields_text(prod)}
    bm25 = {feat: fit_bm25(texts[field].tolist()) for feat, field in BM25_FIELDS.items()}
    return FittedState(bm25=bm25, brand_vocab=fit_brand_vocab(prod))


def fitted_features(df: pd.DataFrame, state: FittedState) -> pd.DataFrame:
    """Features that need the fitted state, for any set of pairs (train, valid or test)."""
    assert_no_labels(df)
    texts = {"title": df["title"], "bullets": df["bullets"],
             "description": df["description"], "all_fields": all_fields_text(df)}
    out = pd.DataFrame(index=df.index)
    for feat, field in BM25_FIELDS.items():
        out[feat] = bm25_column(df, texts[field], state.bm25[feat])
    q_brands = {q: brands_in_text(q, state.brand_vocab) for q in df["query"].unique()}
    pbrand = df["brand"].map(normalize_brand).to_numpy()
    out["brand_conflict"] = [
        int(bool(q_brands[q]) and b != "" and b not in q_brands[q])
        for q, b in zip(df["query"].to_numpy(), pbrand)
    ]
    out["bm25_all_rank"], out["bm25_all_z"] = within_query(df["query_id"], out["bm25_all"])
    return out


def within_query(query_id: pd.Series, x: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Rank (1 = highest, ties averaged) and z-score of x within each query."""
    g = x.groupby(query_id.to_numpy())
    rank = g.rank(ascending=False, method="average").to_numpy()
    std = g.transform("std").fillna(0).to_numpy()
    z = np.where(std > 0, (x.to_numpy() - g.transform("mean").to_numpy()) / np.where(std > 0, std, 1), 0.0)
    return rank, z


# ---------------------------------------------------------------- static features
def _pair_text_features(query: str, title: str, all_text: str, brand: str, colour: str) -> dict:
    """Lexical, attribute (brand_match, colours), numeric and negation features of one pair."""
    ql, tl = query.lower(), title.lower()
    q_ct, t_ct = set(content_tokens(ql)), set(content_tokens(tl))
    a_ct = set(content_tokens(all_text.lower()))
    q_tok, t_tok = tokenize(ql), tokenize(tl)
    qc, tc = char_ngrams(ql), char_ngrams(tl)
    nb = normalize_brand(brand)
    q_col = colour_tokens(ql)
    p_col = colour_tokens(colour) or colour_tokens(tl)
    f = {
        "jaccard_title": len(q_ct & t_ct) / len(q_ct | t_ct) if q_ct | t_ct else 0.0,
        "coverage_title": len(q_ct & t_ct) / len(q_ct) if q_ct else 0.0,
        "coverage_all": len(q_ct & a_ct) / len(q_ct) if q_ct else 0.0,
        "char3_jaccard_title": len(qc & tc) / len(qc | tc) if qc | tc else 0.0,
        "phrase_in_title": int(bool(q_tok) and contains_phrase(t_tok, " ".join(q_tok))),
        "first_token_in_title": int(bool(q_tok) and q_tok[0] in t_tok),
        "last_token_in_title": int(bool(q_tok) and q_tok[-1] in t_tok),
        "brand_match": int(nb != "" and contains_phrase(q_tok, nb)),
        "colour_match": int(bool(q_col & p_col)),
        "colour_conflict": int(bool(q_col) and bool(p_col) and not (q_col & p_col)),
        "title_len": len(t_tok),
    }
    f.update(number_unit_features(ql, tl))
    f.update(size_features(ql, tl))
    f.update(negation_features(ql, tl))
    return f


def static_features(paths: RunPaths, pairs: pd.DataFrame) -> pd.DataFrame:
    """All features that need no fitting, for every row of pairs."""
    df = feature_input(pairs)
    all_text = all_fields_text(df)
    rows = [
        _pair_text_features(q, t, a, b, c)
        for q, t, a, b, c in zip(df["query"], df["title"], all_text, df["brand"], df["colour"])
    ]
    out = pd.DataFrame(rows, index=df.index)
    out["no_description"] = (df["description"] == "").astype(int)
    out["no_bullets"] = (df["bullets"] == "").astype(int)
    out["no_brand"] = (df["brand"] == "").astype(int)
    out["no_colour"] = (df["colour"] == "").astype(int)
    out["query_len"] = df["query"].map(lambda q: len(tokenize(q)))
    out["query_has_digit"] = df["query"].str.contains(r"\d").astype(int)
    out["n_candidates"] = df.groupby("query_id")["product_id"].transform("size")
    if semantic_available(paths):
        df["title_bullets_text"] = [title_bullets_text(t, b) for t, b in zip(df["title"], df["bullets"])]
        cos = cosine_features(paths, df)
        out["cos_title"], out["cos_title_bullets"] = cos["cos_title"], cos["cos_title_bullets"]
        out["cos_title_rank"], out["cos_title_z"] = within_query(df["query_id"], out["cos_title"])
    out.insert(0, "product_id", df["product_id"])
    out.insert(0, "query_id", df["query_id"])
    assert_no_labels(out)
    return out


def load_or_build_static(paths: RunPaths, pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    """Static features of one split, cached in data/processed/features_static_<split>.parquet.
    Returned rows are aligned with `pairs` (same order)."""
    f = paths.processed / f"features_static_{split}.parquet"
    if f.exists():
        static = pd.read_parquet(f)
    else:
        t0 = time.time()
        static = static_features(paths, pairs)
        static.to_parquet(f, index=False)
        print(f"static features for {split}: {len(static):,} rows in {time.time() - t0:.0f}s")
    key = pairs[["query_id", "product_id"]].reset_index(drop=True)
    static = key.merge(static, on=["query_id", "product_id"], how="left", validate="1:1")
    assert len(static) == len(pairs)
    return static.set_index(pairs.index)


def build_matrix(pairs: pd.DataFrame, static: pd.DataFrame, state: FittedState,
                 features: list[str]) -> pd.DataFrame:
    """Final feature matrix (static + fitted) for `pairs`, columns in `features` order."""
    fitted = fitted_features(feature_input(pairs), state)
    X = pd.concat([static.drop(columns=["query_id", "product_id"]), fitted], axis=1)[features]
    assert_no_labels(X)
    assert list(X.columns) == features
    return X.astype(float)


# ---------------------------------------------------------------- data loading
def load_train(paths: RunPaths) -> pd.DataFrame:
    """Training pairs (sorted by query_id, product_id) with their CV fold number."""
    pairs = pd.read_parquet(paths.processed / "pairs.parquet")
    train = pairs[pairs["split"] == "train"].sort_values(["query_id", "product_id"])
    folds = pd.read_parquet(paths.processed / "folds.parquet")
    train = train.merge(folds, on="query_id", how="left", validate="m:1")
    assert train["fold"].notna().all()
    return train.reset_index(drop=True)


def fold_matrix(paths: RunPaths, train: pd.DataFrame, k: int | None) -> pd.DataFrame:
    """Feature matrix for ALL training rows, with fitted features learned on the
    training part of fold k only (rows whose fold != k). k=None fits on every
    training row (final model). Cached in data/processed/features_fold<k>.parquet."""
    name = "full" if k is None else str(k)
    f = paths.processed / f"features_fold{name}.parquet"
    features = available_features(paths)
    if f.exists():
        X = pd.read_parquet(f)
        assert (X["query_id"].to_numpy() == train["query_id"].to_numpy()).all()
        assert (X["product_id"].to_numpy() == train["product_id"].to_numpy()).all()
        return X[features]
    t0 = time.time()
    static = load_or_build_static(paths, train, "train")
    fit_rows = train if k is None else train[train["fold"] != k]
    state = fit_state(feature_input(fit_rows))
    X = build_matrix(train, static, state, features)
    X.insert(0, "product_id", train["product_id"].to_numpy())
    X.insert(0, "query_id", train["query_id"].to_numpy())
    X.to_parquet(f, index=False)
    print(f"fold {name}: features built in {time.time() - t0:.0f}s")
    return X[features]


# ---------------------------------------------------------------- FEATURES.md
def write_features_md(path) -> None:
    lines = [
        "# Features",
        "",
        "Generated from `FEATURE_DOCS` in `scripts/features.py` (single source of truth).",
        f"{len(FEATURE_NAMES)} features in {len(FEATURE_GROUPS)} groups. "
        "No feature uses relevance labels. *Fitted* features learn statistics "
        "(BM25 IDF / average length, brand vocabulary) from the training part of each CV fold, "
        "or from the full training split for the final model.",
        "",
        "| # | Feature | Group | Formula | Why it might help | Fitted? |",
        "|--:|---|---|---|---|---|",
    ]
    for i, (name, (group, formula, why)) in enumerate(FEATURE_DOCS.items(), 1):
        fitted = "yes" if name in FITTED_FEATURES else "no"
        lines.append(f"| {i} | `{name}` | {group} | {formula} | {why} | {fitted} |")
    lines += ["", "## Groups used in the ablation", ""]
    for g, names in FEATURE_GROUPS.items():
        lines.append(f"- **{g}** ({len(names)}): " + ", ".join(f"`{n}`" for n in names))
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    write_features_md(ROOT / "FEATURES.md")
    train = load_train(paths)
    static = load_or_build_static(paths, train, "train")
    print(static.describe().T.to_string())
    for k in sorted(train["fold"].unique()):  # fitted features, one matrix per CV fold
        fold_matrix(paths, train, int(k))
    log_runtime(paths, "features_static_train", time.time() - t0)


if __name__ == "__main__":
    main()
