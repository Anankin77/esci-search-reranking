"""Phase 1 data audit -> results/report_1_data_audit.md, results/audit.json, figures.

Also computes the two cross-split integrity checks of spec §4:
- share of test pairs whose product also appears in training (rule 4.2)
- near-duplicate test queries (rule 4.3), saved to data/processed/near_dup_test_queries.parquet

The audit only describes the data; no modelling decision is taken from the test split.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import NEAR_DUP_COSINE, RAW_DIR, SEED, base_arg_parser, get_paths, log_runtime
from features import brands_in_text, fit_brand_vocab, normalize_brand
from text_utils import colour_tokens, has_negation_cue, normalize_query, tokenize

FIELDS = ["title", "description", "bullets", "brand", "colour"]
LABEL_ORDER = ["E", "S", "C", "I"]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Tiny DataFrame -> Markdown table (no extra dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def near_duplicates(train_q: pd.Series, test_q: pd.DataFrame) -> pd.DataFrame:
    """Flag test queries that equal a training query after normalisation, or whose
    char-3-gram TF-IDF cosine with some training query is >= 0.90.
    The TF-IDF vocabulary/IDF is fit on training queries only."""
    train_norm = train_q.map(normalize_query).unique()
    test_norm = test_q["query"].map(normalize_query)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
    X_train = vec.fit_transform(train_norm)
    X_test = vec.transform(test_norm)
    max_cos = np.zeros(X_test.shape[0])
    for start in range(0, X_test.shape[0], 500):  # chunks keep memory small
        sims = (X_test[start:start + 500] @ X_train.T).toarray()
        max_cos[start:start + 500] = sims.max(axis=1)
    out = test_q[["query_id", "query"]].copy()
    out["exact_dup"] = test_norm.isin(set(train_norm)).to_numpy()
    out["max_cosine"] = max_cos
    out["near_dup"] = out["exact_dup"] | (out["max_cosine"] >= NEAR_DUP_COSINE)
    return out


def save_hist(values, bins, title, xlabel, path, logy=False) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(values, bins=bins, color="#4a6fa5", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    pairs = pd.read_parquet(paths.processed / "pairs.parquet")
    build_stats = json.loads((paths.results / "build_stats.json").read_text())
    train, test = pairs[pairs["split"] == "train"], pairs[pairs["split"] == "test"]
    audit: dict = {}
    md: list[str] = ["# Report 1 — Data audit", "",
                     "Amazon Shopping Queries Dataset (ESCI), small version, US locale "
                     "(Reddy et al., 2022, Apache-2.0).", ""]

    # ---- sizes and candidates per query
    sizes = []
    for name, part in (("train", train), ("test", test), ("total", pairs)):
        cpq = part.groupby("query_id").size()
        sizes.append({"split": name, "queries": part["query_id"].nunique(), "pairs": len(part),
                      "products": part["product_id"].nunique(),
                      "cand_min": int(cpq.min()), "cand_median": float(cpq.median()),
                      "cand_max": int(cpq.max())})
    sizes_df = pd.DataFrame(sizes)
    audit["sizes"] = sizes
    cpq_all = pairs.groupby("query_id").size()
    save_hist(cpq_all, bins=range(0, int(cpq_all.max()) + 2), title="Candidates per query",
              xlabel="candidates", path=paths.figures / "candidates_per_query.png")
    audit["candidates_hist"] = cpq_all.value_counts().sort_index().to_dict()
    md += ["## 1. Size", "", md_table(sizes_df, "{:.1f}"), "",
           f"Expected sizes from spec §2.1 matched: `{json.dumps(build_stats['sizes']) if isinstance(build_stats['sizes'], str) else all(v['within_1pct'] for v in build_stats['sizes'].values())}`.",
           "", "![candidates](figures/candidates_per_query.png)", ""]

    # ---- labels
    mix = (pairs.groupby("split")["esci_label"].value_counts(normalize=True).unstack()
           .reindex(columns=LABEL_ORDER))
    mix.loc["total"] = pairs["esci_label"].value_counts(normalize=True).reindex(LABEL_ORDER)
    audit["label_mix"] = mix.round(4).to_dict(orient="index")
    share_e = pairs.groupby(["split", "query_id"])["esci_label"].apply(lambda s: (s == "E").mean())
    zero_idcg = pairs.groupby("query_id")["esci_label"].apply(lambda s: s.isin(["E", "S", "C"]).any())
    audit["queries_without_relevant"] = int((~zero_idcg).sum())
    audit["share_exact_per_query"] = {
        s: {"mean": float(share_e[s].mean()), "median": float(share_e[s].median()),
            "share_all_exact": float((share_e[s] == 1).mean()),
            "share_no_exact": float((share_e[s] == 0).mean()),
            "n_all_exact": int((share_e[s] == 1).sum()),
            "n_no_exact": int((share_e[s] == 0).sum())} for s in ("train", "test")}
    save_hist(share_e, bins=20, title="Share of Exact candidates per query",
              xlabel="share of E", path=paths.figures / "share_exact_per_query.png")
    mix_df = mix.reset_index().rename(columns={"index": "split"})
    md += ["## 2. Labels", "", "Share of pairs per label:", "", md_table(mix_df), "",
           "Share of Exact per query: " + "; ".join(
               f"{s}: mean {v['mean']:.3f}, median {v['median']:.3f}, queries with all candidates Exact: "
               f"{v['n_all_exact']}, with no Exact: {v['n_no_exact']}" for s, v in audit["share_exact_per_query"].items()),
           "", f"Queries with no E/S/C candidate (ideal DCG = 0, excluded from metric averages): "
           f"**{audit['queries_without_relevant']}**.", "",
           "![share exact](figures/share_exact_per_query.png)", "",
           "Most candidates are Exact, so even a random ordering puts relevant products near the top: "
           "random nDCG will be high and gains must be read relative to random and BM25.", ""]

    # ---- missing fields
    miss = pd.DataFrame({s: [(part[f] == "").mean() for f in FIELDS] for s, part in
                         (("train", train), ("test", test))}, index=FIELDS)
    audit["missing"] = miss.round(4).to_dict()
    md += ["## 3. Missing fields (empty after cleaning)", "",
           md_table(miss.reset_index().rename(columns={"index": "field"}), "{:.1%}"), ""]

    # ---- brands
    prod = pairs.drop_duplicates("product_id")
    top_brands = prod.loc[prod["brand"] != "", "brand"].value_counts().head(20)
    nb = pairs["brand"].map(normalize_brand)
    qtok = pairs["query"].map(lambda q: f" {' '.join(tokenize(q))} ")
    in_query = [bool(b) and f" {b} " in q for b, q in zip(nb, qtok)]
    pairs_bq = pd.DataFrame({"brand": pairs["brand"], "in_query": in_query})
    brand_rows = []
    for b, n in top_brands.items():
        sub = pairs_bq[pairs_bq["brand"] == b]
        brand_rows.append({"brand": b, "products": int(n), "pairs": len(sub),
                           "share_pairs_brand_in_query": float(sub["in_query"].mean())})
    audit["top_brands"] = brand_rows
    audit["share_pairs_brand_in_query"] = float(np.mean(in_query))
    md += ["## 4. Brands", "",
           f"Across all pairs, the product's brand appears in the query text for "
           f"**{np.mean(in_query):.1%}** of pairs.", "", md_table(pd.DataFrame(brand_rows)), ""]

    # ---- queries: length and examples
    queries = pairs.drop_duplicates("query_id")[["query_id", "query", "split"]]
    qlen = queries["query"].map(lambda q: len(tokenize(q)))
    save_hist(qlen, bins=range(0, int(qlen.max()) + 2), title="Query length (tokens)",
              xlabel="tokens", path=paths.figures / "query_length.png")
    audit["query_length"] = {"mean": float(qlen.mean()), "median": float(qlen.median()),
                             "max": int(qlen.max()), "hist": qlen.value_counts().sort_index().to_dict()}
    vocab = fit_brand_vocab(train)  # brand vocabulary from training products only
    rng = np.random.default_rng(SEED)
    tq = queries[queries["split"] == "train"]

    def examples(mask) -> list[str]:
        sub = tq[mask.to_numpy()]
        return sub["query"].iloc[rng.choice(len(sub), size=min(8, len(sub)), replace=False)].tolist() if len(sub) else []

    ex_numeric = examples(tq["query"].str.contains(r"\d"))
    ex_neg = examples(tq["query"].map(has_negation_cue))
    ex_brand = examples(tq["query"].map(lambda q: bool(brands_in_text(q, vocab))))
    ex_colour = examples(tq["query"].map(lambda q: bool(colour_tokens(q))))
    audit["query_examples"] = {"numeric": ex_numeric, "negation": ex_neg, "brand": ex_brand,
                               "colour": ex_colour}
    shares = {"numeric": float(queries["query"].str.contains(r"\d").mean()),
              "negation": float(queries["query"].map(has_negation_cue).mean())}
    audit["query_type_shares"] = shares
    md += ["## 5. Queries", "",
           f"Query length: mean {qlen.mean():.2f} tokens, median {qlen.median():.0f}, max {qlen.max()}. "
           f"{shares['numeric']:.1%} of queries contain a digit; {shares['negation']:.1%} contain a negation cue.",
           "", "![query length](figures/query_length.png)", "",
           "Examples (training queries, random):", "",
           "- numeric: " + " · ".join(f"`{q}`" for q in ex_numeric),
           "- negation: " + " · ".join(f"`{q}`" for q in ex_neg),
           "- brand: " + " · ".join(f"`{q}`" for q in ex_brand),
           "- colour: " + " · ".join(f"`{q}`" for q in ex_colour), ""]
    sources_file = RAW_DIR / "shopping_queries_dataset_sources.csv"
    if sources_file.exists():
        src = pd.read_csv(sources_file)
        src_mix = queries.merge(src, on="query_id", how="left")["source"].fillna("missing").value_counts()
        audit["query_sources"] = src_mix.to_dict()
        md += ["Query sources (from `sources.csv`; metadata only, not used as a feature): " +
               ", ".join(f"{k} {v:,}" for k, v in src_mix.items()) + ".", ""]

    # ---- join, duplicates
    md += ["## 6. Join and duplicates", "",
           f"- `examples` joined to `products` on (`product_locale`, `product_id`): "
           f"coverage **{build_stats['join_coverage']:.2%}** "
           f"({build_stats['pairs_without_product']} pairs without a product row).",
           f"- Duplicate (query_id, product_id) pairs: **{build_stats['duplicate_pairs_dropped']}** "
           f"(would be dropped keeping the first).", ""]
    audit["join_coverage"] = build_stats["join_coverage"]
    audit["duplicate_pairs"] = build_stats["duplicate_pairs_dropped"]

    # ---- cross-split integrity
    train_products = set(train["product_id"])
    shared = test["product_id"].isin(train_products)
    nd = near_duplicates(train["query"], test.drop_duplicates("query_id"))
    nd.to_parquet(paths.processed / "near_dup_test_queries.parquet", index=False)
    audit["product_overlap"] = {"share_test_pairs_product_in_train": float(shared.mean()),
                                "share_test_products_in_train":
                                    float(pd.Series(test["product_id"].unique()).isin(train_products).mean())}
    audit["near_dup"] = {"test_queries": int(len(nd)), "exact_dup": int(nd["exact_dup"].sum()),
                         "cosine_ge_090": int((nd["max_cosine"] >= NEAR_DUP_COSINE).sum()),
                         "flagged": int(nd["near_dup"].sum()),
                         "share_flagged": float(nd["near_dup"].mean())}
    ex_nd = nd[nd["near_dup"]].head(8)
    md += ["## 7. Cross-split integrity (spec §4)", "",
           f"- **Shared products (rule 4.2):** {shared.mean():.1%} of test pairs have a product that also "
           f"appears in training ({audit['product_overlap']['share_test_products_in_train']:.1%} of unique test products). "
           "Allowed (shared catalog); no feature uses labels, so no label can leak through them.",
           f"- **Near-duplicate test queries (rule 4.3):** {audit['near_dup']['exact_dup']} exact matches after "
           f"normalisation and {audit['near_dup']['cosine_ge_090']} with char-3-gram TF-IDF cosine ≥ 0.90 "
           f"(TF-IDF fit on training queries); **{audit['near_dup']['flagged']}** flagged in total "
           f"({audit['near_dup']['share_flagged']:.1%} of test queries). Final results are reported with and without them.",
           "", "Examples of flagged test queries: " + " · ".join(
               f"`{q}` (cos {c:.2f})" for q, c in zip(ex_nd["query"], ex_nd["max_cosine"])), ""]

    # ---- sanity examples (training queries)
    md += ["## 8. Ten example queries (training split)", ""]
    ex_q = rng.choice(tq["query_id"].to_numpy(), size=min(10, len(tq)), replace=False)
    sanity = []
    for qid in ex_q:
        g = train[train["query_id"] == qid]
        md += [f"**{g['query'].iloc[0]}** ({len(g)} candidates)", ""]
        for _, r in g.head(8).iterrows():
            md.append(f"- `{r['esci_label']}` {r['title'][:90]} — *{r['brand'] or 'no brand'}*")
        if len(g) > 8:
            md.append(f"- … {len(g) - 8} more")
        md.append("")
        sanity.append({"query": g["query"].iloc[0],
                       "labels": g["esci_label"].value_counts().to_dict()})
    audit["sanity_examples"] = sanity
    md += ["`notebooks/01_eda.ipynb` reproduces the main figures.", ""]

    (paths.results / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (paths.results / "report_1_data_audit.md").write_text("\n".join(md))
    print("\n".join(md[:40]))
    log_runtime(paths, "audit", time.time() - t0)


if __name__ == "__main__":
    main()
