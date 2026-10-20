"""Build the website's data files in site/data/ (must stay under 10 MB in total).

site/data/site.json      every number shown on the site, copied from results/*.json
site/data/inspector.json a stratified sample of test queries with both rankings and the
                         chosen model's top-5 feature contributions per product
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from categories import ANALYSIS_ONLY_CATEGORIES, CATEGORY_DESCRIPTIONS, TEXT_CATEGORIES
from config import (BM25_B, BM25_K1, ENCODER_NAME, LGBM_GRID, LR_C_GRID, LR_CLASS_WEIGHTS, SEED,
                    SITE_DATA_DIR, TEXT_CHAR_BUDGET, TITLE_BULLETS_MAX_TOKENS, base_arg_parser,
                    get_paths, log_runtime)
from features import FEATURE_DOCS, FEATURE_LABELS
from metrics import tiebreak_keys

N_INSPECTOR = 1800
PER_CATEGORY = 150
PER_BIN = 25
TITLE_CHARS = 90
MAX_SITE_BYTES = 10 * 1024 * 1024


def r(x, d: int = 4):
    """Round floats recursively (smaller JSON)."""
    if isinstance(x, float):
        return round(x, d)
    if isinstance(x, dict):
        return {k: r(v, d) for k, v in x.items()}
    if isinstance(x, list):
        return [r(v, d) for v in x]
    return x


def load(paths, name: str) -> dict:
    f = paths.results / name
    return json.loads(f.read_text()) if f.exists() else {}


def sample_queries(qinfo: pd.DataFrame, must: list[int], edges: list[float]) -> list[int]:
    """Stratified sample: every category and every ΔnDCG histogram bin is represented."""
    rng = np.random.default_rng(SEED)
    chosen = list(dict.fromkeys(must))
    for c in TEXT_CATEGORIES + ANALYSIS_ONLY_CATEGORIES:
        ids = qinfo.index[qinfo[c]].to_numpy()
        chosen += rng.choice(ids, size=min(PER_CATEGORY, len(ids)), replace=False).tolist()
    bins = np.digitize(qinfo["d"].clip(edges[0], edges[-1] - 1e-9), edges)
    for b in np.unique(bins):
        ids = qinfo.index[bins == b].to_numpy()
        chosen += rng.choice(ids, size=min(PER_BIN, len(ids)), replace=False).tolist()
    for o in ("helps", "hurts", "tie"):
        ids = qinfo.index[qinfo["o"] == o].to_numpy()
        chosen += rng.choice(ids, size=min(PER_BIN, len(ids)), replace=False).tolist()
    chosen = list(dict.fromkeys(chosen))
    rest = np.setdiff1d(qinfo.index.to_numpy(), chosen)
    n_more = max(N_INSPECTOR - len(chosen), 0)
    chosen += rng.choice(rest, size=min(n_more, len(rest)), replace=False).tolist()
    return chosen


def build_inspector(paths, test_res: dict, features: list[str]) -> tuple[dict, dict]:
    pred = pd.read_parquet(paths.results / "test_predictions.parquet")
    pairs = pd.read_parquet(paths.processed / "pairs.parquet", columns=["query_id", "product_id", "query", "title", "brand", "split"])
    pred = pred.merge(pairs[pairs["split"] == "test"].drop(columns="split"), on=["query_id", "product_id"])
    contrib = pd.read_parquet(paths.processed / "test_contributions.parquet")
    cats = pd.read_parquet(paths.processed / "test_categories.parquet")
    from metrics import evaluate_queries
    nb = evaluate_queries(pred, "bm25_title").set_index("query_id")["ndcg"]
    nm = evaluate_queries(pred, "model").set_index("query_id")["ndcg"]
    qinfo = cats.copy()
    qinfo["nb"], qinfo["nm"] = nb, nm
    qinfo = qinfo.dropna(subset=["nb"])
    qinfo["d"] = qinfo["nm"] - qinfo["nb"]
    qinfo["o"] = np.where(qinfo["d"] > 1e-9, "helps", np.where(qinfo["d"] < -1e-9, "hurts", "tie"))
    must = [e["query_id"] for k in ("helps", "hurts") for e in test_res["examples"][k]]
    edges = test_res["delta_hist"]["edges"]
    ids = sample_queries(qinfo, must, edges)

    pred["tb"] = tiebreak_keys(pred)  # same tie-breaking as the metrics
    sub = pred[pred["query_id"].isin(set(ids))].copy()
    for col, rank in (("bm25_title", "rb"), ("model", "rm")):
        sub = sub.sort_values(["query_id", col, "tb"], ascending=[True, False, True])
        sub[rank] = sub.groupby("query_id").cumcount() + 1
    C = contrib.set_index(["query_id", "product_id"]).loc[list(zip(sub["query_id"], sub["product_id"]))][features].to_numpy()
    top = np.argsort(-np.abs(C), axis=1)[:, :5]
    cats_all = TEXT_CATEGORIES + ANALYSIS_ONLY_CATEGORIES
    queries = []
    sub["row"] = np.arange(len(sub))
    for qid, g in sub.groupby("query_id", sort=False):
        q = qinfo.loc[qid]
        prods = []
        for row in g.sort_values("rb").itertuples():
            c = [[int(j), round(float(C[row.row, j]), 3)] for j in top[row.row]]
            prods.append([row.title[:TITLE_CHARS], (row.brand or "")[:30], row.esci_label,
                          int(row.rb), int(row.rm), c])
        queries.append({"id": int(qid), "q": g["query"].iloc[0],
                        "c": [i for i, c in enumerate(cats_all) if q[c]],
                        "d": round(float(q["d"]), 4), "nb": round(float(q["nb"]), 4),
                        "nm": round(float(q["nm"]), 4), "o": q["o"], "p": prods})
    # example query ids per Δ-histogram bin (from the inspector sample, so they can be opened)
    in_sample = qinfo.loc[[x["id"] for x in queries]]
    bins = np.digitize(in_sample["d"].clip(edges[0], edges[-1] - 1e-9), edges) - 1
    bin_examples = {int(b): in_sample.index[bins == b].tolist()[:40] for b in np.unique(bins)}
    inspector = {"features": [FEATURE_LABELS[f] for f in features], "categories": cats_all,
                 "queries": sorted(queries, key=lambda x: x["q"])}
    meta = {"n_queries": len(queries), "bin_examples": bin_examples,
            "n_products": int(len(sub))}
    return inspector, meta


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    out_dir = SITE_DATA_DIR if not paths.is_sample else paths.results / "site_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit, cv, test = load(paths, "audit.json"), load(paths, "cv_results.json"), load(paths, "test_results.json")
    extra, runtime = load(paths, "extra_analyses.json"), load(paths, "runtime.json")
    build = load(paths, "build_stats.json")
    frozen = cv["chosen"]
    features = frozen["features"]

    inspector, meta = build_inspector(paths, test, features)
    site = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "sample_run": paths.is_sample,
        "audit": {k: audit.get(k) for k in ["sizes", "label_mix", "missing", "near_dup", "product_overlap",
                                            "join_coverage", "duplicate_pairs", "query_length",
                                            "query_type_shares", "share_exact_per_query",
                                            "queries_without_relevant", "share_pairs_brand_in_query"]},
        "top_brands": audit.get("top_brands", [])[:10],
        "build": {"products_in_locale": build.get("products_in_locale")},
        "config": {"text_char_budget": TEXT_CHAR_BUDGET, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
                   "encoder": ENCODER_NAME, "title_bullets_max_tokens": TITLE_BULLETS_MAX_TOKENS,
                   "lr_grid": {"C": LR_C_GRID, "class_weight": [str(w) for w in LR_CLASS_WEIGHTS]},
                   "lgbm_grid": LGBM_GRID},
        "features": [{"name": n, "group": g, "formula": f, "why": w, "label": FEATURE_LABELS[n]}
                     for n, (g, f, w) in FEATURE_DOCS.items() if n in features],
        "cv": {"n_train_queries": cv["n_train_queries"], "baselines": cv["baselines"],
               "models": {k: {"config": v["config"], "summary": v["summary"]} for k, v in cv["models"].items()},
               "ablation": [{k: s[k] for k in ("name", "n_features", "summary", "ndcg_ci")} for s in cv["ablation"]],
               "importance": cv["importance"][:15], "lr_coefficients": cv["lr_coefficients"][:10],
               "chosen": {k: frozen.get(k) for k in ("family", "config", "n_rounds", "reason")},
               "cross_encoder": cv.get("cross_encoder")},
        "test": {k: test[k] for k in ["n_test_queries", "n_excluded_zero_idcg", "n_near_dup", "all",
                                      "without_near_dup", "win_tie_loss", "delta_hist", "categories"]},
        "examples": {k: test["examples"][k][:15] for k in ("helps", "hurts")},
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "extra": extra,
        "runtime": runtime,
        "inspector_meta": meta,
    }
    (out_dir / "site.json").write_text(json.dumps(r(site), separators=(",", ":"), default=str))
    (out_dir / "inspector.json").write_text(json.dumps(inspector, separators=(",", ":"), ensure_ascii=False))
    total = sum(f.stat().st_size for f in out_dir.glob("*.json"))
    print(f"site data: {meta['n_queries']} inspector queries, {meta['n_products']} products, "
          f"{total / 1e6:.2f} MB in {out_dir}")
    assert total < MAX_SITE_BYTES, "site data must stay under 10 MB"
    log_runtime(paths, "export_site_data", time.time() - t0)


if __name__ == "__main__":
    main()
