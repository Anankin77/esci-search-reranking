"""Phase 2 baselines, evaluated per CV fold on the training split only.

1. random ordering (mean over 20 seeds)      - the floor
2. BM25 on title                              - main baseline
3. BM25 on title + brand + colour + bullets   - stronger lexical baseline
4. encoder cosine (query vs title), zero-shot - semantic baseline, no learning

BM25 IDF/avgdl come from the training part of each fold (via features.fold_matrix).
Writes results/cv_baselines.json and data/processed/oof_baselines.parquet.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from config import (BM25_B, BM25_K1, N_RANDOM_SEEDS, SEED, base_arg_parser, get_paths,
                    log_runtime)
from embeddings import semantic_available
from features import fold_matrix, load_train
from metrics import METRICS, evaluate_queries, summarize

BASELINE_COLUMNS = {"bm25_title": "bm25_title", "bm25_all": "bm25_all", "encoder_cosine": "cos_title"}


def random_scores(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def fold_summary(per_fold: list[dict]) -> dict:
    """Mean and std across folds of each metric."""
    out = {}
    for m in METRICS:
        vals = np.array([f[m] for f in per_fold])
        out[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                  "folds": vals.round(5).tolist()}
    out["n_excluded_zero_idcg"] = int(sum(f["n_excluded_zero_idcg"] for f in per_fold))
    return out


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    train = load_train(paths)
    methods = ["random", "bm25_title", "bm25_all"]
    if semantic_available(paths):
        methods.append("encoder_cosine")
    per_fold: dict[str, list[dict]] = {m: [] for m in methods}
    oof = train[["query_id", "product_id", "fold", "gain"]].copy()

    for k in sorted(train["fold"].unique()):
        X = fold_matrix(paths, train, int(k))
        valid = train["fold"] == k
        vdf = train.loc[valid, ["query_id", "product_id", "gain"]].copy()
        # random: average the metrics of 20 seeded random orderings
        seed_res = []
        for s in range(N_RANDOM_SEEDS):
            vdf["score"] = random_scores(len(vdf), SEED + 1000 * int(k) + s)
            seed_res.append(summarize(evaluate_queries(vdf, "score")))
        per_fold["random"].append({m: float(np.mean([r[m] for r in seed_res])) for m in METRICS}
                                  | {"n_excluded_zero_idcg": seed_res[0]["n_excluded_zero_idcg"]})
        for method in methods[1:]:
            vdf["score"] = X.loc[valid, BASELINE_COLUMNS[method]].to_numpy()
            oof.loc[valid, method] = vdf["score"].to_numpy()
            per_fold[method].append(summarize(evaluate_queries(vdf, "score")))
        print(f"fold {k}: " + ", ".join(f"{m} {per_fold[m][-1]['ndcg']:.4f}" for m in methods))

    results = {
        "bm25_params": {"k1": BM25_K1, "b": BM25_B, "library": "rank_bm25.BM25Okapi"},
        "methods": {m: fold_summary(per_fold[m]) for m in methods},
    }
    (paths.results / "cv_baselines.json").write_text(json.dumps(results, indent=2))
    oof.to_parquet(paths.processed / "oof_baselines.parquet", index=False)
    for m in methods:
        r = results["methods"][m]
        print(f"{m:16s} nDCG {r['ndcg']['mean']:.4f} ± {r['ndcg']['std']:.4f}  "
              f"nDCG@5 {r['ndcg5']['mean']:.4f}  P@1 {r['p1']['mean']:.4f}")
    log_runtime(paths, "baselines", time.time() - t0)


if __name__ == "__main__":
    main()
