"""Ranking metrics: nDCG (full list), nDCG@5, Exact-P@1, and paired bootstrap CIs.

Conventions (spec §5, Phase 2):
- DCG = sum_i gain_i / log2(rank_i + 1), ranks start at 1.
- The ideal DCG uses the same candidate set sorted by gain.
- Queries whose ideal DCG is 0 (no E/S/C candidate) are excluded from averages.
- Score ties are broken with a seeded random key, never by row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import GAINS, N_BOOTSTRAP, SEED

EXACT_GAIN = GAINS["E"]
METRICS = ("ndcg", "ndcg5", "p1")
METRIC_NAMES = {"ndcg": "nDCG", "ndcg5": "nDCG@5", "p1": "Exact-P@1"}


# ---------------------------------------------------------------- single list
def dcg(gains_in_ranked_order, k: int | None = None) -> float:
    g = np.asarray(gains_in_ranked_order, dtype=float)[:k]
    return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))


def ndcg(gains_in_ranked_order, k: int | None = None) -> float:
    """nDCG of one ranked list. Returns NaN if the ideal DCG is 0."""
    ideal = dcg(np.sort(np.asarray(gains_in_ranked_order, dtype=float))[::-1], k)
    if ideal == 0:
        return float("nan")
    return dcg(gains_in_ranked_order, k) / ideal


def ndcg_at_5(gains_in_ranked_order) -> float:
    return ndcg(gains_in_ranked_order, k=5)


def exact_precision_at_1(gains_in_ranked_order) -> float:
    """1.0 if the top-ranked product is Exact, else 0.0."""
    return float(np.asarray(gains_in_ranked_order, dtype=float)[0] == EXACT_GAIN)


def rank_order(scores, tiebreak) -> np.ndarray:
    """Indices that sort by score (descending), ties broken by the random key."""
    return np.lexsort((np.asarray(tiebreak), -np.asarray(scores, dtype=float)))


# ---------------------------------------------------------------- many queries
def tiebreak_keys(df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """One seeded random number per (query_id, product_id), independent of row order."""
    order = np.lexsort((df["product_id"].to_numpy(), df["query_id"].to_numpy()))
    keys = np.empty(len(df))
    keys[order] = np.random.default_rng(seed).random(len(df))
    return keys


def evaluate_queries(df: pd.DataFrame, score_col: str, seed: int = SEED) -> pd.DataFrame:
    """Per-query nDCG, nDCG@5 and Exact-P@1 for the ranking given by df[score_col].

    df needs columns query_id, product_id, gain and score_col.
    Returns one row per query; metrics are NaN for queries with ideal DCG 0.
    """
    d = pd.DataFrame({
        "query_id": df["query_id"].to_numpy(),
        "gain": df["gain"].to_numpy(dtype=float),
        "score": df[score_col].to_numpy(dtype=float),
        "tb": tiebreak_keys(df, seed),
    })
    # model ranking
    d = d.sort_values(["query_id", "score", "tb"], ascending=[True, False, True])
    rank = d.groupby("query_id").cumcount().to_numpy() + 1
    disc = 1.0 / np.log2(rank + 1)
    d["dcg"] = d["gain"] * disc
    d["dcg5"] = d["dcg"] * (rank <= 5)
    d["top_exact"] = (rank == 1) & (d["gain"] == EXACT_GAIN)
    # ideal ranking of the same candidates
    ideal_gain = d.sort_values(["query_id", "gain"], ascending=[True, False])["gain"].to_numpy()
    q_sorted = np.sort(d["query_id"].to_numpy(), kind="stable")
    ideal_rank = pd.Series(q_sorted).groupby(q_sorted).cumcount().to_numpy() + 1
    ideal = pd.DataFrame({"query_id": q_sorted,
                          "idcg": ideal_gain / np.log2(ideal_rank + 1)})
    ideal["idcg5"] = ideal["idcg"] * (ideal_rank <= 5)

    g = d.groupby("query_id")[["dcg", "dcg5", "top_exact"]].sum()
    gi = ideal.groupby("query_id")[["idcg", "idcg5"]].sum()
    out = g.join(gi)
    valid = out["idcg"] > 0
    res = pd.DataFrame(index=out.index)
    res["ndcg"] = np.where(valid, out["dcg"] / out["idcg"].where(valid, 1), np.nan)
    res["ndcg5"] = np.where(valid, out["dcg5"] / out["idcg5"].where(valid, 1), np.nan)
    res["p1"] = np.where(valid, out["top_exact"].astype(float), np.nan)
    return res.reset_index()


def summarize(per_query: pd.DataFrame) -> dict[str, float]:
    """Mean of each metric over queries with a non-zero ideal DCG."""
    valid = per_query["ndcg"].notna()
    out = {m: float(per_query.loc[valid, m].mean()) for m in METRICS}
    out["n_queries"] = int(valid.sum())
    out["n_excluded_zero_idcg"] = int((~valid).sum())
    return out


# ---------------------------------------------------------------- bootstrap
def bootstrap_indices(n: int, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> np.ndarray:
    """(n_boot, n) matrix of resampled query positions. Reuse it for every method so
    that confidence intervals of differences are paired."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    """Mean and percentile CI of per-query values (or per-query differences)."""
    values = np.asarray(values, dtype=float)
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}
