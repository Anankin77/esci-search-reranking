"""Metrics against hand-computed examples."""
import math

import numpy as np
import pandas as pd
import pytest

from metrics import (bootstrap_ci, bootstrap_indices, dcg, evaluate_queries,
                     exact_precision_at_1, ndcg, ndcg_at_5, rank_order)


def test_dcg_hand_computed():
    # 1/log2(2) + 0.1/log2(3) + 0/log2(4) = 1 + 0.06309
    assert dcg([1.0, 0.1, 0.0]) == pytest.approx(1 + 0.1 / math.log2(3))


def test_ndcg_perfect_and_reversed():
    assert ndcg([1.0, 0.1, 0.01, 0.0]) == pytest.approx(1.0)
    # reversed: DCG = 0/1 + 0.01/log2(3) + 0.1/2 + 1/log2(5)
    rev = 0.01 / math.log2(3) + 0.1 / 2 + 1 / math.log2(5)
    ideal = 1 + 0.1 / math.log2(3) + 0.01 / 2
    assert ndcg([0.0, 0.01, 0.1, 1.0]) == pytest.approx(rev / ideal)


def test_ndcg_zero_ideal_is_nan():
    assert math.isnan(ndcg([0.0, 0.0, 0.0]))


def test_ndcg_at_k_uses_ideal_of_full_candidate_set():
    # top-2 cut-off: ranked [0, 1, 1] -> DCG@2 = 1/log2(3); ideal@2 = 1 + 1/log2(3)
    assert ndcg([0.0, 1.0, 1.0], k=2) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_ndcg_at_5_ignores_positions_after_5():
    assert ndcg_at_5([1, 1, 1, 1, 1, 0, 0]) == pytest.approx(1.0)
    assert ndcg_at_5([1, 1, 1, 1, 1, 1, 0]) == pytest.approx(1.0)


def test_exact_p_at_1():
    assert exact_precision_at_1([1.0, 0.0]) == 1.0
    assert exact_precision_at_1([0.1, 1.0]) == 0.0


def test_rank_order_breaks_ties_with_key_not_row_order():
    scores = [0.5, 0.5, 0.9]
    assert list(rank_order(scores, [0.7, 0.2, 0.5])) == [2, 1, 0]
    assert list(rank_order(scores, [0.1, 0.2, 0.5])) == [2, 0, 1]


def _toy_frame():
    return pd.DataFrame({
        "query_id": [1, 1, 1, 2, 2, 3, 3],
        "product_id": ["a", "b", "c", "a", "d", "e", "f"],
        "gain": [0.0, 1.0, 0.1, 0.01, 1.0, 0.0, 0.0],
        "score": [3.0, 2.0, 1.0, 5.0, 1.0, 1.0, 2.0],
    })


def test_evaluate_queries_matches_single_list_function():
    df = _toy_frame()
    res = evaluate_queries(df, "score").set_index("query_id")
    assert res.loc[1, "ndcg"] == pytest.approx(ndcg([0.0, 1.0, 0.1]))
    assert res.loc[2, "ndcg"] == pytest.approx(ndcg([0.01, 1.0]))
    assert res.loc[1, "p1"] == 0.0
    assert math.isnan(res.loc[3, "ndcg"])  # no relevant product: excluded


def test_evaluate_queries_is_independent_of_row_order():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "query_id": np.repeat(np.arange(20), 20),
        "product_id": [f"p{i}" for i in range(n)],
        "gain": rng.choice([0, 0.01, 0.1, 1.0], n),
        "score": rng.integers(0, 3, n).astype(float),  # many ties
    })
    a = evaluate_queries(df, "score")
    b = evaluate_queries(df.sample(frac=1, random_state=1), "score")
    pd.testing.assert_frame_equal(a, b)


def test_vectorised_agrees_with_reference_on_random_data():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "query_id": np.repeat(np.arange(30), 12),
        "product_id": [f"p{i}" for i in range(360)],
        "gain": rng.choice([0, 0.01, 0.1, 1.0], 360),
        "score": rng.random(360),
    })
    res = evaluate_queries(df, "score").set_index("query_id")
    for qid, g in df.groupby("query_id"):
        ranked = g.sort_values("score", ascending=False)["gain"].to_numpy()
        ref = ndcg(ranked)
        if math.isnan(ref):
            assert math.isnan(res.loc[qid, "ndcg"])
        else:
            assert res.loc[qid, "ndcg"] == pytest.approx(ref)
            assert res.loc[qid, "ndcg5"] == pytest.approx(ndcg_at_5(ranked))


def test_paired_bootstrap_of_identical_methods_is_zero():
    idx = bootstrap_indices(50, n_boot=200, seed=0)
    ci = bootstrap_ci(np.zeros(50), idx)
    assert ci["lo"] == ci["hi"] == 0.0
