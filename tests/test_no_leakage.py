"""Leakage guards: no label reaches feature code; learned preprocessing uses training rows only."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rank_bm25 import BM25Okapi

import features as F
from config import ROOT
from text_utils import tokenize
from train_cv import fit_logreg


def toy_pairs() -> pd.DataFrame:
    rows = [
        (1, "nike running shoes size 9", "p1", "Nike Air Running Shoe, Size 9, Black", "Nike", "Black", "E"),
        (1, "nike running shoes size 9", "p2", "Adidas Running Shoe Size 10", "adidas", "White", "S"),
        (1, "nike running shoes size 9", "p3", "Shoe laces 2 pack", "Generic", "", "C"),
        (2, "sulfate free shampoo 12 oz", "p4", "Shampoo, Sulfate-Free, 12 oz", "Acme", "", "E"),
        (2, "sulfate free shampoo 12 oz", "p5", "Shampoo with Sulfate, 32 oz", "Acme", "", "I"),
        (3, "red mug", "p6", "Red Ceramic Mug", "Mugco", "Red", "E"),
        (3, "red mug", "p7", "Blue Ceramic Mug zzzunique", "Mugco", "Blue", "S"),
    ]
    df = pd.DataFrame(rows, columns=["query_id", "query", "product_id", "title", "brand", "colour", "esci_label"])
    df["description"] = ""
    df["bullets"] = "durable everyday item"
    df["gain"] = df["esci_label"].map({"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0})
    df["split"] = "train"
    return df


class NoSemantic:
    """RunPaths stand-in without embeddings (semantic features off)."""
    processed = Path("/nonexistent")


@pytest.fixture(autouse=True)
def no_semantic(monkeypatch):
    monkeypatch.setattr(F, "semantic_available", lambda paths: False)


# ---------------------------------------------------------------- labels never reach features
def test_feature_input_drops_labels():
    inp = F.feature_input(toy_pairs())
    for c in F.LABEL_COLUMNS:
        assert c not in inp.columns


def test_assert_no_labels_raises():
    with pytest.raises(AssertionError):
        F.assert_no_labels(toy_pairs())


def test_features_do_not_change_when_labels_change():
    a = toy_pairs()
    b = a.copy()
    b["esci_label"] = b["esci_label"].sample(frac=1, random_state=0).to_numpy()
    b["gain"] = b["gain"].sample(frac=1, random_state=1).to_numpy()
    sa, sb = F.static_features(NoSemantic, a), F.static_features(NoSemantic, b)
    pd.testing.assert_frame_equal(sa, sb)
    state_a, state_b = F.fit_state(F.feature_input(a)), F.fit_state(F.feature_input(b))
    pd.testing.assert_frame_equal(F.fitted_features(F.feature_input(a), state_a),
                                  F.fitted_features(F.feature_input(b), state_b))


def test_feature_matrix_has_only_feature_columns():
    df = toy_pairs()
    static = F.static_features(NoSemantic, df)
    feats = [f for f in F.FEATURE_NAMES if f not in F.SEMANTIC_DEPENDENT]
    X = F.build_matrix(df, static, F.fit_state(F.feature_input(df)), feats)
    assert list(X.columns) == feats
    assert not set(X.columns) & set(F.LABEL_COLUMNS)


# ---------------------------------------------------------------- BM25 fit on training rows only
def test_bm25_score_matches_rank_bm25_on_corpus_docs():
    docs = ["red ceramic mug", "blue ceramic mug large", "nike running shoe", "shampoo sulfate free"]
    stats = F.fit_bm25(docs)
    ref = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75).get_scores(tokenize("ceramic mug red"))
    from collections import Counter
    ours = [F.bm25_score(tokenize("ceramic mug red"), Counter(tokenize(d)), len(tokenize(d)), stats) for d in docs]
    np.testing.assert_allclose(ours, ref)


def test_bm25_statistics_ignore_validation_products():
    df = toy_pairs()
    train, valid = df[df["query_id"] != 3], df[df["query_id"] == 3]
    state = F.fit_state(F.feature_input(train))
    # 'zzzunique' only occurs in a validation title -> unknown to the fitted IDF
    assert "zzzunique" not in state.bm25["bm25_title"].idf
    # fitting on train + valid would change it: the fit really depends on its rows
    state_all = F.fit_state(F.feature_input(df))
    assert state_all.bm25["bm25_title"].idf != state.bm25["bm25_title"].idf
    assert len(F.fitted_features(F.feature_input(valid), state)) == len(valid)


def test_brand_vocab_built_from_training_products_only():
    df = toy_pairs()
    train = df[df["query_id"] != 3]
    vocab = F.fit_brand_vocab(F.feature_input(train), min_products=1)
    assert "mugco" not in vocab  # brand that appears only in validation products
    assert "acme" in vocab and "generic" not in vocab


def test_scaler_is_fit_on_training_rows_only():
    df = toy_pairs()
    static = F.static_features(NoSemantic, df)
    feats = ["coverage_title", "title_len", "jaccard_title"]
    tr = df["query_id"] != 3
    X = F.build_matrix(df, static, F.fit_state(F.feature_input(df[tr])), feats)
    model = fit_logreg(X[tr.to_numpy()], df[tr], C=1.0, class_weight=None)
    np.testing.assert_allclose(model[0].mean_, X[tr.to_numpy()].mean().to_numpy())


# ---------------------------------------------------------------- splits
def test_folds_are_grouped_by_query():
    from build_dataset import make_folds
    df = pd.DataFrame({"query_id": np.repeat(np.arange(50), 4), "split": "train"})
    folds = make_folds(df)
    assert folds["query_id"].is_unique  # each query in exactly one fold
    assert folds["fold"].nunique() == 5


def test_no_pair_level_random_split_in_code():
    code = "\n".join(p.read_text() for p in (ROOT / "scripts").glob("*.py"))
    assert "train_test_split" not in code
    assert not re.search(r"(?<!Group)KFold\(", code)
    assert not re.search(r"(?<!Group)ShuffleSplit\(", code)


def test_feature_code_never_reads_labels():
    src = (ROOT / "scripts" / "features.py").read_text()
    # the only mentions of esci_label in features.py are in the guard list
    uses = [l for l in src.splitlines() if "esci_label" in l]
    assert uses == ['LABEL_COLUMNS = ("esci_label", "gain", "grade", "label", "split")']


# ---------------------------------------------------------------- freeze guard
def test_freeze_check_refuses_untracked_file(tmp_path):
    from evaluate_test import freeze_check
    ok, msg = freeze_check("results/this_file_does_not_exist.md")
    assert not ok and "not tracked" in msg
