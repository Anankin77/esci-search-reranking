"""Build data/processed/pairs.parquet: one row per (query, product) of the small US subset,
with joined, cleaned product fields.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import GroupKFold

from config import (EXPECTED_SIZES, GAINS, LOCALE, N_FOLDS, RAW_DIR, SEED, base_arg_parser,
                    get_paths, log_runtime)
from text_utils import clean_long_text, clean_text

EXAMPLES = RAW_DIR / "shopping_queries_dataset_examples.parquet"
PRODUCTS = RAW_DIR / "shopping_queries_dataset_products.parquet"


def load_examples() -> pd.DataFrame:
    ex = pd.read_parquet(EXAMPLES)
    return ex[(ex["small_version"] == 1) & (ex["product_locale"] == LOCALE)].copy()


def load_products(product_ids: set[str]) -> pd.DataFrame:
    table = pq.read_table(PRODUCTS, filters=[("product_locale", "=", LOCALE)])
    products = table.to_pandas()
    return products[products["product_id"].isin(product_ids)]


def check_sizes(pairs: pd.DataFrame) -> dict:
    """Compare query/pair counts with spec §2.1; warn if off by more than 1 %."""
    report = {}
    for split in ("total", "train", "test"):
        part = pairs if split == "total" else pairs[pairs["split"] == split]
        got = {"queries": int(part["query_id"].nunique()), "pairs": int(len(part))}
        exp = EXPECTED_SIZES[split]
        ok = all(abs(got[k] - exp[k]) / exp[k] <= 0.01 for k in exp)
        if not ok:
            print(f"WARNING: {split} sizes {got} differ from expected {exp} by > 1%")
        report[split] = {"got": got, "expected": exp, "within_1pct": ok}
    return report


def sample_queries(pairs: pd.DataFrame, n_train: int) -> pd.DataFrame:
    """Smoke-test subset: n_train training queries and n_train // 2 test queries."""
    rng = np.random.default_rng(SEED)
    keep = []
    for split, n in (("train", n_train), ("test", max(n_train // 2, 1))):
        qids = np.sort(pairs.loc[pairs["split"] == split, "query_id"].unique())
        keep.extend(rng.choice(qids, size=min(n, len(qids)), replace=False))
    return pairs[pairs["query_id"].isin(set(keep))]


def make_folds(pairs: pd.DataFrame) -> pd.DataFrame:
    """Assign every training query to one of N_FOLDS folds (grouped by query_id, seeded).
    Saved once so every baseline and model uses identical folds."""
    train = pairs[pairs["split"] == "train"]
    qids = np.sort(train["query_id"].unique())
    gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.empty(len(qids), dtype=int)
    for k, (_, valid_idx) in enumerate(gkf.split(qids, groups=qids)):
        fold[valid_idx] = k
    return pd.DataFrame({"query_id": qids, "fold": fold})


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()

    ex = load_examples()
    stats: dict = {"n_examples_small_us": int(len(ex))}

    # Duplicate (query_id, product_id) pairs: keep the first, report how many.
    dup = ex.duplicated(["query_id", "product_id"], keep="first")
    stats["duplicate_pairs_dropped"] = int(dup.sum())
    ex = ex[~dup]

    products = load_products(set(ex["product_id"]))
    stats["products_in_locale"] = int(len(products))
    products = products.drop_duplicates(["product_locale", "product_id"])
    pairs = ex.merge(products, on=["product_locale", "product_id"], how="left", indicator=True)
    stats["join_coverage"] = float((pairs["_merge"] == "both").mean())
    stats["pairs_without_product"] = int((pairs["_merge"] != "both").sum())
    pairs = pairs.drop(columns="_merge")

    if args.sample:
        pairs = sample_queries(pairs, args.sample)

    print(f"cleaning text for {len(pairs):,} pairs ...")
    # Clean each unique product once, then map back (many products repeat).
    prod = pairs.drop_duplicates("product_id")[
        ["product_id", "product_title", "product_description", "product_bullet_point",
         "product_brand", "product_color"]
    ].copy()
    prod["title"] = prod["product_title"].map(clean_text)
    prod["description"] = prod["product_description"].map(clean_long_text)
    prod["bullets"] = prod["product_bullet_point"].map(clean_long_text)
    prod["brand"] = prod["product_brand"].map(clean_text)
    prod["colour"] = prod["product_color"].map(clean_text)
    prod = prod[["product_id", "title", "description", "bullets", "brand", "colour"]]

    out = pairs[["query_id", "query", "product_id", "split", "esci_label"]].merge(
        prod, on="product_id", how="left"
    )
    out["query"] = out["query"].map(clean_text)
    out["gain"] = out["esci_label"].map(GAINS).astype(float)
    out = out.sort_values(["query_id", "product_id"]).reset_index(drop=True)

    stats["sizes"] = check_sizes(out) if not args.sample else "sample run"
    out.to_parquet(paths.processed / "pairs.parquet", index=False)
    folds = make_folds(out)
    folds.to_parquet(paths.processed / "folds.parquet", index=False)
    stats["fold_sizes_queries"] = folds["fold"].value_counts().sort_index().tolist()
    (paths.results / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    log_runtime(paths, "build_dataset", time.time() - t0)


if __name__ == "__main__":
    main()
