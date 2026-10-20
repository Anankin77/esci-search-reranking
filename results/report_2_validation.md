# Report 2 — Validation (5-fold grouped CV on the training split)

All numbers: mean ± std across the 5 `GroupKFold` folds (grouped by `query_id`, 20,888 training queries). The test split was not used. Queries with ideal DCG 0 excluded: 0.


## 1. Baselines and models

| Type | Method | nDCG | nDCG@5 | Exact-P@1 | ΔnDCG vs random | ΔnDCG vs BM25-title |
|---|---|---|---|---|---|---|
| baseline | Random (20 seeds) | 0.7440 ± 0.0019 | 0.4976 ± 0.0019 | 0.4286 ± 0.0030 | +0.0000 | -0.0700 |
| baseline | BM25 title | 0.8140 ± 0.0025 | 0.6312 ± 0.0050 | 0.6115 ± 0.0093 | +0.0700 | +0.0000 |
| baseline | BM25 all fields | 0.8160 ± 0.0021 | 0.6349 ± 0.0033 | 0.6157 ± 0.0055 | +0.0720 | +0.0019 |
| baseline | Encoder cosine (zero-shot) | 0.8254 ± 0.0021 | 0.6533 ± 0.0033 | 0.6371 ± 0.0058 | +0.0815 | +0.0114 |
| model | Logistic regression | 0.8468 ± 0.0019 | 0.6976 ± 0.0029 | 0.6942 ± 0.0079 | +0.1028 | +0.0328 |
| model | LightGBM pointwise | 0.8506 ± 0.0015 | 0.7050 ± 0.0023 | 0.7094 ± 0.0039 | +0.1066 | +0.0366 |
| model | LightGBM LambdaRank | 0.8508 ± 0.0013 | 0.7049 ± 0.0022 | 0.7095 ± 0.0048 | +0.1069 | +0.0368 |

Gains over random and BM25 are differences of fold means (nDCG points on a 0–1 scale). Random already scores high because Amazon pre-filtered the candidates (≈44 % of them are Exact).

## 2. Hyper-parameter grids (mean CV nDCG)

**Logistic regression**

- `{"C": 0.01, "class_weight": null}`: 0.8468 ± 0.0019
- `{"C": 0.01, "class_weight": "balanced"}`: 0.8467 ± 0.0019
- `{"C": 0.1, "class_weight": null}`: 0.8468 ± 0.0018
- `{"C": 0.1, "class_weight": "balanced"}`: 0.8467 ± 0.0019
- `{"C": 1.0, "class_weight": null}`: 0.8468 ± 0.0019
- `{"C": 1.0, "class_weight": "balanced"}`: 0.8467 ± 0.0019
- `{"C": 10.0, "class_weight": null}`: 0.8468 ± 0.0019
- `{"C": 10.0, "class_weight": "balanced"}`: 0.8467 ± 0.0019

**LightGBM pointwise**

- `{"num_leaves": 15, "min_data_in_leaf": 20}`: 0.8504 ± 0.0016, rounds per fold [887, 769, 586, 988, 823]
- `{"num_leaves": 15, "min_data_in_leaf": 100}`: 0.8506 ± 0.0015, rounds per fold [678, 1030, 736, 904, 793]
- `{"num_leaves": 31, "min_data_in_leaf": 20}`: 0.8501 ± 0.0011, rounds per fold [496, 589, 386, 618, 460]
- `{"num_leaves": 31, "min_data_in_leaf": 100}`: 0.8502 ± 0.0011, rounds per fold [448, 344, 397, 581, 409]
- `{"num_leaves": 63, "min_data_in_leaf": 20}`: 0.8503 ± 0.0013, rounds per fold [307, 301, 186, 230, 381]
- `{"num_leaves": 63, "min_data_in_leaf": 100}`: 0.8502 ± 0.0016, rounds per fold [231, 333, 234, 297, 279]

**LightGBM LambdaRank**

- `{"num_leaves": 15, "min_data_in_leaf": 20}`: 0.8508 ± 0.0016, rounds per fold [389, 253, 263, 265, 296]
- `{"num_leaves": 15, "min_data_in_leaf": 100}`: 0.8508 ± 0.0013, rounds per fold [360, 308, 323, 316, 312]
- `{"num_leaves": 31, "min_data_in_leaf": 20}`: 0.8505 ± 0.0015, rounds per fold [321, 228, 241, 181, 197]
- `{"num_leaves": 31, "min_data_in_leaf": 100}`: 0.8508 ± 0.0011, rounds per fold [380, 216, 257, 158, 227]
- `{"num_leaves": 63, "min_data_in_leaf": 20}`: 0.8499 ± 0.0014, rounds per fold [312, 193, 242, 189, 145]
- `{"num_leaves": 63, "min_data_in_leaf": 100}`: 0.8503 ± 0.0015, rounds per fold [287, 275, 221, 112, 111]

## 3. Ablation (LambdaRank, best configuration, same folds)

nDCG columns: mean ± std across folds, and a 95% bootstrap CI over training queries of the out-of-fold nDCG (1,000 resamples, same resamples for every row).

| Feature set | #features | nDCG | 95% CI (OOF) | nDCG@5 | Exact-P@1 |
|---|--:|---|---|---|---|
| lexical only | 11 | 0.8356 ± 0.0020 | [0.8336, 0.8376] | 0.6740 ± 0.0024 | 0.6698 ± 0.0078 |
| + attribute + numeric + negation | 24 | 0.8430 ± 0.0020 | [0.8411, 0.8449] | 0.6893 ± 0.0020 | 0.6877 ± 0.0075 |
| + semantic | 26 | 0.8512 ± 0.0019 | [0.8493, 0.8532] | 0.7062 ± 0.0026 | 0.7086 ± 0.0071 |
| all (+ quality, context, within-query) | 38 | 0.8508 ± 0.0013 | [0.8488, 0.8528] | 0.7049 ± 0.0022 | 0.7095 ± 0.0048 |

## 4. Feature importance

LambdaRank (best config): share of total split gain and mean |SHAP| (LightGBM `pred_contrib=True`) on each fold's validation part, averaged over folds. Top 15:

| Feature | Group | Gain share | Mean abs SHAP |
|---|---|--:|--:|
| `cos_title_bullets` | semantic | 0.052 | 0.2643 |
| `cos_title_z` | within_query | 0.370 | 0.2005 |
| `coverage_all` | lexical | 0.127 | 0.1554 |
| `char3_jaccard_title` | lexical | 0.037 | 0.1348 |
| `bm25_all_z` | within_query | 0.144 | 0.1226 |
| `first_token_in_title` | lexical | 0.021 | 0.0914 |
| `neg_term_conflict` | negation | 0.046 | 0.0709 |
| `cos_title` | semantic | 0.013 | 0.0692 |
| `title_len` | quality | 0.019 | 0.0579 |
| `bm25_bullets` | lexical | 0.011 | 0.0490 |
| `num_conflict` | numeric | 0.024 | 0.0464 |
| `brand_match` | attribute | 0.013 | 0.0428 |
| `num_match` | numeric | 0.015 | 0.0399 |
| `bm25_title` | lexical | 0.014 | 0.0385 |
| `jaccard_title` | lexical | 0.006 | 0.0281 |

Logistic regression (best config), standardised coefficients averaged over folds. Top 15 by |coef|:

| Feature | Group | Coefficient (mean ± std) |
|---|---|---|
| `cos_title_bullets` | semantic | +0.323 ± 0.003 |
| `coverage_all` | lexical | +0.282 ± 0.008 |
| `char3_jaccard_title` | lexical | +0.240 ± 0.008 |
| `neg_term_conflict` | negation | -0.197 ± 0.002 |
| `jaccard_title` | lexical | -0.196 ± 0.009 |
| `cos_title_z` | within_query | +0.187 ± 0.008 |
| `brand_conflict` | attribute | -0.162 ± 0.005 |
| `query_has_digit` | context | -0.146 ± 0.004 |
| `colour_conflict` | attribute | -0.131 ± 0.004 |
| `bm25_all_z` | within_query | +0.124 ± 0.003 |
| `num_match` | numeric | +0.116 ± 0.002 |
| `cos_title` | semantic | -0.116 ± 0.009 |
| `first_token_in_title` | lexical | +0.115 ± 0.007 |
| `query_len` | context | -0.104 ± 0.005 |
| `unit_conflict` | numeric | -0.096 ± 0.005 |

## 5. Optional reference: zero-shot cross-encoder (not a candidate for selection)

`cross-encoder/ms-marco-MiniLM-L-6-v2` on a subsample of 500 training queries (9,913 pairs; throughput in `cv_results.json`), compared with the out-of-fold scores of other methods on the same queries.

| Method | nDCG | nDCG@5 | Exact-P@1 |
|---|---|---|---|
| cross_encoder | 0.8400 ± 0.0166 | 0.6737 ± 0.0299 | 0.7204 ± 0.0472 |
| bm25_title | 0.8158 ± 0.0186 | 0.6362 ± 0.0270 | 0.6216 ± 0.0572 |
| encoder_cosine | 0.8356 ± 0.0135 | 0.6682 ± 0.0257 | 0.6669 ± 0.0400 |
| lgbm_lambdarank | 0.8535 ± 0.0192 | 0.7081 ± 0.0304 | 0.7136 ± 0.0554 |

## 6. Chosen model (frozen before the test run)

**LightGBM pointwise** — LightGBM pointwise has mean CV nDCG 0.8506; the best is 0.8508. It is the simplest model within 0.002 of the best (rule of spec §3).

Hyper-parameters: `{"num_leaves": 15, "min_data_in_leaf": 100}`; final number of boosting rounds = median of the per-fold early-stopping rounds = **793**.

Features (38): `bm25_title`, `bm25_bullets`, `bm25_description`, `bm25_all`, `jaccard_title`, `coverage_title`, `coverage_all`, `char3_jaccard_title`, `phrase_in_title`, `first_token_in_title`, `last_token_in_title`, `brand_match`, `brand_conflict`, `colour_match`, `colour_conflict`, `num_match`, `num_conflict`, `unit_match`, `unit_conflict`, `size_match`, `size_conflict`, `neg_cue`, `neg_phrase_in_title`, `neg_term_conflict`, `cos_title`, `cos_title_bullets`, `title_len`, `no_description`, `no_bullets`, `no_brand`, `no_colour`, `query_len`, `query_has_digit`, `n_candidates`, `bm25_all_rank`, `bm25_all_z`, `cos_title_rank`, `cos_title_z`

`scripts/evaluate_test.py` reads the block below and refuses to run unless this file is committed and unchanged.

<!-- FROZEN-MODEL-BEGIN -->
```json
{
  "family": "lgbm_pointwise",
  "config": {
    "num_leaves": 15,
    "min_data_in_leaf": 100
  },
  "features": [
    "bm25_title",
    "bm25_bullets",
    "bm25_description",
    "bm25_all",
    "jaccard_title",
    "coverage_title",
    "coverage_all",
    "char3_jaccard_title",
    "phrase_in_title",
    "first_token_in_title",
    "last_token_in_title",
    "brand_match",
    "brand_conflict",
    "colour_match",
    "colour_conflict",
    "num_match",
    "num_conflict",
    "unit_match",
    "unit_conflict",
    "size_match",
    "size_conflict",
    "neg_cue",
    "neg_phrase_in_title",
    "neg_term_conflict",
    "cos_title",
    "cos_title_bullets",
    "title_len",
    "no_description",
    "no_bullets",
    "no_brand",
    "no_colour",
    "query_len",
    "query_has_digit",
    "n_candidates",
    "bm25_all_rank",
    "bm25_all_z",
    "cos_title_rank",
    "cos_title_z"
  ],
  "reason": "LightGBM pointwise has mean CV nDCG 0.8506; the best is 0.8508. It is the simplest model within 0.002 of the best (rule of spec \u00a73).",
  "n_rounds": 793,
  "all_models": {
    "logreg": {
      "config": {
        "C": 0.01,
        "class_weight": null
      },
      "n_rounds": null
    },
    "lgbm_pointwise": {
      "config": {
        "num_leaves": 15,
        "min_data_in_leaf": 100
      },
      "n_rounds": 793
    },
    "lgbm_lambdarank": {
      "config": {
        "num_leaves": 15,
        "min_data_in_leaf": 100
      },
      "n_rounds": 316
    }
  }
}
```
<!-- FROZEN-MODEL-END -->
