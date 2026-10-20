# SPEC — When Does Keyword Matching Fail? Learning to Re-rank Amazon Product Search Results on Hard Queries

Course: ES 335 (Machine Learning), undergraduate track. Final deadline: **Nov 17, 2026**.
Author: Vishaw Raj Meena (24110388).
Deliverables: a reproducible Python pipeline, a set of written reports, and a static interactive website on GitHub Pages.

> **Instructions for Claude Code**
> - Build the **complete project end to end**, phase by phase, without stopping for confirmation.
> - At every `📋 REPORT`, write the report as Markdown in `results/` and continue. I review everything after the build.
> - Log **every** judgment call (feature added or dropped, parameter, data issue, fallback) in `results/DECISIONS.md`: one numbered line per decision, with a one-line reason.
> - If something in this spec is impossible (blocked host, missing field, too slow), apply the fallback named here. If none is named, choose the most conservative option, log it, and continue.
> - **Never** use relevance labels (`esci_label`) to build features. Never use the test split for anything before Phase 6. All tuning and model selection happen inside the training split.
> - Keep the code simple, typed where helpful, and readable. I must be able to explain every line in a viva. Prefer clear functions over clever abstractions.
> - Commit after each phase with a clear message. Never commit raw data, model binaries over 50 MB, API tokens or `.env`.
> - Finish with `results/BUILD_SUMMARY.md` and `results/VIVA.md` (see §11).

---

## 1. The question

**For difficult English shopping queries, does a learning-to-rank model built on interpretable query–product features order candidate products better than the BM25 keyword ranker? Which kinds of queries does each method get wrong?**

- **Unit of prediction:** one (query, product) pair gets a relevance score.
- **Unit of evaluation:** one query, meaning the ranked list of its ~20 candidate products.
- **Prediction-time information:** the query text, and the catalog fields of each candidate (title, description, bullet points, brand, colour). Never the labels.
- **Target:** the graded relevance label: Exact (E), Substitute (S), Complement (C) or Irrelevant (I).
- **Limits of the claim:** we re-rank candidates that Amazon has already retrieved for this benchmark. We say nothing about Amazon's live search, the retrieval stage, or why people buy. A modest or negative result is acceptable and must be reported plainly.

---

## 2. Data

**Amazon Shopping Queries Dataset (ESCI)**, Reddy et al., 2022, *"Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search"* (arXiv:2206.06588). Licence: **Apache-2.0**. Cite it in the README and on the site.

| File | URL (try in order) |
|---|---|
| examples | `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_examples.parquet` |
| products | `https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_products.parquet` |
| sources | `https://raw.githubusercontent.com/amazon-science/esci-data/main/shopping_queries_dataset/shopping_queries_dataset_sources.csv` |

Fallback: the Hugging Face mirror (`tasksource/esci` or similar). If you use it, verify that row counts match §2.1 and log it in DECISIONS.

`scripts/download_data.py` saves the files to `data/raw/`, which is git-ignored, and records SHA-256 checksums and the download date in `data/raw/MANIFEST.json`. The download resumes if interrupted.

### 2.1 Subset

- Filter to `small_version == 1` and `product_locale == "us"`.
- Join `examples` to `products` on **(`product_locale`, `product_id`)**. Report the join coverage.
- **Assert these expected sizes.** Warn and log if counts differ by more than 1%, but don't stop.

| | queries | pairs |
|---|--:|--:|
| total | ~29,844 | ~601,354 |
| train | ~20,888 | ~419,653 |
| test | ~8,956 | ~181,701 |

### 2.2 Relevance gains (fixed for the whole project)

`E = 1.0, S = 0.1, C = 0.01, I = 0.0` (the official ESCI gains). Define them once in `scripts/config.py`.

---

## 3. Splits and validation design

- **Test:** the official `split == "test"` queries. It is **opened exactly once, in Phase 6.**
- **Model selection and tuning:** 5-fold `GroupKFold` on the training queries, grouped by `query_id`. All candidates of a query stay together. Fix the folds once (seeded) and save them to `data/processed/folds.parquet`, so every model uses the same folds.
- **Learned preprocessing:** BM25 statistics, TF-IDF vocabularies, scalers, imputers and vocabularies (e.g. the brand list) are **fit inside each training fold only**. For the final model they are fit on the full training split. This applies to every learned transformation.
- **Model selection rule (fixed before any results):** pick the highest mean CV nDCG. If two models are within 0.002, pick the simpler one, in the order logistic regression < LightGBM pointwise < LightGBM LambdaRank.
- **Freeze before test:** Phase 5 writes `results/report_2_validation.md` containing the chosen model, its hyperparameters and its feature list, and **commits it**. `scripts/evaluate_test.py` must refuse to run unless that file is committed in git and unchanged (check with `git log` and `git diff --quiet`).

---

## 4. Leakage and integrity rules (most important section)

1. **No label-derived features.** No product-level "how often was this product Exact", no query-level label stats. `features.py` must assert that `esci_label` is absent from the feature matrix and from any table used to compute features.
2. **Shared products across splits.** Measure the share of test pairs whose product also appears in training. It's allowed, because the catalog is shared, but report it. Rule 1 prevents it from leaking labels.
3. **Near-duplicate queries across splits.** Normalise queries (lowercase, strip punctuation, collapse spaces, sort tokens), then flag test queries that (a) exactly match a training query after normalisation, or (b) have char-3-gram TF-IDF cosine ≥ 0.90 with one. Report final results **with and without** these queries.
4. **Pretrained-encoder contamination.** Use only encoders **publicly released before June 2022** (ESCI's release), so they cannot have been trained on it. Default: `sentence-transformers/all-MiniLM-L6-v2` (2021). Optional cross-encoder reference: `cross-encoder/ms-marco-MiniLM-L-6-v2` (2021). **Never** use models whose card mentions ESCI, Amazon shopping queries, or e-commerce search fine-tuning. State this choice, and the remaining limitation, on the site.
5. **No LLM judgments** of relevance anywhere.
6. **Query-level normalisation is allowed:** e.g. a product's BM25 rank within its query, or a z-score within the query. It uses only features, not labels.

---

## 5. Build phases

### Phase 1 — Download, build, audit
Scripts: `download_data.py`, `build_dataset.py` → `data/processed/pairs.parquet` (one row per pair, with joined product fields and cleaned text).

Text cleaning (log each step as a decision):
- strip HTML tags and entities, and normalise whitespace and unicode (NFKC)
- lowercase for lexical features only; keep the original text for display
- truncate long descriptions and bullets to a fixed character budget (e.g. 1,000) for speed

📋 **REPORT 1 — `results/report_1_data_audit.md`** (plus figures in `results/figures/`):
- rows, queries and pairs per split, and candidates per query (min, median, max, histogram)
- label mix overall and per split, and the distribution of "share of Exact per query"
- % missing (empty after cleaning) for title, description, bullets, brand and colour, per split
- top 20 brands and how often the brand appears in the query text
- query length distribution, and examples of numeric, negation and brand queries
- join coverage, duplicate (query_id, product_id) pairs and how they were handled
- product overlap across splits (rule 4.2) and near-duplicate queries (rule 4.3)
- 10 example queries with their candidate lists and labels, as a sanity check
- `notebooks/01_eda.ipynb` reproduces the main figures

### Phase 2 — Metrics and baselines (CV on train only)
`scripts/metrics.py`:
- `ndcg(gains_in_ranked_order, k=None)`. Use standard DCG with `log2(rank+1)` discount and the ideal DCG of the same candidate set. **Queries whose ideal DCG is 0** (no E/S/C) are excluded from averages; count and report them.
- `ndcg_at_5`, `exact_precision_at_1` (is the top-ranked product Exact?)
- tie handling: break score ties with a seeded random key, never by row order
- unit tests in `tests/test_metrics.py` against hand-computed examples

Baselines (`scripts/baselines.py`), evaluated per CV fold:
1. **Random ordering**, averaged over 20 seeds. This is the floor.
2. **BM25 on title.** This is the main baseline. Use `rank_bm25` (BM25Okapi, k1 = 1.5, b = 0.75; log both), with IDF from the training fold's product titles.
3. **BM25 on all fields** (title + brand + colour + bullets), as a stronger lexical baseline.
4. **Encoder cosine only:** zero-shot query–title cosine from the §4.4 encoder. A semantic baseline with no learning.

### Phase 3 — Features
`scripts/features.py` builds one matrix per (query, product). Every feature has a one-line docstring entry in `FEATURES.md` covering its name, group, formula and why it might help. Target 30–45 features, in groups (the groups are used for ablations):

| Group | Features (examples) |
|---|---|
| **Lexical** | BM25 for title, bullets, description and all fields; token overlap (Jaccard, query-coverage = share of query tokens in title); char-3-gram Jaccard; exact phrase match; first-query-token-in-title |
| **Attribute** | brand present in query and matches product brand; brand present in query and conflicts with it; colour in query matches or conflicts with product colour |
| **Numeric/units** | parse numbers + units from query and title (`oz, ml, l, lb, kg, g, gb, tb, inch/in/", ft, pack/count/ct, size, x`); flags: number matches, number conflicts, unit matches; size-token agreement (e.g. "size 9", "xl") |
| **Negation** | query contains a negation cue (`without, no, not, free, -free, non`); the negated term appears in the title (a likely conflict) |
| **Semantic** | encoder cosine (query vs title; query vs title + bullets) |
| **Product quality** | title length; missing-field flags (no description / bullets / brand / colour) |
| **Query context** | query length (tokens); has digits; candidate-list size |
| **Within-query normalised** | rank and z-score of the BM25-all and cosine features within the query |

Engineering:
- Compute embeddings **once per unique text**: unique queries, and unique product titles and title + bullets. Cache them to `data/processed/emb_*.npy` with an index. Batch on CPU; log throughput.
- **Fallback:** if the encoder can't be downloaded (blocked host), skip the semantic group, log it, and continue. Mark the results accordingly.
- A `--sample N` flag on every script runs on N training queries, for a fast smoke test (target < 5 minutes end to end).

### Phase 4 — Models (5-fold grouped CV on train)
All models are trained per fold, with preprocessing fit on that fold's training part only.

1. **Logistic regression** (pointwise, simplest learned model): target = "Exact vs not" (log this choice); standardised features; `class_weight` none vs balanced; C ∈ {0.01, 0.1, 1, 10}. Rank by predicted probability.
2. **LightGBM pointwise regression** on the gain value (E/S/C/I → 1/0.1/0.01/0). This is a middle step showing what ranking-aware training adds.
3. **LightGBM LambdaRank** (main model): `objective="lambdarank"`, `group` = candidates per query, `label_gain` set to match §2.2 by mapping labels to integer grades (I=0, C=1, S=2, E=3) with `label_gain=[0, 0.01, 0.1, 1.0]`; `metric="ndcg"`. Small grid: `num_leaves ∈ {15, 31, 63}`, `min_data_in_leaf ∈ {20, 100}`, `learning_rate = 0.05`, early stopping on an inner grouped split taken from the fold's training part, never the fold's validation part. Log the best number of iterations and refit with it.
4. **Optional reference** (only if time and compute allow, and never the main result): cross-encoder `ms-marco-MiniLM-L-6-v2` zero-shot score on a CV subsample.

**Ablations** (LambdaRank, same CV): lexical only → + attribute + numeric + negation → + semantic → all. This shows which feature groups matter and whether complexity pays.

### Phase 5 — Validation report and freeze
📋 **REPORT 2 — `results/report_2_validation.md`:**
- one table: every baseline and model × {nDCG, nDCG@5, Exact-P@1}, as mean ± std across folds, plus the gain over BM25-title and over random
- the ablation table
- feature importance for the main model (gain-based, and mean |SHAP| via LightGBM `pred_contrib=True`, with no extra dependency) and the logistic regression coefficients
- the **chosen model**, hyperparameters, feature list and the reason, applying the §3 rule
- commit this report before Phase 6

### Phase 6 — Final test (run once)
`scripts/evaluate_test.py`:
- refit the chosen model and all its preprocessing on the full training split, with frozen hyperparameters
- score the test split once, and save `results/test_predictions.parquet` (query_id, product_id, scores of every baseline and model, label)
- compute every metric for every method

📋 **REPORT 3 — `results/report_3_test.md`:**
- the main table with **95% bootstrap CIs**: 1,000 resamples of test queries, the same resample for every method, so the CIs are paired
- **paired differences** (model − BM25-title, model − BM25-all, model − encoder-cosine) with CIs
- results **with and without** the near-duplicate test queries (§4.3)
- the win/tie/loss count of queries vs BM25-title, and a histogram of per-query ΔnDCG
- **error analysis by query category.** Categories are rule-based taggers in `scripts/categories.py`, defined from query text only, so they're usable on any query: `numeric`, `negation`, `brand_in_query`, `colour_in_query`, `short (≤2 tokens)`, `long (≥6 tokens)`. Plus one label-based analysis category, `substitute_heavy` (≥ 50% S), clearly marked as analysis-only. Show ΔnDCG with CIs per category.
- **15 queries where the model helps most and 15 where it hurts most**, each with the query, the top 3 of both rankings with labels, and a one-line diagnosis
- an honest interpretation: what improved, where it fails, and how large the gain is relative to random → BM25. Note that random already scores high, because candidate lists are pre-filtered.

### Phase 7 — Extra analyses (optional, if time allows; each clearly labelled)
- **Learning curve:** CV nDCG of the main model vs fraction of training queries (10, 25, 50, 100%)
- **Calibration of the LR probabilities** (reliability plot)
- **Sensitivity to gains:** rescore the frozen test predictions with alternative gains (e.g. E=1, S=0.5, C=0.25), labelled post-hoc

### Phase 8 — Website
See §7.

---

## 6. Metrics summary

| Metric | Why |
|---|---|
| **nDCG (full list), primary** | Official ESCI metric. Graded gains reward putting E above S above C above I. |
| nDCG@5 | What a shopper sees first. |
| Exact-P@1 | "Is the first result exactly what I asked for?" — easy to explain. |

Rules:
- Always report the gain over random and over BM25, not just raw nDCG. Random scores high here.
- **Do not report accuracy or classification metrics as headlines.**
- Every comparison between methods uses paired bootstrap CIs over queries.

---

## 7. Website spec (static, GitHub Pages)

Plain HTML + CSS + vanilla JS. Chart.js from cdnjs is allowed. No build step and no server. Data comes from precomputed JSON in `site/data/`, **under 10 MB total**.

**Structure:** a left sidebar with one section shown at a time (hash routing, e.g. `#results`), previous/next links, and a slide-in menu on phones. Sections: Question · Data · Preprocessing · Baselines & Methods · Evaluation · Results · Query Inspector · Errors · Limitations · Conclusion · Reproduce.

**Interactions:**
1. **Query Inspector (main interaction):**
   - a search box with autocomplete over a sample of test queries (≥ 1,500, stratified so every category and all win/loss levels appear), plus filter chips by category and "helps / hurts / tie"
   - for the chosen query, the BM25-title and model rankings side by side; each product shows title, brand and a colour-coded label chip (E/S/C/I); lines connect the same product across the two lists, so moves are visible
   - the query's nDCG under both methods, and Δ
   - clicking a product shows its **top 5 feature contributions** (from `pred_contrib`), as plain-language bars, e.g. "brand matches query +0.42", "number conflicts −0.31"
2. **Category explorer:** a bar chart of ΔnDCG with CIs per category; clicking a bar filters the inspector to that category.
3. **Ablation chart:** nDCG as feature groups are added, with CIs.
4. **Δ histogram:** per-query ΔnDCG vs BM25. Clicking a bin opens example queries in the inspector.

**Content rules:**
- every number on the site comes from `results/*.json`; no hand-typed numbers
- the Data section credits ESCI (Reddy et al., 2022, Apache-2.0) and shows only the sampled product titles and brands needed for the inspector
- the Limitations section covers §9
- the Reproduce section has the exact commands
- design: clean and minimal, readable on a 390 px phone and at 1280 px, light and dark mode, no console errors. Test with headless Chromium at both widths and save screenshots to `results/screenshots/`.

Deploy via `.github/workflows/pages.yml` (GitHub Actions publishes `site/` on every push to `main`).

---

## 8. Repo structure

```
├── SPEC.md
├── README.md               # overview, setup, data access, run, results summary, licence, AI disclosure
├── FEATURES.md             # every feature: group, definition, reason
├── requirements.txt        # pinned versions
├── run_all.py              # full pipeline; --sample N for a smoke test
├── .gitignore              # data/, *.npy, .env, models over 50 MB
├── .github/workflows/pages.yml
├── scripts/
│   ├── config.py           # paths, seed=42, gains, grids
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── audit.py
│   ├── metrics.py
│   ├── baselines.py
│   ├── text_utils.py       # cleaning, tokenising, number/unit parsing, negation cues
│   ├── embeddings.py
│   ├── features.py
│   ├── categories.py
│   ├── train_cv.py         # CV for all models + ablations → Report 2
│   ├── evaluate_test.py    # the one-time test run → Report 3
│   ├── extra_analyses.py   # Phase 7, labelled post-hoc
│   └── export_site_data.py
├── tests/
│   ├── test_metrics.py
│   ├── test_text_utils.py  # number/unit/negation parsing on tricky examples
│   └── test_no_leakage.py  # asserts: no label in features; preprocessing fit on train only
├── notebooks/01_eda.ipynb
├── results/                # reports, decisions, json, figures, screenshots
└── site/                   # index.html, style.css, app.js, data/
```

Rules:
- `python run_all.py` reproduces everything from download to site data.
- `pytest` passes.
- Fix all seeds (`SEED = 42`). Log runtime per phase in `results/runtime.json`.
- Run `python run_all.py --sample 500` from a **fresh clone in a clean virtual environment** before finishing, and record the result in BUILD_SUMMARY. The full run may reuse cached data.

---

## 9. Limitations to document (site and Report 3)

- Re-ranking only: the candidates were pre-retrieved by Amazon, so retrieval errors are invisible.
- Labels are human judgments with noise and ambiguity, especially S vs E.
- The "hard" subset was chosen to be difficult, so results don't transfer to typical traffic.
- English/US only.
- Shared products across splits (quantified); near-duplicate queries (quantified, with results both ways).
- Encoder: chosen to predate ESCI, but its general web training may still include Amazon-like text.
- Feature contributions explain the model, not shopper intent. Prediction is not causation.

---

## 10. Definition of done

- [ ] Reports 1–3, `DECISIONS.md`, `FEATURES.md`, `BUILD_SUMMARY.md` and `VIVA.md` are written.
- [ ] The test split is used exactly once. Git history shows Report 2 committed **before** `evaluate_test.py` ran.
- [ ] `pytest` passes; the `--sample` run from a fresh clone works.
- [ ] The site renders at 390 px and 1280 px, in light and dark mode, with no console errors. All 11 sections and all 4 interactions work.
- [ ] Every site number is traceable to `results/*.json`.
- [ ] README has setup, data access, licence (ESCI Apache-2.0), citations and the AI-use disclosure.
- [ ] No data files, tokens or `.env` anywhere in the git history.

---

## 11. Final documents

**`results/BUILD_SUMMARY.md`:** what was built; the headline results table (random, BM25-title, BM25-all, encoder cosine, LR, LGBM pointwise, LambdaRank, with CIs); the ablation result in one line; deviations from this spec; known issues; and what remains to be done by hand (e.g. enabling Pages).

**`results/VIVA.md`:** 15 likely viva questions with short answers grounded in this project's numbers, including:
- Why group the CV folds by query? What breaks with a random split?
- Why is BM25 a meaningful baseline? Why does random score so high?
- What does LambdaRank optimise that pointwise regression doesn't?
- How did you prevent leakage from shared products?
- Why this encoder? What if it had seen ESCI?
- Which category does the model fail on, and why?
- What changes if the gains are E=1, S=0.5?
- What would you try next, and what would it cost?

---

## 12. Reviewer acceptance checklist (for me, after the build)

**A. It works**
- [ ] `python run_all.py --sample 500` runs from a fresh clone.
- [ ] The site works on GitHub Pages, on a laptop and a phone.

**B. It's honest (preprocessing 10, evaluation 20)**
- [ ] `GroupKFold` by `query_id`; no random pair-level split anywhere (search for `train_test_split`, `KFold(`).
- [ ] BM25/TF-IDF/scalers are fit inside folds (check `test_no_leakage.py`).
- [ ] No `esci_label` in any feature code path.
- [ ] Report 2's commit is earlier than the test run.

**C. It answers the question (baseline 15, results 10)**
- [ ] The model is compared with BM25-title, BM25-all and encoder cosine, using paired CIs.
- [ ] Gains are stated relative to random and BM25, not just as raw nDCG.
- [ ] The category analysis names where it fails, not only where it wins.
- [ ] Results are reported neutrally, even if the gain is small.

**D. Real-world check (my judgment)**
- [ ] Open 10 random test queries in the inspector. Is the model's top 3 actually better for a shopper?
- [ ] Do the top feature contributions make intuitive sense?

**E. Viva-ready**
- [ ] I can explain every DECISIONS line and every VIVA answer in my own words.
