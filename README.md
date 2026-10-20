# When Does Keyword Matching Fail? Learning to Re-rank Amazon Product Search Results on Hard Queries

ES 335 (Machine Learning) final project · Vishaw Raj Meena (24110388)

**Question.** For difficult English shopping queries, does a learning-to-rank model built on interpretable
query–product features order candidate products better than the BM25 keyword ranker? Which kinds of queries
does each method get wrong?

We re-rank the ~20 candidates that Amazon already retrieved for each query of the ESCI benchmark (small
version = hard queries, US locale), compare three learned models with four non-learned baselines under a
leakage-safe protocol, and open the test split exactly once.

- Reports: [data audit](results/report_1_data_audit.md) · [validation](results/report_2_validation.md) ·
  [test](results/report_3_test.md) · [extra analyses](results/report_4_extra.md)
- Every judgment call: [results/DECISIONS.md](results/DECISIONS.md) · features: [FEATURES.md](FEATURES.md)
- Summary and viva prep: [BUILD_SUMMARY](results/BUILD_SUMMARY.md) · [VIVA](results/VIVA.md)
- Website: `site/` (GitHub Pages, see below)

<!-- RESULTS-START -->
## Results summary

Official test split (8,956 queries), scored once after the model was frozen; 95 % paired bootstrap CIs.

| Method | nDCG | nDCG@5 | Exact-P@1 |
|---|---|---|---|
| Random (20 seeds) | 0.747 [0.744, 0.750] | 0.502 | 0.436 |
| BM25 title | 0.818 [0.814, 0.821] | 0.638 | 0.620 |
| BM25 all fields | 0.820 [0.817, 0.824] | 0.645 | 0.623 |
| Encoder cosine (zero-shot) | 0.828 [0.825, 0.831] | 0.658 | 0.640 |
| Logistic regression | 0.850 [0.847, 0.853] | 0.704 | 0.701 |
| **LightGBM pointwise (chosen)** | **0.854 [0.851, 0.857]** | **0.713** | **0.716** |
| LightGBM LambdaRank | 0.854 [0.851, 0.857] | 0.712 | 0.714 |

- Chosen model − BM25-title: **+0.036 nDCG [0.034, 0.039]**, about half of BM25's own gain over random; Exact-P@1 +9.6 points.
  Unchanged without the 101 near-duplicate test queries.
- Lexical features carry most of the gain; attribute/numeric/negation and semantic features add the rest; the learning
  algorithm matters little (pointwise ≈ LambdaRank, by the pre-registered rule the simpler pointwise model was chosen).
- The model gains in every query category; smallest gains on colour and long queries, and its typical large loss is a
  semantically related item promoted above an exact keyword match. Details in [Report 3](results/report_3_test.md).
<!-- RESULTS-END -->

## Setup

Python 3.13, CPU only (4 cores were used; no GPU needed).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data access

`python scripts/download_data.py` downloads the three ESCI files to `data/raw/` (git-ignored), resumes
interrupted downloads, and records SHA-256 checksums and the download date in `data/raw/MANIFEST.json`.
No data file is committed to this repository.

## Run

```bash
python run_all.py --sample 500      # smoke test on 500 training (+250 test) queries; outputs in data/sample_results/
python run_all.py                   # full pipeline; stops before the test phase until Report 2 is committed
git add results/report_2_validation.md && git commit -m "Freeze model before test"
python run_all.py --from test       # one-time test run, extra analyses, site data
pytest                              # metric, parsing and leakage tests
python scripts/screenshot_site.py   # headless check of the site at 390 px and 1280 px, light and dark
python -m http.server -d site       # view the site at http://localhost:8000
```

| Phase | Script | Output |
|---|---|---|
| 1 | `download_data.py`, `build_dataset.py`, `audit.py` | `data/processed/pairs.parquet`, `folds.parquet`, Report 1 |
| 2 | `metrics.py`, `baselines.py` | `results/cv_baselines.json` |
| 3 | `embeddings.py`, `features.py` | embedding cache, feature matrices, `FEATURES.md` |
| 4–5 | `train_cv.py` | `results/cv_results.json`, Report 2 (frozen model) |
| 6 | `evaluate_test.py` | `results/test_results.json`, Report 3 (refuses to run unless Report 2 is committed and unchanged) |
| 7 | `extra_analyses.py` | `results/extra_analyses.json`, Report 4 |
| 8 | `export_site_data.py` | `site/data/*.json` |

All seeds are fixed (`SEED = 42`); runtimes are logged in `results/runtime.json`.

## Integrity rules (short version)

- Relevance labels never enter feature code (`features.py` asserts it; `tests/test_no_leakage.py` checks it).
- 5-fold `GroupKFold` by `query_id`; BM25 statistics, brand vocabulary and scaler are fit inside each training fold.
- The test split is used only by `evaluate_test.py`, after Report 2 is committed (`git log` + `git diff --quiet` check).
- Encoder released before ESCI (all-MiniLM-L6-v2, 2021); no LLM relevance judgments.
- Near-duplicate test queries are flagged and results are reported with and without them.

## Website

`site/` is plain HTML/CSS/JS with Chart.js from cdnjs; all numbers are read from `site/data/site.json`
(exported from `results/*.json`). `.github/workflows/pages.yml` publishes `site/` on every push to `main`.
To enable it: repository **Settings → Pages → Build and deployment → Source: GitHub Actions**.

## Data licence and citation

The Shopping Queries Dataset is released by Amazon under the **Apache License 2.0**.

> Chandan K. Reddy, Lluís Màrquez, Fran Valero, Nikhil Rao, Hugo Zaragoza, Sambaran Bandyopadhyay,
> Arnab Biswas, Anlu Xing, Karthik Subbian. 2022. *Shopping Queries Dataset: A Large-Scale ESCI Benchmark
> for Improving Product Search.* arXiv:2206.06588. https://github.com/amazon-science/esci-data

Other tools cited: BM25 via `rank_bm25` (Okapi BM25, Robertson & Zaragoza 2009); LightGBM (Ke et al., 2017)
and LambdaRank (Burges, 2010); Sentence-Transformers `all-MiniLM-L6-v2` (Reimers & Gurevych, 2019; model
released 2021); cross-encoder `ms-marco-MiniLM-L-6-v2` (optional reference only).

## AI-use disclosure

The code, reports, website and documentation in this repository were drafted with Claude Code (an AI
coding assistant) from a written specification (`SPEC.md`) authored by the student. The student specified
the research question, the protocol, the leakage rules, the metrics and the deliverables, and is responsible
for reviewing, understanding and defending every line. Every judgment call made during the build is logged
in `results/DECISIONS.md`. No large language model was used to judge relevance or to produce any label,
feature or prediction.
