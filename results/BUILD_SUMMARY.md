# Build summary

## What was built

A reproducible pipeline (`run_all.py`, 15 scripts in `scripts/`), four reports, a decisions log, 39 unit tests
and a static website for the question *“For difficult English shopping queries, does a learning-to-rank model
on interpretable query–product features order candidates better than BM25, and where does each method fail?”*

| Phase | Output |
|---|---|
| 1 Data | ESCI small/US subset built and audited — [Report 1](report_1_data_audit.md), `notebooks/01_eda.ipynb` |
| 2 Metrics + baselines | `metrics.py` (unit-tested), random / BM25-title / BM25-all / encoder cosine, 5-fold grouped CV |
| 3 Features | 38 interpretable features in 8 groups — [FEATURES.md](../FEATURES.md); MiniLM embeddings cached per unique text |
| 4–5 Models + freeze | logistic regression, LightGBM pointwise, LightGBM LambdaRank, ablations, importance — [Report 2](report_2_validation.md), committed in `3024bcb` before the test run |
| 6 Test (run once) | [Report 3](report_3_test.md): paired bootstrap CIs, near-duplicate sensitivity, categories, best/worst queries |
| 7 Extras | [Report 4](report_4_extra.md): learning curve, LR calibration, gain sensitivity (post-hoc) |
| 8 Website | `site/` (11 sections, query inspector, category explorer, ablation chart, Δ histogram), Pages workflow |

## Headline results (official test split, 8,956 queries, scored once)

95 % CIs from 1,000 paired bootstrap resamples of test queries. Chosen model (pre-registered rule): **LightGBM pointwise**.

| Method | nDCG | nDCG@5 | Exact-P@1 |
|---|---|---|---|
| Random (20 seeds) | 0.7469 [0.7440, 0.7498] | 0.5022 [0.4981, 0.5063] | 0.4356 [0.4302, 0.4410] |
| BM25 title | 0.8176 [0.8144, 0.8208] | 0.6380 [0.6322, 0.6436] | 0.6203 [0.6105, 0.6305] |
| BM25 all fields | 0.8204 [0.8170, 0.8239] | 0.6446 [0.6388, 0.6504] | 0.6226 [0.6120, 0.6323] |
| Encoder cosine (zero-shot) | 0.8280 [0.8248, 0.8310] | 0.6577 [0.6523, 0.6632] | 0.6396 [0.6292, 0.6490] |
| Logistic regression | 0.8498 [0.8467, 0.8526] | 0.7040 [0.6984, 0.7090] | 0.7012 [0.6907, 0.7098] |
| **LightGBM pointwise (chosen)** | **0.8538 [0.8507, 0.8569]** | **0.7132 [0.7079, 0.7185]** | **0.7157 [0.7063, 0.7249]** |
| LightGBM LambdaRank | 0.8541 [0.8511, 0.8572] | 0.7119 [0.7067, 0.7171] | 0.7136 [0.7039, 0.7229] |

Paired differences of the chosen model: vs BM25-title **+0.0362 [0.0337, 0.0386]** nDCG (51 % of the random → BM25 gap),
vs BM25-all +0.0333 [0.0309, 0.0359], vs encoder +0.0258 [0.0236, 0.0279]. Without the 101 near-duplicate test queries:
+0.0361 [0.0336, 0.0384]. Win / tie / loss vs BM25-title: 5,549 / 189 / 3,218. CV estimate of the same gain was +0.0366.

**Ablation in one line (CV, LambdaRank):** lexical only 0.8356 → + attribute/numeric/negation 0.8430 → + semantic 0.8512 →
all 38 features 0.8508 (BM25-title 0.8140): lexical features give 59 % of the gain, quality/context/within-query features add nothing.

**Where it fails:** smallest gains on colour (+0.031) and long queries (+0.032); negation and numeric queries remain the
hardest for both methods; the typical large loss is semantic similarity promoting a related item above an exact literal match.

## Deviations from the spec

1. The chosen model is LightGBM pointwise, not LambdaRank: the §3 rule picks the simpler model within 0.002 (0.8506 vs 0.8508 CV).
   Ablation and SHAP importance in Report 2 still use LambdaRank, as the spec names it for those analyses (DECISIONS 57–58).
2. `torch` was installed from PyPI (the pytorch.org CPU index is blocked here), so `requirements.txt` pins the PyPI build (DECISIONS 16).
3. Title + bullets embeddings were truncated to 128 word-pieces for CPU speed (DECISIONS 40).
4. `results/test_predictions.parquet` is produced but not committed (contains labels; `*.parquet` is git-ignored) (DECISIONS 67).
5. `--sample` runs write to `data/sample_results/` and skip the git freeze check, so a smoke test cannot touch the real results (DECISIONS in code: `evaluate_test.py`).
6. Report 3 has an interpretation section written after reading the results (`results/report_3_interpretation.md`, merged by `--report-only`), plus two post-hoc checks clearly labelled.
7. Extra script `scripts/screenshot_site.py` (headless site check) is not in the spec's file list.

## Verification

- `pytest`: 39 tests pass (metrics vs hand-computed values, parsing edge cases, leakage guards, freeze guard, no `train_test_split`/`KFold(`).
- Freeze: Report 2 committed at 2026-10-20 08:02 (`3024bcb`); test scored once at 08:42 (`results/test_run_log.json`).
- Site: headless Chromium at 390 px and 1280 px, light and dark — no console errors, no horizontal overflow, all four interactions
  pass (`results/screenshots/check.json`); site data 6.2 MB.
- Fresh-clone smoke test: passed (section below).
- Full run time on 4 CPU cores ≈ 3.1 h, dominated by the encoder (1.7 h train, ~0.6 h test) — `results/runtime.json`.

## Fresh-clone smoke test (spec §8)

`git clone` of `main` at `fbf4739` from GitHub into an empty folder, new virtual environment, `pip install -r requirements.txt`
(2 min 17 s), then `python run_all.py --sample 500` → **all 10 phases succeeded in 407 s** (6 min 47 s), and `pytest` →
39 passed in the clean environment. The download included the full 1.1 GB products file and hit one connection reset,
which the resume logic recovered from (81 s). Excluding the download, processing took 326 s (≈ 5.4 min) — slightly over
the 5-minute target, mostly embedding the sample's ~10k titles and title+bullets texts on CPU (145 s).

| download | build | audit | embed | features | baselines | train | test | extra | site |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 81 s | 16 s | 4 s | 145 s | 19 s | 2 s | 41 s | 89 s | 8 s | 1 s |


## Known issues

- The run reproduces deterministically on the same machine and library versions; a different CPU or library build can
  change LightGBM and float results in the last digits, which would make the freeze check refuse a rerun of the test
  (by design — use `--report-only`, or `--rerun-reason` which is logged).
- The cross-encoder reference uses only 500 training queries (CIs wide); it is not part of any comparison.
- Rule-based taggers are approximate (e.g. "no" in "no show socks" counts as negation; brand detection needs a brand
  name that is distinctive in training titles, so "apple" and "samsung" are not detected as brands).
- Encoder contamination cannot be fully excluded (general web pretraining).

## What remains to be done by hand

1. Enable GitHub Pages: repository **Settings → Pages → Source: GitHub Actions** (the workflow runs on every push to `main`;
   until Pages is enabled its deploy step fails).
2. Delete the leftover remote branch `ccr-d515e1c4-utamtm` (same commit history as `main`; the build environment could not delete it).
3. Reviewer checklist (§12 D–E): open ~10 random queries in the inspector; read every DECISIONS line and VIVA answer.
