# Decisions log

One numbered line per judgment call, with a one-line reason. Appended in build order.

## Phase 0–1: environment, data, audit
1. Data downloaded from the primary `media.githubusercontent.com` URLs; the Hugging Face fallback was not needed — the first URL worked and preserves official files.
2. SHA-256 and download date recorded in `data/raw/MANIFEST.json`; downloads resume from a `.part` file via HTTP Range — spec requirement.
3. Subset sizes match spec §2.1 exactly (29,844 queries / 601,354 pairs; train 20,888 / 419,653; test 8,956 / 181,701) — no warning needed.
4. Duplicate (query_id, product_id) policy: keep the first occurrence; 0 duplicates were found — conservative and deterministic.
5. `examples` ⟕ `products` on (`product_locale`, `product_id`) as a left join; coverage is 100 % so no pair was dropped — spec §2.1.
6. Text cleaning step 1: strip HTML tags, unescape HTML entities, then strip tags again (catches escaped tags like `&lt;b&gt;`) — descriptions contain raw and escaped HTML.
7. Text cleaning step 2: Unicode NFKC normalisation and whitespace collapsing — makes ligatures, full-width characters and odd spaces comparable.
8. Text cleaning step 3: description and bullet points truncated to 1,000 characters at a word boundary; title, brand and colour are never truncated — speed (spec example budget) while keeping the short fields intact.
9. Lowercasing happens only inside feature code; `pairs.parquet` keeps the cleaned original case for display — spec Phase 1.
10. Tokeniser = lowercase alphanumeric runs, decimals kept (`10.5`), in-word apostrophes dropped (`levi's`→`levis`, `men's`→`mens`) — matches how shoppers type brand and gender words.
11. Folds: `GroupKFold(n_splits=5, shuffle=True, random_state=42)` over training query ids, saved to `data/processed/folds.parquet` (4,177–4,178 queries per fold) — spec §3; every model reads the same file.
12. The audit reads the test split only for the descriptive statistics Report 1 asks for (counts, label mix, missing fields, overlap, near-duplicates); no modelling choice uses them — spec allows these audit items and they cannot influence tuning.
13. Near-duplicate detection: `TfidfVectorizer(analyzer="char", ngram_range=(3,3))` fit on normalised training queries only, cosine computed in chunks of 500 test queries — learned IDF stays on the training side; chunking bounds memory.
14. `sources.csv` (query source such as "negations") is used only as audit metadata, never as a feature — it is not available for an arbitrary new query.
15. Brand normalisation drops trailing corporate words (`inc, llc, ltd, co, corp, corporation, company, electronics, official, store, usa, international, group`) — e.g. "Samsung Electronics" is typed "samsung" by shoppers.
16. Python packages installed from PyPI; `torch` comes from PyPI because the pytorch.org CPU wheel index is blocked by the proxy — same library, larger download only.
17. Brand vocabulary (fit per training fold) keeps brands with ≥ 2 training products, ≥ 3 characters, ≤ 4 tokens, not a placeholder ("generic", "unknown", …), colour, stopword or number — removes non-brands while staying label-free.
18. Brand vocabulary also requires *distinctiveness* ≥ 0.3: among training titles containing the brand name, ≥ 30 % must belong to that brand — the catalog contains brands literally named "without", "not", "premium", "box"; without this filter `brand_conflict` fired on 34 % of pairs and on every negation query.

## Phase 2: metrics and baselines
19. Queries with ideal DCG 0 are excluded from *all* metric averages (nDCG, nDCG@5, Exact-P@1) so every metric uses the same query set; the training and test splits contain 0 such queries — spec Phase 2.
20. Score ties are broken by a seeded random key per (query_id, product_id), generated after sorting rows canonically — makes the tie-break independent of row order (spec).
21. Random baseline: per fold, the metrics of 20 seeded random orderings are averaged; on test, per-query metrics are averaged over the 20 seeds before bootstrapping — spec "averaged over 20 seeds".
22. BM25: `rank_bm25.BM25Okapi(k1=1.5, b=0.75)` is fit on the unique products of the training fold to obtain IDF and average length; scores for any product (including validation/test products outside that corpus) use the identical Okapi formula, unit-tested against `BM25Okapi.get_scores` — rank_bm25 can only score its own corpus.
23. Query terms unseen in the fitted corpus contribute 0 to BM25, as in `rank_bm25` (`idf.get(q) or 0`) — keeps behaviour identical to the library.
24. Duplicate query tokens are counted each time, as in `rank_bm25.get_scores` — identical to the library.
25. "BM25 all fields" text = title + brand + colour + bullets (spec baseline 3); the same definition is used for the `bm25_all` feature; description gets its own BM25 feature — consistency between baseline and feature.
26. Encoder-cosine baseline = cosine(query, title) with L2-normalised all-MiniLM-L6-v2 embeddings — spec baseline 4.

## Phase 3: features
27. 38 features in 8 groups (target 30–45); `FEATURES.md` is generated from the registry in `features.py`, so documentation cannot drift from code.
28. Features split into *static* (pure functions of one pair, computed once per split) and *fitted* (BM25 ×4, `brand_conflict`, `bm25_all` rank/z-score: refit per fold) — only fitted ones need per-fold computation.
29. Overlap features use content tokens (tiny stopword list: a, an, and, the, for, of, with, …) — "for" and "with" otherwise inflate overlap.
30. `char3_jaccard_title` uses character 3-grams of the lowercased, space-collapsed strings (spaces included) — robust to typos and glued words.
31. Colour features compare query colour words with the product's colour field, falling back to colours in the title when the field is empty (31 % empty) — avoids undefined features for a third of the pairs.
32. Colour list is a fixed list of 43 common colour words with gray→grey, multicolour→multicolor, transparent→clear — nothing learned.
33. Number/unit parser: canonical units oz, ml, l, lb, kg, g, gb, tb, inch (in, ", ''), ft, pack (count, ct, pk, pcs, …), x, plus "size N" and "pack of N"; "N in 1" is not read as inches — spec unit list, with the most frequent false positive fixed.
34. `unit_conflict` is set only when there is no exact unit match — a title listing "12 oz" and "32 oz" should not count as a conflict.
35. Size tokens: xxs…5xl (incl. x-large), small/medium/large, petite, plus, king, queen, twin, full, and "size N" — clothing, shoe and bedding sizes.
36. Negation cues: without, no, not, non (cue before the term) and free/-free (cue after the term); "hands free" and "tax free" are excluded; "nothing" is not a cue (whole tokens only) — spec cue list.
37. `neg_term_conflict` = negated term appears in the title *without* a negation next to it; a title saying "sulfate-free" sets `neg_phrase_in_title` instead — a title repeating the negation satisfies the constraint.
38. Within-query rank uses average rank for ties, 1 = highest; z-score is 0 when the within-query std is 0 — avoids division by zero, rank 1 = best is easy to read.
39. `n_candidates` and within-query features use only the query's own candidate list (no labels) — allowed by rule 4.6.
40. Embedding texts: query, title, and "title. bullets" (bullets already truncated to 1,000 characters); title + bullets is truncated to 128 word-pieces, titles use the model's 256 — CPU throughput measured at ~110 title+bullet texts/s at 128 tokens (about half that at 256).
41. Embeddings are computed once per unique text and cached with a text → row index; the training split is embedded in Phase 3 and the test split only inside `evaluate_test.py` — the test split is not touched before Phase 6.
42. Semantic fallback implemented but not needed: the encoder downloaded from huggingface.co.

## Phase 4–5: models and selection
43. Logistic regression target "Exact vs not" (spec), `max_iter=2000`, `StandardScaler` in the same sklearn pipeline so it is fit on the fold's training rows only.
44. LightGBM uses the sklearn API; `min_data_in_leaf` is passed as its sklearn name `min_child_samples`; `deterministic=True`, `force_row_wise=True`, `random_state=42` — reproducible results so that Report 2 can be regenerated byte-identically.
45. Early stopping: `GroupShuffleSplit(test_size=0.2, random_state=42)` on the fold's training queries (never the validation fold), patience 50 rounds, max 2,000 rounds, then refit on the whole fold-training part with the best round count — spec Phase 4.
46. LambdaRank early-stopping metric is full-list nDCG (`eval_at=[1000]`, longer than any candidate list) with `label_gain=[0, 0.01, 0.1, 1]` — matches the primary metric and the official gains.
47. Pointwise LightGBM uses the same 6-point grid as LambdaRank and early-stops on L2 — the spec gives no grid for it; same grid keeps the comparison fair.
48. Each family's configuration is chosen by mean CV nDCG over the same 5 folds used to report it — standard practice; the small optimism this adds is the same for all three families.
49. Final number of boosting rounds for the refit on the full training split = median of the 5 per-fold early-stopping rounds — a single frozen value, written into Report 2.
50. Ablations reuse the CV-selected LambdaRank configuration (re-early-stopped per fold) — re-tuning per feature set would multiply compute by 6 for little information.
51. Ablation CIs: 95 % bootstrap over training queries of out-of-fold nDCG, the same 1,000 resamples for every step — the website needs CIs; fold std alone is shown too.
52. Feature importance = gain share and mean |SHAP| (`pred_contrib=True`) on each fold's validation part, averaged over the 5 fold models; LR coefficients are averaged standardised coefficients — spec Phase 5, no extra dependency.
53. Optional cross-encoder reference run on 500 random training queries (all folds), zero-shot, compared with out-of-fold scores of other methods on the same queries; it is never a selection candidate — spec Phase 4 item 4.
54. The frozen model (family, hyper-parameters, rounds, features) and the CV-selected settings of all three families are stored as a JSON block inside Report 2; `evaluate_test.py` reads them from the committed report — the committed file is the single source of truth.
55. `LGBM_THREADS` can be overridden by an environment variable — LightGBM's OpenMP threads were ~70× slower when the CPU was shared with the embedding job; default stays 4.
