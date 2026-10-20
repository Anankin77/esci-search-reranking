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
