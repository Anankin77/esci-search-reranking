# Viva preparation — 15 likely questions

Numbers are from Reports 1–4 (test = official test split, 8,956 queries; CV = 5-fold grouped CV on 20,888 training queries).

**1. Why group the CV folds by query? What breaks with a random split?**
The unit of evaluation is a query's ranked list, so all ~20 candidates of a query must sit in the same fold. A random
pair-level split would put some candidates of a query in training and others in validation: the model would learn
query-specific patterns (e.g. which product type is Exact for "moto g7 play") and nDCG on that query could not even be
computed on a complete list. Validation scores would be optimistic and the lists incomplete. `GroupKFold` by `query_id`
avoids both; `tests/test_no_leakage.py` checks that no query appears in two folds and that no `train_test_split` /
`KFold(` exists in the code.

**2. Why is BM25 a meaningful baseline? Why does random score so high?**
BM25 is the standard keyword ranker that production systems start from; beating it is the honest bar for "learning
helps". Random already gets nDCG 0.747 because Amazon pre-filtered the candidates: 44 % of them are Exact and every
query has relevant items, so most orderings put something relevant near the top. That is why we always report gains
relative to random and BM25: BM25-title adds +0.071 over random; our model adds +0.036 over BM25 (51 % of that gap).

**3. What does LambdaRank optimise that pointwise regression doesn't?**
Pointwise regression predicts each product's gain independently (squared error), ignoring the other candidates.
LambdaRank uses pairwise gradients within a query, weighted by how much swapping two products would change nDCG,
so it directly targets the ranking metric and the top positions. Here it made no measurable difference: test nDCG
0.8541 (LambdaRank) vs 0.8538 (pointwise), CV 0.8508 vs 0.8506. Because they were within 0.002, the pre-registered rule
picked the simpler pointwise model. Most of the value is in the features, not the loss.

**4. How did you prevent leakage from shared products?**
25.6 % of test pairs involve a product that also appears in training (shared catalog). This could only leak labels if
a feature carried label information about a product (e.g. "how often was this product Exact"). No feature uses labels:
`features.py` receives only query/catalog text (`feature_input`), asserts that `esci_label`, `gain`, `split` are absent,
and a test shuffles the labels and checks that every feature is unchanged. Learned statistics (BM25 IDF, brand
vocabulary, scaler) are fit on training-fold rows only.

**5. Why this encoder? What if it had seen ESCI?**
all-MiniLM-L6-v2 was released in 2021, before ESCI (June 2022), so it cannot have been trained on ESCI labels; no model
whose card mentions ESCI or e-commerce search fine-tuning was used. If it had seen ESCI, its cosine would partly encode
the test labels and the semantic features would inflate results. Remaining limitation: its web pretraining may contain
Amazon-like product text. The ablation bounds its role: semantic features add +0.008 CV nDCG on top of 0.843.

**6. Which category does the model fail on, and why?**
Every text-defined category improves on average, but the smallest gains are colour queries (+0.031 [0.024, 0.039]) and
long queries (+0.032 [0.026, 0.039]): colours are often missing or inconsistent in the colour field, and long
natural-language queries dilute every overlap feature. Negation (BM25 0.763 → model 0.810) and numeric queries (0.792 →
0.834) remain the hardest in absolute terms. Per query, the model loses on 36 % of queries; big losses (Δ < −0.2) are
2.1 %, typically semantic similarity promoting a related item over an exact literal match ("projectors" → an accessory;
the book "live not by lies" → similar self-help books).

**7. What changes if the gains are E=1, S=0.5?**
Substitutes then count as half-relevant, so every method scores higher and the differences shrink: with E=1, S=0.5,
C=0.25 the model − BM25 gap is +0.0215 [0.0199, 0.0231] instead of +0.0362; with exact-only gains it grows to +0.0421.
The conclusion (model > BM25) is unchanged, but part of the model's advantage is about separating Exact from
Substitute. This analysis is post-hoc on frozen predictions (Report 4).

**8. What would you try next, and what would it cost?**
(a) A cross-encoder fine-tuned on the ESCI training split as a feature: the zero-shot ms-marco cross-encoder already
scores 0.840 on a 500-query CV subsample (vs BM25 0.816 there); fine-tuning needs a GPU-hour scale budget and inference
at ~225 pairs/s on CPU (~15 min for the test split). (b) Better attribute extraction (colour normalisation, product
type) to attack the weakest categories — cheap, rule-based. (c) A guard for known-item queries (exact phrase match) to
avoid the big semantic losses — cheap, but must be validated in CV first.

**9. How did you make sure the test split was used only once?**
Report 2, with the chosen model, hyper-parameters, rounds and feature list in a JSON block, was committed (`3024bcb`,
08:02) before `evaluate_test.py` ran (08:42). The script refuses to run unless the report is tracked, committed and
unchanged (`git ls-files`, `git log`, `git diff --quiet HEAD`), reads the frozen settings from that committed file,
and refuses a second scoring run unless a logged `--rerun-reason` is given. Test embeddings were computed only inside
that script. `results/test_run_log.json` shows one run.

**10. Why are your CIs "paired", and why does that matter?**
Every method is evaluated on the same 1,000 bootstrap resamples of test queries, and for differences we resample the
per-query difference. Query difficulty varies a lot (nDCG from ~0.3 to 1), but it affects both methods similarly;
pairing cancels that shared variation, so the CI of model − BM25 ([0.0337, 0.0386]) is much narrower than the two
separate CIs would suggest.

**11. Why did logistic regression predict "Exact vs not", and is it calibrated?**
The spec fixes a simple binary target for the simplest learned model; ranking by P(Exact) matches Exact-P@1 and puts E
first, which dominates nDCG with gains 1 / 0.1 / 0.01. Out-of-fold, it is well calibrated in the middle (predicted 0.45
→ observed 0.43) with a Brier score of 0.218 vs 0.246 for always predicting the base rate (Report 4).

**12. Which features matter most? Do they make sense?**
LambdaRank mean |SHAP|: query vs title+bullets cosine, within-query z-score of the title cosine, share of query words
found in any field, character-3-gram overlap, within-query BM25 z-score. Conflict features have the expected negative
sign in logistic regression (`neg_term_conflict` −0.20, `brand_conflict` −0.16, `colour_conflict` −0.13,
`unit_conflict` −0.10). They explain the model, not shopper intent.

**13. How did you handle BM25 for products not in the training corpus?**
`rank_bm25.BM25Okapi` (k1 = 1.5, b = 0.75) is fit on the training fold's products to obtain IDF and average length; the
same Okapi formula then scores any (query, product) pair. A unit test checks that it equals `BM25Okapi.get_scores` on
corpus documents. Terms unseen in training contribute 0, as in rank_bm25.

**14. You found brands called "without" and "not". What happened there?**
The brand vocabulary is learned from training products. Some catalog brands are ordinary words ("without", "not",
"premium", "box"), so the first version flagged a brand conflict on 34 % of pairs and on every negation query. I added
a label-free distinctiveness filter: a brand name is kept only if at least 30 % of training titles containing it belong
to that brand. Now 14 % of queries contain a brand, and the most frequent are Nike, Disney, adidas, Sony, LEGO
(DECISIONS 18).

**15. How large and how general is the result?**
On this benchmark, the model improves nDCG from 0.818 to 0.854 and puts an Exact product first 9.6 points more often.
It is not a claim about Amazon's search: we only re-rank candidates Amazon already retrieved, the queries were chosen
to be hard, labels are noisy (especially E vs S), and the data are English/US only. Near-duplicate queries (101) and
shared products were measured; removing near-duplicates leaves the gain unchanged (+0.0361).
