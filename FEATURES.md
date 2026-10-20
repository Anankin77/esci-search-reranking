# Features

Generated from `FEATURE_DOCS` in `scripts/features.py` (single source of truth).
38 features in 8 groups. No feature uses relevance labels. *Fitted* features learn statistics (BM25 IDF / average length, brand vocabulary) from the training part of each CV fold, or from the full training split for the final model.

| # | Feature | Group | Formula | Why it might help | Fitted? |
|--:|---|---|---|---|---|
| 1 | `bm25_title` | lexical | BM25(query, title), IDF from training-fold titles | classic keyword relevance on the most informative field | yes |
| 2 | `bm25_bullets` | lexical | BM25(query, bullet points) | attributes often appear only in bullets | yes |
| 3 | `bm25_description` | lexical | BM25(query, description) | long-form text catches extra query terms | yes |
| 4 | `bm25_all` | lexical | BM25(query, title + brand + colour + bullets) | one keyword score over all short catalog fields | yes |
| 5 | `jaccard_title` | lexical | |Q ∩ T| / |Q ∪ T| on content tokens | symmetric overlap; penalises long off-topic titles | no |
| 6 | `coverage_title` | lexical | share of query content tokens found in the title | did the title mention everything the shopper typed? | no |
| 7 | `coverage_all` | lexical | share of query content tokens found in title/brand/colour/bullets | same, but over all short fields | no |
| 8 | `char3_jaccard_title` | lexical | Jaccard of character 3-gram sets of query and title | robust to typos, plurals and glued words | no |
| 9 | `phrase_in_title` | lexical | 1 if the whole lowercased query appears in the title | exact phrase match is strong evidence | no |
| 10 | `first_token_in_title` | lexical | 1 if the first query token is in the title | first token is often the brand or main noun | no |
| 11 | `last_token_in_title` | lexical | 1 if the last query token is in the title | in English queries the head noun is often last | no |
| 12 | `brand_match` | attribute | 1 if the product's brand appears in the query | shopper asked for this brand | no |
| 13 | `brand_conflict` | attribute | 1 if the query names a known brand (training vocabulary) and the product has a different brand | wrong brand is usually a substitute, not exact | yes |
| 14 | `colour_match` | attribute | 1 if a query colour is in the product colour (title colours if colour field empty) | colour requested and present | no |
| 15 | `colour_conflict` | attribute | 1 if the query has a colour and the product has only other colours | wrong colour is a common substitute | no |
| 16 | `num_match` | numeric | 1 if a query number also appears in the title | model numbers, sizes and quantities agree | no |
| 17 | `num_conflict` | numeric | 1 if query and title have numbers but share none | different size/quantity/model | no |
| 18 | `unit_match` | numeric | 1 if a query (number, unit) pair appears in the title | '12 oz' asked and offered | no |
| 19 | `unit_conflict` | numeric | 1 if a query unit appears in the title only with other numbers | '12 oz' asked but '32 oz' offered | no |
| 20 | `size_match` | numeric | 1 if a query size token (xl, size 9, queen...) is in the title | size agreement for clothes, shoes, bedding | no |
| 21 | `size_conflict` | numeric | 1 if both have size tokens but none is shared | wrong size | no |
| 22 | `neg_cue` | negation | 1 if the query has a negation cue (without/no/not/non/free) | keyword matching rewards the very term the shopper wants to avoid | no |
| 23 | `neg_phrase_in_title` | negation | 1 if the title repeats the negation ('sulfate-free') | product satisfies the negative constraint | no |
| 24 | `neg_term_conflict` | negation | 1 if a negated term appears in the title without negation | product likely contains what the shopper wants to avoid | no |
| 25 | `cos_title` | semantic | cosine(MiniLM(query), MiniLM(title)) | meaning-level similarity; handles synonyms | no |
| 26 | `cos_title_bullets` | semantic | cosine(MiniLM(query), MiniLM(title + bullets)) | semantic match using attribute text too | no |
| 27 | `title_len` | quality | number of title tokens | very short/long titles behave differently | no |
| 28 | `no_description` | quality | 1 if description is empty | sparse listings may be lower quality | no |
| 29 | `no_bullets` | quality | 1 if bullet points are empty | sparse listings | no |
| 30 | `no_brand` | quality | 1 if brand is empty | sparse listings; brand features undefined | no |
| 31 | `no_colour` | quality | 1 if colour is empty | colour features fall back to title | no |
| 32 | `query_len` | context | number of query tokens | long queries are more specific | no |
| 33 | `query_has_digit` | context | 1 if the query contains a digit | numeric queries are harder | no |
| 34 | `n_candidates` | context | number of candidates of this query | list size changes base rates | no |
| 35 | `bm25_all_rank` | within_query | rank of bm25_all within the query (1 = highest) | relative keyword strength among the candidates | yes |
| 36 | `bm25_all_z` | within_query | z-score of bm25_all within the query | relative keyword strength, scale-free | yes |
| 37 | `cos_title_rank` | within_query | rank of cos_title within the query (1 = highest) | relative semantic strength among the candidates | no |
| 38 | `cos_title_z` | within_query | z-score of cos_title within the query | relative semantic strength, scale-free | no |

## Groups used in the ablation

- **lexical** (11): `bm25_title`, `bm25_bullets`, `bm25_description`, `bm25_all`, `jaccard_title`, `coverage_title`, `coverage_all`, `char3_jaccard_title`, `phrase_in_title`, `first_token_in_title`, `last_token_in_title`
- **attribute** (4): `brand_match`, `brand_conflict`, `colour_match`, `colour_conflict`
- **numeric** (6): `num_match`, `num_conflict`, `unit_match`, `unit_conflict`, `size_match`, `size_conflict`
- **negation** (3): `neg_cue`, `neg_phrase_in_title`, `neg_term_conflict`
- **semantic** (2): `cos_title`, `cos_title_bullets`
- **quality** (5): `title_len`, `no_description`, `no_bullets`, `no_brand`, `no_colour`
- **context** (3): `query_len`, `query_has_digit`, `n_candidates`
- **within_query** (4): `bm25_all_rank`, `bm25_all_z`, `cos_title_rank`, `cos_title_z`
