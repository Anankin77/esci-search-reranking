"""Phase 6: the one-time test run -> results/test_predictions.parquet, results/test_results.json
and results/report_3_test.md.

Step 1 (scoring, guarded): refuses to run unless results/report_2_validation.md is committed
in git and unchanged, and refuses a second run unless --rerun-reason is given (logged in
results/test_run_log.json). Every model is refit on the full training split with the
hyper-parameters frozen in Report 2, every learned preprocessing step (BM25 statistics,
brand vocabulary, scaler) is refit on the full training split, then the test split is scored.

Step 2 (analysis): reads the saved predictions only; `--report-only` re-runs this step
without touching the models or rescoring the test split.
"""
from __future__ import annotations

import json
import re
import subprocess
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from categories import (ANALYSIS_ONLY_CATEGORIES, CATEGORY_DESCRIPTIONS, TEXT_CATEGORIES,
                        query_categories)
from config import (GRADES, MODEL_SIMPLICITY, N_RANDOM_SEEDS, ROOT, SEED, RunPaths,
                    base_arg_parser, get_paths, log_runtime)
from embeddings import embed_split, semantic_available
from features import (FEATURE_LABELS, build_matrix, feature_input, fit_state, fold_matrix,
                      load_or_build_static, load_train)
from metrics import METRICS, bootstrap_ci, bootstrap_indices, evaluate_queries
from train_cv import (BASELINE_LABELS, FREEZE_BEGIN, FREEZE_END, MODEL_LABELS, fit_lgbm,
                      fit_logreg, score)

REPORT_2 = "results/report_2_validation.md"
BASELINE_COLUMNS = {"bm25_title": "bm25_title", "bm25_all": "bm25_all", "encoder_cosine": "cos_title"}
TOP_K_CONTRIB = 5


# ---------------------------------------------------------------- freeze guard
def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def freeze_check(rel_path: str = REPORT_2) -> tuple[bool, str]:
    """True if the file is tracked, has at least one commit, and has no uncommitted change."""
    if git("ls-files", "--error-unmatch", rel_path).returncode != 0:
        return False, f"{rel_path} is not tracked by git"
    commit = git("log", "-1", "--format=%H %cI", "--", rel_path).stdout.strip()
    if not commit:
        return False, f"{rel_path} has no commit"
    if git("diff", "--quiet", "HEAD", "--", rel_path).returncode != 0:
        return False, f"{rel_path} has uncommitted changes"
    return True, commit


def read_frozen(report: str) -> dict:
    block = report.split(FREEZE_BEGIN)[1].split(FREEZE_END)[0]
    return json.loads(re.search(r"```json\n(.*?)```", block, re.S).group(1))


# ---------------------------------------------------------------- scoring (once)
def contributions(model, X: pd.DataFrame) -> np.ndarray:
    """Per-feature contribution to each score. LightGBM: SHAP values (pred_contrib).
    Logistic regression: coefficient x standardised value (exact for a linear model)."""
    if hasattr(model, "booster_"):
        return model.predict(X, pred_contrib=True)[:, :-1]
    scaler, lr = model[0], model[-1]
    return scaler.transform(X) * lr.coef_[0]


def score_test(paths: RunPaths, frozen: dict) -> None:
    pairs = pd.read_parquet(paths.processed / "pairs.parquet")
    test = pairs[pairs["split"] == "test"].sort_values(["query_id", "product_id"]).reset_index(drop=True)
    train = load_train(paths)
    train["grade"] = train["esci_label"].map(GRADES)
    features = frozen["features"]

    embed_split(paths, "test")  # encoder is frozen; only embeds the new texts
    assert semantic_available(paths) or not any(f.startswith("cos_") for f in features)
    X_train = fold_matrix(paths, train, None)[features]          # fit on the full training split
    state = fit_state(feature_input(train))
    X_test = build_matrix(test, load_or_build_static(paths, test, "test"), state, features)

    out = test[["query_id", "product_id", "esci_label", "gain"]].copy()
    rng = np.random.default_rng(SEED)
    for s in range(N_RANDOM_SEEDS):
        out[f"random_{s}"] = rng.random(len(out))
    for name, col in BASELINE_COLUMNS.items():
        if col in X_test:
            out[name] = X_test[col].to_numpy()
    for family in MODEL_SIMPLICITY:
        spec = frozen["all_models"][family]
        t0 = time.time()
        if family == "logreg":
            model = fit_logreg(X_train, train, spec["config"]["C"], spec["config"]["class_weight"])
        else:
            objective = "lambdarank" if family == "lgbm_lambdarank" else "regression"
            model, _ = fit_lgbm(X_train, train, spec["config"], objective, n_rounds=spec["n_rounds"])
        out[family] = score(model, X_test)
        print(f"{family}: refit + score in {time.time() - t0:.0f}s")
        if family == frozen["family"]:
            contrib = pd.DataFrame(contributions(model, X_test), columns=features)
            contrib.insert(0, "product_id", test["product_id"].to_numpy())
            contrib.insert(0, "query_id", test["query_id"].to_numpy())
            contrib.to_parquet(paths.processed / "test_contributions.parquet", index=False)
            X_test.assign(query_id=test["query_id"].to_numpy(), product_id=test["product_id"].to_numpy()) \
                .to_parquet(paths.processed / "test_features.parquet", index=False)
    out["model"] = out[frozen["family"]]
    out.to_parquet(paths.results / "test_predictions.parquet", index=False)


# ---------------------------------------------------------------- analysis
def per_query_metrics(pred: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
    """Per-query metrics for each method; random = mean over its 20 seeds."""
    out = {}
    rnd = [evaluate_queries(pred, f"random_{s}").set_index("query_id") for s in range(N_RANDOM_SEEDS)]
    out["random"] = sum(rnd) / len(rnd)
    for m in methods:
        out[m] = evaluate_queries(pred, m).set_index("query_id")
    return out


def table(pq: dict[str, pd.DataFrame], qids: np.ndarray, idx: np.ndarray) -> dict:
    """Mean + paired bootstrap CI of each metric for each method on the given queries."""
    return {m: {k: bootstrap_ci(df.loc[qids, k].to_numpy(), idx) for k in METRICS}
            for m, df in pq.items()}


def diff_table(pq: dict, qids: np.ndarray, idx: np.ndarray, a: str, refs: list[str]) -> dict:
    return {f"{a} - {r}": {k: bootstrap_ci(pq[a].loc[qids, k].to_numpy() - pq[r].loc[qids, k].to_numpy(), idx)
                           for k in METRICS} for r in refs if r in pq}


def top3(g: pd.DataFrame, col: str) -> list[dict]:
    h = g.sort_values([col, "tb"], ascending=[False, True]).head(3)
    return [{"label": l, "title": t[:80], "brand": b} for l, t, b in zip(h["esci_label"], h["title"], h["brand"])]


def diagnose(g: pd.DataFrame, contrib: pd.DataFrame) -> str:
    """Mechanical one-line diagnosis from labels and the model's feature contributions."""
    b = g.sort_values(["bm25_title", "tb"], ascending=[False, True]).iloc[0]
    m = g.sort_values(["model", "tb"], ascending=[False, True]).iloc[0]
    if b["product_id"] == m["product_id"]:
        lead = f"same #1 ({m['esci_label']}); the difference is lower in the list"
    else:
        lead = f"BM25 #1 is {b['esci_label']}, model #1 is {m['esci_label']}"
    diff = contrib.loc[m["product_id"]] - contrib.loc[b["product_id"]]
    top = diff.abs().sort_values(ascending=False).index[:2]
    why = ", ".join(f"{FEATURE_LABELS[f]} {diff[f]:+.2f}" for f in top)
    return f"{lead}; model #1 vs BM25 #1 contribution gap: {why}"


def analyse(paths: RunPaths) -> None:
    frozen = read_frozen((paths.results / "report_2_validation.md").read_text())
    pred = pd.read_parquet(paths.results / "test_predictions.parquet")
    pairs = pd.read_parquet(paths.processed / "pairs.parquet")
    test_text = pairs[pairs["split"] == "test"][["query_id", "product_id", "query", "title", "brand"]]
    pred = pred.merge(test_text, on=["query_id", "product_id"], how="left")
    methods = [m for m in ["bm25_title", "bm25_all", "encoder_cosine", *MODEL_SIMPLICITY] if m in pred]
    pq = per_query_metrics(pred, methods)
    pq["model"] = pq[frozen["family"]]
    valid = pq["bm25_title"]["ndcg"].dropna().index.to_numpy()
    n_excluded = int(pq["bm25_title"]["ndcg"].isna().sum())

    nd = pd.read_parquet(paths.processed / "near_dup_test_queries.parquet")
    nd_ids = set(nd.loc[nd["near_dup"], "query_id"])
    clean = np.array([q for q in valid if q not in nd_ids])
    idx_all, idx_clean = bootstrap_indices(len(valid)), bootstrap_indices(len(clean))
    refs = ["bm25_title", "bm25_all", "encoder_cosine", "random"]
    res = {"chosen_family": frozen["family"], "n_test_queries": int(len(valid)),
           "n_excluded_zero_idcg": n_excluded, "n_near_dup": int(len(set(valid) & nd_ids)),
           "all": {"table": table(pq, valid, idx_all), "diff": diff_table(pq, valid, idx_all, "model", refs)},
           "without_near_dup": {"table": table(pq, clean, idx_clean),
                                "diff": diff_table(pq, clean, idx_clean, "model", refs)}}
    # other models vs BM25-title too (paired)
    res["all"]["diff_models"] = {f: diff_table(pq, valid, idx_all, f, ["bm25_title"]) for f in MODEL_SIMPLICITY}

    # win / tie / loss and per-query delta
    delta = (pq["model"]["ndcg"] - pq["bm25_title"]["ndcg"]).loc[valid]
    res["win_tie_loss"] = {"win": int((delta > 1e-9).sum()), "tie": int((delta.abs() <= 1e-9).sum()),
                           "loss": int((delta < -1e-9).sum())}
    edges = np.round(np.arange(-0.6, 0.6001, 0.05), 2)
    counts, _ = np.histogram(delta.clip(edges[0], edges[-1]), bins=edges)
    res["delta_hist"] = {"edges": edges.tolist(), "counts": counts.tolist()}
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(delta.clip(edges[0], edges[-1]), bins=edges, color="#4a6fa5", edgecolor="white")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(f"ΔnDCG per query ({MODEL_LABELS[frozen['family']]} − BM25 title)")
    ax.set_ylabel("queries")
    fig.tight_layout()
    fig.savefig(paths.figures / "test_delta_hist.png", dpi=120)
    plt.close(fig)

    # categories (text-only taggers; brand vocabulary from the training split)
    train = load_train(paths)
    state_vocab = fit_state(feature_input(train)).brand_vocab
    cats = query_categories(pairs[pairs["split"] == "test"], state_vocab).set_index("query_id")
    cats.to_parquet(paths.processed / "test_categories.parquet")
    cat_res = {}
    for c in TEXT_CATEGORIES + ANALYSIS_ONLY_CATEGORIES:
        q = np.array([x for x in valid if cats.loc[x, c]])
        if len(q) < 2:
            continue
        idx = bootstrap_indices(len(q))
        cat_res[c] = {"n": int(len(q)), "analysis_only": c in ANALYSIS_ONLY_CATEGORIES,
                      "description": CATEGORY_DESCRIPTIONS[c],
                      "model_ndcg": bootstrap_ci(pq["model"].loc[q, "ndcg"].to_numpy(), idx),
                      "bm25_title_ndcg": bootstrap_ci(pq["bm25_title"].loc[q, "ndcg"].to_numpy(), idx),
                      "delta_vs_bm25_title": bootstrap_ci(delta.loc[q].to_numpy(), idx),
                      "delta_vs_encoder": bootstrap_ci((pq["model"]["ndcg"] - pq["encoder_cosine"]["ndcg"]).loc[q].to_numpy(), idx)
                      if "encoder_cosine" in pq else None,
                      "win": int((delta.loc[q] > 1e-9).sum()), "loss": int((delta.loc[q] < -1e-9).sum())}
    res["categories"] = cat_res

    # helps most / hurts most
    contrib = pd.read_parquet(paths.processed / "test_contributions.parquet")
    pred["tb"] = np.random.default_rng(SEED).random(len(pred))
    examples = {}
    for kind, ids in (("helps", delta.sort_values(ascending=False).index[:15]),
                      ("hurts", delta.sort_values().index[:15])):
        rows = []
        for qid in ids:
            g = pred[pred["query_id"] == qid]
            c = contrib[contrib["query_id"] == qid].set_index("product_id").drop(columns="query_id")
            rows.append({"query_id": int(qid), "query": g["query"].iloc[0], "delta": float(delta[qid]),
                         "bm25_ndcg": float(pq["bm25_title"].loc[qid, "ndcg"]),
                         "model_ndcg": float(pq["model"].loc[qid, "ndcg"]),
                         "bm25_top3": top3(g, "bm25_title"), "model_top3": top3(g, "model"),
                         "diagnosis": diagnose(g, c)})
        examples[kind] = rows
    res["examples"] = examples
    (paths.results / "test_results.json").write_text(json.dumps(res, indent=2))
    write_report_3(paths, res, frozen)


# ---------------------------------------------------------------- report 3
def ci(c: dict, d: int = 4) -> str:
    return f"{c['mean']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def label(m: str) -> str:
    return {**BASELINE_LABELS, **MODEL_LABELS, "model": "Chosen model"}.get(m, m)


def main_table(t: dict) -> list[str]:
    lines = ["| Method | nDCG | nDCG@5 | Exact-P@1 |", "|---|---|---|---|"]
    for m in ["random", "bm25_title", "bm25_all", "encoder_cosine", *MODEL_SIMPLICITY]:
        if m in t:
            lines.append(f"| {label(m)} | {ci(t[m]['ndcg'])} | {ci(t[m]['ndcg5'])} | {ci(t[m]['p1'])} |")
    return lines


def diff_lines(d: dict) -> list[str]:
    lines = ["| Paired difference | ΔnDCG | ΔnDCG@5 | ΔExact-P@1 |", "|---|---|---|---|"]
    for name, v in d.items():
        lines.append(f"| {name} | {ci(v['ndcg'])} | {ci(v['ndcg5'])} | {ci(v['p1'])} |")
    return lines


def write_report_3(paths: RunPaths, res: dict, frozen: dict) -> None:
    fam = frozen["family"]
    t, d = res["all"]["table"], res["all"]["diff"]
    rnd, bm, mod = t["random"]["ndcg"]["mean"], t["bm25_title"]["ndcg"]["mean"], t["model"]["ndcg"]["mean"]
    gap = bm - rnd
    wtl = res["win_tie_loss"]
    run_log = json.loads((paths.results / "test_run_log.json").read_text()) if (paths.results / "test_run_log.json").exists() else []
    md = ["# Report 3 — Final test (official test split, scored once)", "",
          f"Chosen model (frozen in Report 2): **{MODEL_LABELS[fam]}** `{json.dumps(frozen['config'])}`"
          + (f", {frozen['n_rounds']} rounds" if frozen.get("n_rounds") else "") + ". "
          "All models and preprocessing were refit on the full training split with frozen settings.", "",
          f"Test queries evaluated: {res['n_test_queries']:,} (excluded for ideal DCG = 0: {res['n_excluded_zero_idcg']}). "
          "95% CIs: 1,000 bootstrap resamples of test queries, the same resamples for every method (paired).", ""]
    if run_log:
        md += [f"Test run log: {len(run_log)} scoring run(s); first at {run_log[0]['time']} with Report 2 commit "
               f"`{run_log[0]['report_2_commit']}`.", ""]
    md += ["## 1. Main results (all test queries)", "", *main_table(t), "",
           *diff_lines(d), "",
           f"Gap random → BM25-title: {gap:+.4f} nDCG. Chosen model − BM25-title: {mod - bm:+.4f} "
           f"(= {100 * (mod - bm) / gap:.0f}% of the random → BM25 gap)." if gap else "", "",
           "Other models vs BM25-title (paired):", ""]
    md += diff_lines({k: v for f in MODEL_SIMPLICITY for k, v in res["all"]["diff_models"][f].items()})
    tc, dc = res["without_near_dup"]["table"], res["without_near_dup"]["diff"]
    md += ["", f"## 2. Without the {res['n_near_dup']} near-duplicate test queries (spec §4.3)", "",
           *main_table(tc), "", *diff_lines(dc), "",
           "## 3. Per-query comparison with BM25-title", "",
           f"Win / tie / loss (nDCG, chosen model vs BM25-title): **{wtl['win']} / {wtl['tie']} / {wtl['loss']}**.", "",
           "![delta histogram](figures/test_delta_hist.png)", "",
           "## 4. Error analysis by query category", "",
           "Categories are rule-based taggers on the query text (`scripts/categories.py`). "
           "`substitute_heavy` uses labels and is **analysis-only**.", "",
           "| Category | Queries | BM25-title nDCG | Model nDCG | ΔnDCG vs BM25-title [95% CI] | ΔnDCG vs encoder | Win / loss |",
           "|---|--:|---|---|---|---|---|"]
    for c, v in res["categories"].items():
        name = f"{c} *(analysis-only)*" if v["analysis_only"] else c
        enc = ci(v["delta_vs_encoder"]) if v.get("delta_vs_encoder") else "–"
        md.append(f"| {name} | {v['n']:,} | {v['bm25_title_ndcg']['mean']:.4f} | {v['model_ndcg']['mean']:.4f} | "
                  f"{ci(v['delta_vs_bm25_title'])} | {enc} | {v['win']} / {v['loss']} |")
    weakest = min(res["categories"].items(), key=lambda kv: kv[1]["delta_vs_bm25_title"]["mean"])
    strongest = max(res["categories"].items(), key=lambda kv: kv[1]["delta_vs_bm25_title"]["mean"])
    md += ["", f"Largest gain over BM25-title: **{strongest[0]}** ({ci(strongest[1]['delta_vs_bm25_title'])}). "
           f"Smallest gain: **{weakest[0]}** ({ci(weakest[1]['delta_vs_bm25_title'])}).", ""]
    for kind, title in (("helps", "15 queries where the model helps most"),
                        ("hurts", "15 queries where the model hurts most")):
        md += [f"## {'5' if kind == 'helps' else '6'}. {title}", ""]
        for e in res["examples"][kind]:
            md.append(f"**{e['query']}** — ΔnDCG {e['delta']:+.3f} (BM25 {e['bm25_ndcg']:.3f} → model {e['model_ndcg']:.3f})")
            md.append("")
            md.append("- BM25 top 3: " + " · ".join(f"`{x['label']}` {x['title'][:60]}" for x in e["bm25_top3"]))
            md.append("- Model top 3: " + " · ".join(f"`{x['label']}` {x['title'][:60]}" for x in e["model_top3"]))
            md.append(f"- Diagnosis: {e['diagnosis']}")
            md.append("")
    interp = paths.results / "report_3_interpretation.md"
    md += ["## 7. Interpretation", ""]
    md += [interp.read_text().strip() if interp.exists() else
           "_(written after inspecting the numbers above; see `results/report_3_interpretation.md`)_", ""]
    md += ["## 8. Limitations", "",
           "- Re-ranking only: candidates were pre-retrieved by Amazon, so retrieval errors are invisible.",
           "- Labels are human judgments with noise and ambiguity, especially Substitute vs Exact.",
           "- The small version was selected to be difficult, so results do not transfer to typical traffic.",
           "- English / US locale only.",
           f"- Shared products across splits (25.6% of test pairs, Report 1) and near-duplicate queries "
           f"({res['n_near_dup']} test queries; results shown with and without).",
           "- Encoder chosen to predate ESCI (2021), but its general web training may include Amazon-like text.",
           "- Feature contributions explain the model, not shopper intent. Prediction is not causation.", ""]
    (paths.results / "report_3_test.md").write_text("\n".join(md))
    print("\n".join(md[:60]))


# ---------------------------------------------------------------- main
def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="re-run the analysis on saved predictions (no refit, no rescoring)")
    parser.add_argument("--rerun-reason", default=None,
                        help="required to score the test split a second time; logged")
    args = parser.parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    if not args.report_only:
        log_file = paths.results / "test_run_log.json"
        log = json.loads(log_file.read_text()) if log_file.exists() else []
        if paths.is_sample:
            commit = "sample run: freeze check skipped (smoke test, results not used)"
            print("SAMPLE RUN: skipping the git freeze check; outputs go to", paths.results)
        else:
            ok, commit = freeze_check()
            if not ok:
                raise SystemExit(f"REFUSING to touch the test split: {commit}. "
                                 f"Commit {REPORT_2} first (spec §3).")
        if log and not args.rerun_reason and not paths.is_sample:
            raise SystemExit("REFUSING: the test split was already scored "
                             f"({log[0]['time']}). Use --report-only, or --rerun-reason to log a rerun.")
        frozen = read_frozen((paths.results / "report_2_validation.md").read_text())
        score_test(paths, frozen)
        log.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "report_2_commit": commit,
                    "rerun_reason": args.rerun_reason})
        log_file.write_text(json.dumps(log, indent=2))
    analyse(paths)
    log_runtime(paths, "evaluate_test" + ("_report" if args.report_only else ""), time.time() - t0)


if __name__ == "__main__":
    main()
