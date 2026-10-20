"""Phase 4-5: 5-fold grouped CV on the training split for every model, ablations,
feature importance, model selection and Report 2 (results/report_2_validation.md).

Models (all preprocessing fit on the fold's training part only):
1. logistic regression, target "Exact vs not", standardised features
2. LightGBM pointwise regression on the gain (1 / 0.1 / 0.01 / 0)
3. LightGBM LambdaRank on integer grades with label_gain = official gains (main model)
Early stopping for LightGBM uses an inner grouped split of the fold's TRAINING part,
then the model is refit on the whole training part with the best number of rounds.
"""
from __future__ import annotations

import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import fold_summary
from config import (CROSS_ENCODER_NAME, GRADES, INNER_VALID_FRACTION, LABEL_GAIN,
                    LGBM_EARLY_STOPPING, LGBM_GRID, LGBM_LEARNING_RATE, LGBM_MAX_ROUNDS,
                    LGBM_THREADS, LR_C_GRID, LR_CLASS_WEIGHTS, MODEL_SIMPLICITY, SEED,
                    TIE_MARGIN, RunPaths, base_arg_parser, get_paths, log_runtime)
from embeddings import semantic_available
from features import FEATURE_DOCS, FEATURE_GROUPS, available_features, fold_matrix, load_train
from metrics import (METRIC_NAMES, METRICS, bootstrap_ci, bootstrap_indices, evaluate_queries,
                     summarize)

MODEL_LABELS = {"logreg": "Logistic regression", "lgbm_pointwise": "LightGBM pointwise",
                "lgbm_lambdarank": "LightGBM LambdaRank"}
BASELINE_LABELS = {"random": "Random (20 seeds)", "bm25_title": "BM25 title",
                   "bm25_all": "BM25 all fields", "encoder_cosine": "Encoder cosine (zero-shot)"}
FREEZE_BEGIN, FREEZE_END = "<!-- FROZEN-MODEL-BEGIN -->", "<!-- FROZEN-MODEL-END -->"


# ---------------------------------------------------------------- model builders
def lgbm_params(config: dict, objective: str) -> dict:
    """LightGBM parameters shared by both LightGBM models (deterministic)."""
    p = {
        "objective": objective,
        "learning_rate": LGBM_LEARNING_RATE,
        "num_leaves": config["num_leaves"],
        "min_child_samples": config["min_data_in_leaf"],  # sklearn-API name of min_data_in_leaf
        "n_jobs": LGBM_THREADS,
        "random_state": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    if objective == "lambdarank":
        p["label_gain"] = LABEL_GAIN
    return p


def group_sizes(query_ids: np.ndarray) -> np.ndarray:
    """Candidates per query, in order of appearance (rows must be sorted by query)."""
    _, idx, counts = np.unique(query_ids, return_index=True, return_counts=True)
    return counts[np.argsort(idx)]


def fit_lgbm(X: pd.DataFrame, df: pd.DataFrame, config: dict, objective: str,
             n_rounds: int | None = None) -> tuple[lgb.LGBMModel, int]:
    """Fit a LightGBM model on rows (X, df). If n_rounds is None, find it by early
    stopping on an inner grouped split of these rows, then refit on all of them."""
    params = lgbm_params(config, objective)
    target = df["gain"].to_numpy() if objective == "regression" else df["grade"].to_numpy()
    qid = df["query_id"].to_numpy()

    def make(n: int) -> lgb.LGBMModel:
        cls = lgb.LGBMRanker if objective == "lambdarank" else lgb.LGBMRegressor
        return cls(n_estimators=n, **params)

    if n_rounds is None:
        gss = GroupShuffleSplit(n_splits=1, test_size=INNER_VALID_FRACTION, random_state=SEED)
        in_tr, in_va = next(gss.split(X, groups=qid))
        model = make(LGBM_MAX_ROUNDS)
        fit_kw = {"eval_X": X.iloc[in_va], "eval_y": target[in_va],
                  "callbacks": [lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)]}
        if objective == "lambdarank":
            fit_kw["group"] = group_sizes(qid[in_tr])
            fit_kw["eval_group"] = [group_sizes(qid[in_va])]
            fit_kw["eval_at"] = [1000]  # longer than any list -> full-list nDCG, the primary metric
        model.fit(X.iloc[in_tr], target[in_tr], **fit_kw)
        n_rounds = max(int(model.best_iteration_ or LGBM_MAX_ROUNDS), 1)
    model = make(n_rounds)
    if objective == "lambdarank":
        model.fit(X, target, group=group_sizes(qid))
    else:
        model.fit(X, target)
    return model, n_rounds


def fit_logreg(X: pd.DataFrame, df: pd.DataFrame, C: float, class_weight) -> object:
    """Standardise (scaler fit on these rows only) + logistic regression on Exact-vs-not."""
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(C=C, class_weight=class_weight, max_iter=2000))
    model.fit(X, (df["esci_label"] == "E").astype(int))
    return model


def score(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


# ---------------------------------------------------------------- CV loop
def cv_evaluate(paths: RunPaths, train: pd.DataFrame, family: str, config: dict,
                features: list[str], keep_models: bool = False) -> dict:
    """Run one model configuration over the 5 folds; return per-fold metrics,
    out-of-fold scores and (optionally) the fold models."""
    per_fold, oof, rounds, models = [], np.full(len(train), np.nan), [], []
    for k in sorted(train["fold"].unique()):
        X = fold_matrix(paths, train, int(k))[features]
        tr, va = (train["fold"] != k).to_numpy(), (train["fold"] == k).to_numpy()
        if family == "logreg":
            model = fit_logreg(X[tr], train[tr], config["C"], config["class_weight"])
        else:
            objective = "lambdarank" if family == "lgbm_lambdarank" else "regression"
            model, n = fit_lgbm(X[tr], train[tr], config, objective)
            rounds.append(n)
        oof[va] = score(model, X[va])
        vdf = train.loc[va, ["query_id", "product_id", "gain"]].assign(score=oof[va])
        per_fold.append(summarize(evaluate_queries(vdf, "score")))
        if keep_models:
            models.append(model)
    out = {"config": config, "summary": fold_summary(per_fold), "oof": oof}
    if rounds:
        out["best_rounds_per_fold"] = rounds
    if keep_models:
        out["models"] = models
    return out


def grid_search(paths: RunPaths, train: pd.DataFrame, family: str, features: list[str]) -> dict:
    """Evaluate every configuration of a family; keep the best by mean CV nDCG."""
    if family == "logreg":
        grid = [{"C": c, "class_weight": w} for c in LR_C_GRID for w in LR_CLASS_WEIGHTS]
    else:
        grid = LGBM_GRID
    runs = []
    for config in grid:
        t0 = time.time()
        r = cv_evaluate(paths, train, family, config, features)
        r["seconds"] = round(time.time() - t0, 1)
        print(f"  {family} {config}: nDCG {r['summary']['ndcg']['mean']:.4f} "
              f"rounds {r.get('best_rounds_per_fold', '-')} ({r['seconds']}s)")
        runs.append(r)
    best = max(runs, key=lambda r: r["summary"]["ndcg"]["mean"])
    return {"best": best, "grid": [{"config": r["config"], "ndcg": r["summary"]["ndcg"]["mean"],
                                    "ndcg_std": r["summary"]["ndcg"]["std"],
                                    "best_rounds_per_fold": r.get("best_rounds_per_fold"),
                                    "seconds": r["seconds"]} for r in runs]}


def choose_model(results: dict[str, dict]) -> tuple[str, str]:
    """Spec §3 rule: highest mean CV nDCG; within TIE_MARGIN prefer the simpler model."""
    best_score = max(results[f]["summary"]["ndcg"]["mean"] for f in MODEL_SIMPLICITY)
    for family in MODEL_SIMPLICITY:  # ordered simplest first
        s = results[family]["summary"]["ndcg"]["mean"]
        if best_score - s <= TIE_MARGIN:
            reason = (f"{MODEL_LABELS[family]} has mean CV nDCG {s:.4f}; the best is {best_score:.4f}. "
                      f"It is the simplest model within {TIE_MARGIN} of the best (rule of spec §3).")
            return family, reason
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- importance
def lgbm_importance(paths: RunPaths, train: pd.DataFrame, models: list, features: list[str]) -> pd.DataFrame:
    """Gain importance and mean |SHAP| (pred_contrib) on each fold's validation part,
    averaged over the 5 fold models."""
    gains, shaps = [], []
    for k, model in zip(sorted(train["fold"].unique()), models):
        X = fold_matrix(paths, train, int(k))[features]
        va = (train["fold"] == k).to_numpy()
        g = model.booster_.feature_importance(importance_type="gain")
        gains.append(g / g.sum())
        contrib = model.predict(X[va], pred_contrib=True)[:, :-1]  # last column = bias
        shaps.append(np.abs(contrib).mean(axis=0))
    imp = pd.DataFrame({"feature": features, "gain_share": np.mean(gains, axis=0),
                        "mean_abs_shap": np.mean(shaps, axis=0)})
    imp["group"] = imp["feature"].map(lambda f: FEATURE_DOCS[f][0])
    return imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def logreg_coefficients(models: list, features: list[str]) -> pd.DataFrame:
    """Standardised coefficients averaged over the 5 fold models."""
    coefs = np.array([m[-1].coef_[0] for m in models])
    out = pd.DataFrame({"feature": features, "coef_mean": coefs.mean(axis=0),
                        "coef_std": coefs.std(axis=0, ddof=1)})
    out["group"] = out["feature"].map(lambda f: FEATURE_DOCS[f][0])
    return out.reindex(out["coef_mean"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# ---------------------------------------------------------------- cross-encoder reference
def cross_encoder_reference(paths: RunPaths, train: pd.DataFrame, oof: pd.DataFrame,
                            n_queries: int) -> dict | None:
    """Optional zero-shot ms-marco cross-encoder on a CV subsample (reference only)."""
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(CROSS_ENCODER_NAME, device="cpu")
    except Exception as e:
        print(f"cross-encoder unavailable: {e}")
        return {"skipped": str(e)}
    rng = np.random.default_rng(SEED)
    qids = rng.choice(train["query_id"].unique(), size=min(n_queries, train["query_id"].nunique()),
                      replace=False)
    mask = train["query_id"].isin(set(qids)).to_numpy()
    sub = train.loc[mask, ["query_id", "product_id", "gain", "fold", "query", "title"]].copy()
    t0 = time.time()
    sub["cross_encoder"] = ce.predict(list(zip(sub["query"], sub["title"])), batch_size=64,
                                      show_progress_bar=False)
    seconds = time.time() - t0
    out = {"n_queries": int(len(qids)), "n_pairs": int(mask.sum()),
           "pairs_per_second": round(mask.sum() / seconds, 1), "methods": {}}
    for col in ["cross_encoder", "bm25_title", "encoder_cosine", "lgbm_lambdarank"]:
        if col == "cross_encoder":
            s = sub["cross_encoder"].to_numpy()
        elif col in oof:
            s = oof.loc[mask, col].to_numpy()
        else:
            continue
        per_fold = [summarize(evaluate_queries(sub[sub["fold"] == k].assign(score=s[sub["fold"].to_numpy() == k]), "score"))
                    for k in sorted(sub["fold"].unique())]
        out["methods"][col] = fold_summary(per_fold)
    return out


# ---------------------------------------------------------------- report
def fmt(s: dict, m: str) -> str:
    return f"{s[m]['mean']:.4f} ± {s[m]['std']:.4f}"


def write_report(paths: RunPaths, res: dict) -> str:
    base, models, abl = res["baselines"], res["models"], res["ablation"]
    rnd, bm = base["random"], base["bm25_title"]
    rows = [("baseline", BASELINE_LABELS[b], base[b]) for b in base] + \
           [("model", MODEL_LABELS[f], models[f]["summary"]) for f in MODEL_SIMPLICITY]
    md = ["# Report 2 — Validation (5-fold grouped CV on the training split)", "",
          f"All numbers: mean ± std across the 5 `GroupKFold` folds (grouped by `query_id`, "
          f"{res['n_train_queries']:,} training queries). The test split was not used. "
          f"Queries with ideal DCG 0 excluded: {bm['n_excluded_zero_idcg']}.",
          "" if res["semantic"] else "\n**Semantic features unavailable (encoder could not be loaded); "
          "semantic group and encoder baseline skipped.**\n", "",
          "## 1. Baselines and models", "",
          "| Type | Method | nDCG | nDCG@5 | Exact-P@1 | ΔnDCG vs random | ΔnDCG vs BM25-title |",
          "|---|---|---|---|---|---|---|"]
    for kind, name, s in rows:
        md.append(f"| {kind} | {name} | {fmt(s, 'ndcg')} | {fmt(s, 'ndcg5')} | {fmt(s, 'p1')} | "
                  f"{s['ndcg']['mean'] - rnd['ndcg']['mean']:+.4f} | {s['ndcg']['mean'] - bm['ndcg']['mean']:+.4f} |")
    md += ["", "Gains over random and BM25 are differences of fold means (nDCG points on a 0–1 scale). "
           "Random already scores high because Amazon pre-filtered the candidates (≈44 % of them are Exact).", ""]
    md += ["## 2. Hyper-parameter grids (mean CV nDCG)", ""]
    for f in MODEL_SIMPLICITY:
        md.append(f"**{MODEL_LABELS[f]}**")
        md.append("")
        for g in models[f]["grid"]:
            extra = f", rounds per fold {g['best_rounds_per_fold']}" if g.get("best_rounds_per_fold") else ""
            md.append(f"- `{json.dumps(g['config'])}`: {g['ndcg']:.4f} ± {g['ndcg_std']:.4f}{extra}")
        md.append("")
    md += ["## 3. Ablation (LambdaRank, best configuration, same folds)", "",
           "nDCG columns: mean ± std across folds, and a 95% bootstrap CI over training queries "
           "of the out-of-fold nDCG (1,000 resamples, same resamples for every row).", "",
           "| Feature set | #features | nDCG | 95% CI (OOF) | nDCG@5 | Exact-P@1 |", "|---|--:|---|---|---|---|"]
    for step in abl:
        ci = step["ndcg_ci"]
        md.append(f"| {step['name']} | {step['n_features']} | {fmt(step['summary'], 'ndcg')} | "
                  f"[{ci['lo']:.4f}, {ci['hi']:.4f}] | "
                  f"{fmt(step['summary'], 'ndcg5')} | {fmt(step['summary'], 'p1')} |")
    md += ["", "## 4. Feature importance", "",
           "LambdaRank (best config): share of total split gain and mean |SHAP| "
           "(LightGBM `pred_contrib=True`) on each fold's validation part, averaged over folds. Top 15:", "",
           "| Feature | Group | Gain share | Mean abs SHAP |", "|---|---|--:|--:|"]
    for r in res["importance"][:15]:
        md.append(f"| `{r['feature']}` | {r['group']} | {r['gain_share']:.3f} | {r['mean_abs_shap']:.4f} |")
    md += ["", "Logistic regression (best config), standardised coefficients averaged over folds. Top 15 by |coef|:", "",
           "| Feature | Group | Coefficient (mean ± std) |", "|---|---|---|"]
    for r in res["lr_coefficients"][:15]:
        md.append(f"| `{r['feature']}` | {r['group']} | {r['coef_mean']:+.3f} ± {r['coef_std']:.3f} |")
    ce = res.get("cross_encoder")
    if ce and "methods" in ce:
        md += ["", "## 5. Optional reference: zero-shot cross-encoder (not a candidate for selection)", "",
               f"`{CROSS_ENCODER_NAME}` on a subsample of {ce['n_queries']} training queries "
               f"({ce['n_pairs']:,} pairs; throughput in `cv_results.json`), compared with the "
               "out-of-fold scores of other methods on the same queries.", "",
               "| Method | nDCG | nDCG@5 | Exact-P@1 |", "|---|---|---|---|"]
        for name, s in ce["methods"].items():
            md.append(f"| {name} | {fmt(s, 'ndcg')} | {fmt(s, 'ndcg5')} | {fmt(s, 'p1')} |")
    chosen = res["chosen"]
    md += ["", "## 6. Chosen model (frozen before the test run)", "",
           f"**{MODEL_LABELS[chosen['family']]}** — {chosen['reason']}", "",
           f"Hyper-parameters: `{json.dumps(chosen['config'])}`"
           + (f"; final number of boosting rounds = median of the per-fold early-stopping rounds "
              f"= **{chosen['n_rounds']}**." if chosen.get("n_rounds") else "."), "",
           f"Features ({len(chosen['features'])}): " + ", ".join(f"`{f}`" for f in chosen["features"]), "",
           "`scripts/evaluate_test.py` reads the block below and refuses to run unless this file is "
           "committed and unchanged.", "", FREEZE_BEGIN, "```json",
           json.dumps(chosen, indent=2), "```", FREEZE_END, ""]
    text = "\n".join(md)
    (paths.results / "report_2_validation.md").write_text(text)
    return text


# ---------------------------------------------------------------- main
def ablation_sets(features: list[str]) -> list[tuple[str, list[str]]]:
    g = {k: [f for f in v if f in features] for k, v in FEATURE_GROUPS.items()}
    steps = [("lexical only", g["lexical"]),
             ("+ attribute + numeric + negation", g["lexical"] + g["attribute"] + g["numeric"] + g["negation"])]
    if g.get("semantic"):
        steps.append(("+ semantic", steps[-1][1] + g["semantic"]))
    steps.append(("all (+ quality, context, within-query)", features))
    return steps


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--cross-encoder-queries", type=int, default=500,
                        help="training queries for the optional cross-encoder reference (0 = skip)")
    args = parser.parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    train = load_train(paths)
    train["grade"] = train["esci_label"].map(GRADES)
    features = available_features(paths)
    baselines = json.loads((paths.results / "cv_baselines.json").read_text())["methods"]
    oof = pd.read_parquet(paths.processed / "oof_baselines.parquet")

    models: dict[str, dict] = {}
    full_runs: dict[str, dict] = {}
    for family in MODEL_SIMPLICITY:
        print(f"== {family}")
        gs = grid_search(paths, train, family, features)
        best = cv_evaluate(paths, train, family, gs["best"]["config"], features, keep_models=True)
        full_runs[family] = best
        oof[family] = best["oof"]
        models[family] = {"config": best["config"], "summary": best["summary"], "grid": gs["grid"],
                          "best_rounds_per_fold": best.get("best_rounds_per_fold")}

    print("== ablation")
    lr_cfg = models["lgbm_lambdarank"]["config"]
    ablation = []
    boot = None
    for i, (name, feats) in enumerate(ablation_sets(features)):
        r = full_runs["lgbm_lambdarank"] if feats == features else \
            cv_evaluate(paths, train, "lgbm_lambdarank", lr_cfg, feats)
        oof[f"ablation_{i}"] = r["oof"]
        per_query = evaluate_queries(train.assign(score=r["oof"]), "score").dropna()
        if boot is None:  # same resamples for every ablation step -> comparable CIs
            boot = bootstrap_indices(len(per_query))
        ablation.append({"name": name, "n_features": len(feats), "features": feats,
                         "summary": r["summary"],
                         "ndcg_ci": bootstrap_ci(per_query["ndcg"].to_numpy(), boot)})
        print(f"  {name}: {r['summary']['ndcg']['mean']:.4f}")

    importance = lgbm_importance(paths, train, full_runs["lgbm_lambdarank"]["models"], features)
    coefs = logreg_coefficients(full_runs["logreg"]["models"], features)
    family, reason = choose_model(models)
    chosen = {"family": family, "config": models[family]["config"], "features": features,
              "reason": reason}
    if models[family].get("best_rounds_per_fold"):
        chosen["n_rounds"] = int(np.median(models[family]["best_rounds_per_fold"]))
    # Phase 6 also reports the other two families, so their CV-selected settings are frozen too.
    chosen["all_models"] = {
        f: {"config": models[f]["config"],
            "n_rounds": int(np.median(models[f]["best_rounds_per_fold"]))
            if models[f].get("best_rounds_per_fold") else None}
        for f in MODEL_SIMPLICITY}

    ce = None
    if args.cross_encoder_queries > 0:
        print("== cross-encoder reference")
        ce = cross_encoder_reference(paths, train, oof, args.cross_encoder_queries)

    res = {"n_train_queries": int(train["query_id"].nunique()), "semantic": semantic_available(paths),
           "baselines": baselines, "models": models, "ablation": ablation,
           "importance": importance.to_dict(orient="records"),
           "lr_coefficients": coefs.to_dict(orient="records"), "chosen": chosen,
           "cross_encoder": ce, "metric_names": METRIC_NAMES}
    (paths.results / "cv_results.json").write_text(json.dumps(res, indent=2, default=str))
    oof.to_parquet(paths.processed / "oof_predictions.parquet", index=False)
    print(write_report(paths, res))
    log_runtime(paths, "train_cv", time.time() - t0)


if __name__ == "__main__":
    main()
