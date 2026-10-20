"""Phase 7 - optional extra analyses, each clearly labelled.

1. Learning curve (training split CV only): CV nDCG of LambdaRank vs fraction of
   training queries (10, 25, 50, 100 %).
2. Calibration of the logistic-regression probabilities (out-of-fold, training split).
3. Sensitivity to gains (POST-HOC, on the frozen test predictions): rescore with
   alternative gains. The models are not refit; only the metric changes.

Writes results/extra_analyses.json, results/report_4_extra.md and figures.
"""
from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import GRADES, N_RANDOM_SEEDS, SEED, base_arg_parser, get_paths, log_runtime
from features import fold_matrix, load_train
from metrics import bootstrap_ci, bootstrap_indices, evaluate_queries, summarize
from train_cv import MODEL_LABELS, fit_lgbm, score

FRACTIONS = [0.10, 0.25, 0.50, 1.00]
ALT_GAINS = {
    "official (E=1, S=0.1, C=0.01)": {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0},
    "lenient (E=1, S=0.5, C=0.25)": {"E": 1.0, "S": 0.5, "C": 0.25, "I": 0.0},
    "exact only (E=1, others 0)": {"E": 1.0, "S": 0.0, "C": 0.0, "I": 0.0},
}


def learning_curve(paths, train: pd.DataFrame, config: dict, features: list[str]) -> list[dict]:
    """Subsample each fold's training queries (seeded); validate on the full fold."""
    rng = np.random.default_rng(SEED)
    out = []
    for frac in FRACTIONS:
        per_fold = []
        for k in sorted(train["fold"].unique()):
            X = fold_matrix(paths, train, int(k))[features]
            tr_q = train.loc[train["fold"] != k, "query_id"].unique()
            keep = set(rng.choice(tr_q, size=max(int(frac * len(tr_q)), 2), replace=False))
            tr = (train["fold"] != k).to_numpy() & train["query_id"].isin(keep).to_numpy()
            va = (train["fold"] == k).to_numpy()
            model, _ = fit_lgbm(X[tr], train[tr], config, "lambdarank")
            vdf = train.loc[va, ["query_id", "product_id", "gain"]].assign(score=score(model, X[va]))
            per_fold.append(summarize(evaluate_queries(vdf, "score"))["ndcg"])
        out.append({"fraction": frac, "ndcg_mean": float(np.mean(per_fold)),
                    "ndcg_std": float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else 0.0,
                    "n_train_queries_per_fold": int(frac * len(tr_q))})
        print(f"learning curve {frac:.0%}: {out[-1]['ndcg_mean']:.4f}")
    return out


def calibration(oof: pd.DataFrame, train: pd.DataFrame, n_bins: int = 10) -> dict:
    """Reliability of out-of-fold LR probabilities of 'Exact'."""
    p = oof["logreg"].to_numpy()
    y = (train["esci_label"] == "E").to_numpy().astype(float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = [{"bin_lo": b / n_bins, "bin_hi": (b + 1) / n_bins, "n": int((bins == b).sum()),
             "mean_pred": float(p[bins == b].mean()), "frac_exact": float(y[bins == b].mean())}
            for b in range(n_bins) if (bins == b).any()]
    return {"bins": rows, "brier": float(np.mean((p - y) ** 2)),
            "brier_base_rate": float(np.mean((y.mean() - y) ** 2))}


def gain_sensitivity(pred: pd.DataFrame, methods: list[str]) -> dict:
    out = {}
    for name, gains in ALT_GAINS.items():
        d = pred.assign(gain=pred["esci_label"].map(gains))
        pq = {m: evaluate_queries(d, m).set_index("query_id")["ndcg"] for m in methods}
        rnd = [evaluate_queries(d, f"random_{s}").set_index("query_id")["ndcg"] for s in range(N_RANDOM_SEEDS)]
        pq["random"] = sum(rnd) / len(rnd)
        valid = pq["bm25_title"].dropna().index
        idx = bootstrap_indices(len(valid))
        out[name] = {"ndcg": {m: bootstrap_ci(v.loc[valid].to_numpy(), idx) for m, v in pq.items()},
                     "model_minus_bm25_title": bootstrap_ci((pq["model"] - pq["bm25_title"]).loc[valid].to_numpy(), idx),
                     "n_queries": int(len(valid))}
    return out


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    cv = json.loads((paths.results / "cv_results.json").read_text())
    features = cv["chosen"]["features"]
    train = load_train(paths)
    train["grade"] = train["esci_label"].map(GRADES)
    oof = pd.read_parquet(paths.processed / "oof_predictions.parquet")
    res: dict = {"labels": {"learning_curve": "training-split CV", "calibration": "training-split CV (out-of-fold)",
                            "gain_sensitivity": "POST-HOC on frozen test predictions"}}

    res["learning_curve"] = learning_curve(paths, train, cv["models"]["lgbm_lambdarank"]["config"], features)
    res["learning_curve_bm25_title"] = cv["baselines"]["bm25_title"]["ndcg"]["mean"]
    res["calibration"] = calibration(oof, train)
    pred_file = paths.results / "test_predictions.parquet"
    if pred_file.exists():
        pred = pd.read_parquet(pred_file)
        methods = [m for m in ["bm25_title", "bm25_all", "encoder_cosine", "logreg", "lgbm_pointwise",
                               "lgbm_lambdarank", "model"] if m in pred]
        res["gain_sensitivity"] = gain_sensitivity(pred, methods)

    # figures
    lc = res["learning_curve"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar([r["fraction"] * 100 for r in lc], [r["ndcg_mean"] for r in lc],
                yerr=[r["ndcg_std"] for r in lc], marker="o", capsize=3, label="LambdaRank (CV)")
    ax.axhline(res["learning_curve_bm25_title"], ls="--", color="grey", label="BM25 title (CV)")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
    ax.set_xlabel("share of training queries")
    ax.set_ylabel("CV nDCG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "learning_curve.png", dpi=120)
    plt.close(fig)
    cb = res["calibration"]["bins"]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey")
    ax.plot([b["mean_pred"] for b in cb], [b["frac_exact"] for b in cb], marker="o")
    ax.set_xlabel("predicted P(Exact)")
    ax.set_ylabel("observed share Exact")
    ax.set_title("Logistic regression reliability (OOF)")
    fig.tight_layout()
    fig.savefig(paths.figures / "calibration_lr.png", dpi=120)
    plt.close(fig)

    (paths.results / "extra_analyses.json").write_text(json.dumps(res, indent=2))
    md = ["# Report 4 — Extra analyses (Phase 7, optional)", "",
          "Each analysis is labelled with the data it uses. None of them influenced model selection.", "",
          "## 1. Learning curve — *training-split CV*", "",
          "LambdaRank with the CV-selected configuration, trained on a seeded subsample of each fold's "
          "training queries and validated on the full validation fold. Fitted features (BM25 statistics, brand "
          "vocabulary) still use the fold's full training part: only the number of *labelled* queries varies.", "",
          "| Share of training queries | Queries per fold | CV nDCG (mean ± std) |", "|---|--:|---|"]
    md += [f"| {r['fraction']:.0%} | {r['n_train_queries_per_fold']:,} | {r['ndcg_mean']:.4f} ± {r['ndcg_std']:.4f} |" for r in lc]
    md += ["", f"BM25-title CV nDCG for reference: {res['learning_curve_bm25_title']:.4f}.", "",
           "![learning curve](figures/learning_curve.png)", "",
           "## 2. Calibration of logistic-regression probabilities — *training-split CV, out-of-fold*", "",
           f"Brier score {res['calibration']['brier']:.4f} (always predicting the base rate: "
           f"{res['calibration']['brier_base_rate']:.4f}).", "",
           "| Predicted bin | Pairs | Mean predicted | Observed share Exact |", "|---|--:|--:|--:|"]
    md += [f"| {b['bin_lo']:.1f}–{b['bin_hi']:.1f} | {b['n']:,} | {b['mean_pred']:.3f} | {b['frac_exact']:.3f} |" for b in cb]
    md += ["", "![calibration](figures/calibration_lr.png)", ""]
    if "gain_sensitivity" in res:
        md += ["## 3. Sensitivity to gains — *POST-HOC, frozen test predictions*", "",
               "Same rankings, different gain values in the metric. Labelled post-hoc: chosen after seeing "
               "the test results, so it is descriptive only.", "",
               "| Gains | Random | BM25 title | Chosen model | Model − BM25 title [95% CI] |", "|---|---|---|---|---|"]
        for name, g in res["gain_sensitivity"].items():
            n = g["ndcg"]
            d = g["model_minus_bm25_title"]
            md.append(f"| {name} | {n['random']['mean']:.4f} | {n['bm25_title']['mean']:.4f} | {n['model']['mean']:.4f} | "
                      f"{d['mean']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}] |")
        md.append("")
    (paths.results / "report_4_extra.md").write_text("\n".join(md))
    print("\n".join(md))
    log_runtime(paths, "extra_analyses", time.time() - t0)


if __name__ == "__main__":
    main()
