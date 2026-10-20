"""Project-wide constants: paths, seed, relevance gains, model grids.

Every other script imports from here so that a value is defined exactly once.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SITE_DATA_DIR = ROOT / "site" / "data"

# ---------------------------------------------------------------- data
DATA_URLS = {
    "examples": "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/shopping_queries_dataset_examples.parquet",
    "products": "https://media.githubusercontent.com/media/amazon-science/esci-data/main/"
    "shopping_queries_dataset/shopping_queries_dataset_products.parquet",
    "sources": "https://raw.githubusercontent.com/amazon-science/esci-data/main/"
    "shopping_queries_dataset/shopping_queries_dataset_sources.csv",
}
LOCALE = "us"

# Expected sizes of the small US subset (spec §2.1); we warn if off by > 1 %.
EXPECTED_SIZES = {
    "total": {"queries": 29_844, "pairs": 601_354},
    "train": {"queries": 20_888, "pairs": 419_653},
    "test": {"queries": 8_956, "pairs": 181_701},
}

# ---------------------------------------------------------------- labels
SEED = 42
# Official ESCI gains (spec §2.2). Fixed for the whole project.
GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
# Integer grades for LambdaRank; label_gain[grade] reproduces GAINS.
GRADES = {"I": 0, "C": 1, "S": 2, "E": 3}
LABEL_GAIN = [0.0, 0.01, 0.1, 1.0]

# ---------------------------------------------------------------- text
TEXT_CHAR_BUDGET = 1000  # max characters kept for description / bullets

# ---------------------------------------------------------------- BM25
BM25_K1 = 1.5
BM25_B = 0.75

# ---------------------------------------------------------------- encoder
# Released 2021, before ESCI (June 2022): cannot have been trained on it.
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
ENCODER_BATCH = 256
TITLE_BULLETS_MAX_TOKENS = 128  # truncation for the longer "title + bullets" text

# ---------------------------------------------------------------- CV + models
N_FOLDS = 5
INNER_VALID_FRACTION = 0.2  # inner grouped split for early stopping
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]
LR_CLASS_WEIGHTS = [None, "balanced"]
LGBM_GRID = [
    {"num_leaves": nl, "min_data_in_leaf": md} for nl in (15, 31, 63) for md in (20, 100)
]
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_ROUNDS = 2000
LGBM_EARLY_STOPPING = 50
LGBM_THREADS = 4
TIE_MARGIN = 0.002  # model-selection rule (spec §3)
MODEL_SIMPLICITY = ["logreg", "lgbm_pointwise", "lgbm_lambdarank"]

N_BOOTSTRAP = 1000
N_RANDOM_SEEDS = 20
NEAR_DUP_COSINE = 0.90


# ---------------------------------------------------------------- run paths
@dataclass(frozen=True)
class RunPaths:
    """Where a run reads and writes. Sample (smoke-test) runs are kept separate."""

    processed: Path
    results: Path
    figures: Path
    sample: int | None

    @property
    def is_sample(self) -> bool:
        return self.sample is not None


def get_paths(sample: int | None = None) -> RunPaths:
    if sample is None:
        processed = ROOT / "data" / "processed"
        results = ROOT / "results"
    else:
        processed = ROOT / "data" / "processed" / f"sample_{sample}"
        results = ROOT / "data" / "sample_results" / f"sample_{sample}"
    figures = results / "figures"
    for d in (processed, results, figures):
        d.mkdir(parents=True, exist_ok=True)
    return RunPaths(processed, results, figures, sample)


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    """Every script accepts --sample N (smoke test on N training queries)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--sample", type=int, default=None,
                        help="run on N training queries only (fast smoke test)")
    return parser


def log_runtime(paths: RunPaths, phase: str, seconds: float) -> None:
    """Append the runtime of one phase to results/runtime.json."""
    f = paths.results / "runtime.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data[phase] = {"seconds": round(seconds, 1), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    f.write_text(json.dumps(data, indent=2))
