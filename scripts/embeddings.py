"""Sentence embeddings for queries, titles and title + bullets, computed once per unique text.

Encoder: all-MiniLM-L6-v2 (released 2021, before ESCI) - see spec §4.4.
The encoder is pretrained and frozen, so nothing is learned from our data here.
Phase 3 embeds the training split only; the test split is embedded in Phase 6.

Cache layout (in data/processed/):
    emb_<name>.npy           float32 matrix, one L2-normalised row per text
    emb_<name>_index.parquet text -> row number
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from config import (ENCODER_BATCH, ENCODER_NAME, TITLE_BULLETS_MAX_TOKENS, RunPaths,
                    base_arg_parser, get_paths, log_runtime)

CACHES = ("query", "title", "title_bullets")


def title_bullets_text(title: str, bullets: str) -> str:
    return f"{title}. {bullets}" if bullets else title


def semantic_available(paths: RunPaths) -> bool:
    """False if the encoder could not be loaded (fallback: skip the semantic group)."""
    return not (paths.processed / "semantic_unavailable.json").exists()


def load_cache(paths: RunPaths, name: str) -> tuple[np.ndarray, dict[str, int]]:
    emb_file = paths.processed / f"emb_{name}.npy"
    idx_file = paths.processed / f"emb_{name}_index.parquet"
    if not emb_file.exists():
        return np.zeros((0, 384), dtype=np.float32), {}
    idx = pd.read_parquet(idx_file)
    return np.load(emb_file), dict(zip(idx["text"], idx["row"]))


def save_cache(paths: RunPaths, name: str, emb: np.ndarray, index: dict[str, int]) -> None:
    np.save(paths.processed / f"emb_{name}.npy", emb.astype(np.float32))
    pd.DataFrame({"text": list(index), "row": list(index.values())}).to_parquet(
        paths.processed / f"emb_{name}_index.parquet", index=False
    )


def update_cache(paths: RunPaths, name: str, texts: list[str], model) -> dict:
    """Embed the texts that are not cached yet and append them to the cache."""
    emb, index = load_cache(paths, name)
    missing = sorted({t for t in texts if t not in index})
    if not missing:
        return {"new": 0, "seconds": 0.0}
    model.max_seq_length = TITLE_BULLETS_MAX_TOKENS if name == "title_bullets" else 256
    t0 = time.time()
    new = model.encode(missing, batch_size=ENCODER_BATCH, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    seconds = time.time() - t0
    start = len(index)
    for i, t in enumerate(missing):
        index[t] = start + i
    save_cache(paths, name, np.vstack([emb, new]), index)
    return {"new": len(missing), "seconds": round(seconds, 1),
            "texts_per_second": round(len(missing) / max(seconds, 1e-9), 1)}


def lookup(paths: RunPaths, name: str, texts: pd.Series) -> np.ndarray:
    """Embedding rows for a column of texts (all must be cached)."""
    emb, index = load_cache(paths, name)
    return emb[texts.map(index).to_numpy()]


def cosine_features(paths: RunPaths, df: pd.DataFrame) -> pd.DataFrame:
    """Cosine of query vs title and query vs title + bullets (embeddings are unit-norm)."""
    q = lookup(paths, "query", df["query"])
    t = lookup(paths, "title", df["title"])
    tb = lookup(paths, "title_bullets", df["title_bullets_text"])
    return pd.DataFrame({
        "cos_title": np.einsum("ij,ij->i", q, t),
        "cos_title_bullets": np.einsum("ij,ij->i", q, tb),
    }, index=df.index)


def embed_split(paths: RunPaths, split: str) -> None:
    """Make sure every query / title / title+bullets text of `split` is cached."""
    pairs = pd.read_parquet(paths.processed / "pairs.parquet",
                            columns=["query", "split", "title", "bullets"])
    pairs = pairs[pairs["split"] == split]
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(ENCODER_NAME, device="cpu")
    except Exception as e:  # blocked host, missing package, ...
        print(f"encoder unavailable ({e}); semantic features will be skipped")
        (paths.processed / "semantic_unavailable.json").write_text(json.dumps({"error": str(e)}))
        return
    texts = {
        "query": pairs["query"].unique().tolist(),
        "title": pairs["title"].unique().tolist(),
        "title_bullets": [title_bullets_text(t, b) for t, b in
                          pairs[["title", "bullets"]].drop_duplicates().itertuples(index=False)],
    }
    log_file = paths.results / "embedding_log.json"
    log = json.loads(log_file.read_text()) if log_file.exists() else {}
    for name in CACHES:
        print(f"[{split}] {name}: {len(texts[name]):,} unique texts")
        log[f"{split}/{name}"] = update_cache(paths, name, texts[name], model)
        print(log[f"{split}/{name}"])
    log_file.write_text(json.dumps(log, indent=2))


def main() -> None:
    parser = base_arg_parser(__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()
    paths = get_paths(args.sample)
    t0 = time.time()
    embed_split(paths, args.split)
    log_runtime(paths, f"embeddings_{args.split}", time.time() - t0)


if __name__ == "__main__":
    main()
