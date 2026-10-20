"""Download the ESCI files into data/raw/ and record checksums in MANIFEST.json.

Downloads resume if interrupted (HTTP Range requests on a .part file).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from config import DATA_URLS, RAW_DIR, base_arg_parser, get_paths, log_runtime

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, retries: int = 5) -> None:
    """Download url to dest, resuming from dest.part if it exists."""
    if dest.exists():
        print(f"  already present: {dest.name}")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                if r.status_code == 416:  # range not satisfiable: already complete
                    break
                r.raise_for_status()
                mode = "ab" if r.status_code == 206 else "wb"  # server may ignore Range
                with part.open(mode) as f:
                    for block in r.iter_content(CHUNK):
                        f.write(block)
            break
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not download {url}")
    part.rename(dest)


def main() -> None:
    args = base_arg_parser(__doc__).parse_args()
    paths = get_paths(args.sample)  # raw data is shared; --sample only affects runtime log
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, url in DATA_URLS.items():
        dest = RAW_DIR / Path(url).name
        print(f"{name}: {url}")
        download(url, dest)
        if name not in manifest or manifest[name].get("bytes") != dest.stat().st_size:
            manifest[name] = {
                "file": dest.name,
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": sha256_of(dest),
                "downloaded": time.strftime("%Y-%m-%d"),
            }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    log_runtime(paths, "download_data", time.time() - t0)


if __name__ == "__main__":
    main()
