"""Run the whole pipeline, from download to site data.

    python run_all.py                 # full run (stops before the test if Report 2 is not committed)
    python run_all.py --sample 500    # smoke test on 500 training queries (+250 test queries)
    python run_all.py --from test     # resume from a phase (after committing Report 2)

Each script logs its own runtime to results/runtime.json.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

PHASES = [
    ("download", ["scripts/download_data.py"]),
    ("build", ["scripts/build_dataset.py"]),
    ("audit", ["scripts/audit.py"]),
    ("embed", ["scripts/embeddings.py", "--split", "train"]),
    ("features", ["scripts/features.py"]),
    ("baselines", ["scripts/baselines.py"]),
    ("train", ["scripts/train_cv.py"]),
    ("test", ["scripts/evaluate_test.py"]),
    ("extra", ["scripts/extra_analyses.py"]),
    ("site", ["scripts/export_site_data.py"]),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="smoke test on N training queries")
    parser.add_argument("--from", dest="start", default=PHASES[0][0], choices=[p for p, _ in PHASES])
    args = parser.parse_args()
    names = [p for p, _ in PHASES]
    t_all = time.time()
    for name, cmd in PHASES[names.index(args.start):]:
        cmd = [sys.executable, *cmd] + (["--sample", str(args.sample)] if args.sample else [])
        if args.sample and name == "train":
            cmd += ["--cross-encoder-queries", "20"]
        print(f"\n=== {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            if name == "test" and not args.sample:
                print("\nThe test phase refused or failed. If Report 2 is not committed yet: review it, "
                      "`git add results/report_2_validation.md && git commit`, then "
                      "`python run_all.py --from test`.")
            sys.exit(result.returncode)
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    print(f"\nAll phases done in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
