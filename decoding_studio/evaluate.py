"""Print the most recent per-strategy aggregates produced by train.py."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from shared.utils import ensure_dir  # noqa: E402


def main():
    artifacts = ensure_dir(HERE / "artifacts")
    samples = artifacts / "samples.txt"
    summary = artifacts / "strategy_summary.csv"
    if not samples.exists() or not summary.exists():
        raise SystemExit("run train.py first")

    print(samples.read_text())

    print("---- per-strategy summary (avg over prompts) ----")
    with summary.open() as f:
        for row in csv.DictReader(f):
            print(f"{row['strategy']:20s} "
                  f"d1={float(row['distinct_1']):.3f} "
                  f"d2={float(row['distinct_2']):.3f} "
                  f"d3={float(row['distinct_3']):.3f} "
                  f"len={float(row['mean_token_length']):.1f} "
                  f"self-bleu={float(row['self_bleu']):.3f}")

    js = artifacts / "samples.json"
    if js.exists():
        data = json.loads(js.read_text())
        print(f"model: {data['model']}; prompts: {len(data['prompts'])}; "
              f"strategies: {len(data['strategies'])}")


if __name__ == "__main__":
    main()
