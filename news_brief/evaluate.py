"""Re-score the summarization model on a fresh eval slice with ROUGE."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_summarization_split  # noqa: E402
from model import DEFAULT_MODEL, build_pipeline  # noqa: E402
from shared.utils import ensure_dir, set_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-eval", type=int, default=50)
    args = parser.parse_args()

    set_seed(42)
    artifacts = ensure_dir(HERE / "artifacts")

    _, eval_rows, ds_name = load_summarization_split()
    pipe = build_pipeline(args.model)
    sample = eval_rows[:args.max_eval]
    preds, refs = [], []
    for row in sample:
        out = pipe(row["document"], max_length=80, min_length=15,
                   do_sample=False, truncation=True)
        preds.append(out[0]["summary_text"].strip())
        refs.append(row["summary"])

    import evaluate as hf_eval

    rouge = hf_eval.load("rouge").compute(predictions=preds, references=refs)
    print("ROUGE:", rouge)
    (artifacts / "evaluation.json").write_text(
        json.dumps({"dataset": ds_name, "model": args.model, "rouge": rouge},
                   indent=2)
    )


if __name__ == "__main__":
    main()
