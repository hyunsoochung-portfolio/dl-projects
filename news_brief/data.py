"""Summarization dataset loader — XSum (one-sentence summaries of BBC articles)."""
from __future__ import annotations

from typing import Dict, List, Tuple


def load_summarization_split(max_train: int = 2000,
                             max_eval: int = 200) -> Tuple[List[Dict],
                                                           List[Dict],
                                                           str]:
    """Return (train_rows, eval_rows, dataset_name).

    Each row has 'document' and 'summary' keys.
    """
    from datasets import load_dataset

    train_ds = load_dataset("EdinburghNLP/xsum",
                            split=f"train[:{max_train}]",
                            trust_remote_code=True)
    eval_ds = load_dataset("EdinburghNLP/xsum",
                           split=f"validation[:{max_eval}]",
                           trust_remote_code=True)

    train_rows = [{"document": r["document"], "summary": r["summary"]} for r in train_ds]
    eval_rows = [{"document": r["document"], "summary": r["summary"]} for r in eval_ds]
    return train_rows, eval_rows, "xsum"
