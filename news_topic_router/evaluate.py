"""Reload the embedding RNN and re-evaluate on AG News test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import build_vectorizer, load_raw_datasets, vectorize_ds  # noqa: E402
from shared.utils import ensure_dir, set_seed  # noqa: E402


def main():
    set_seed(42)
    from tensorflow import keras

    artifacts = ensure_dir(HERE / "artifacts")
    ckpt = artifacts / "embedding_rnn.keras"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}\nrun train.py first")

    raw_train, _, raw_test = load_raw_datasets()
    vectorizer = build_vectorizer(raw_train)
    test_ds = vectorize_ds(raw_test, vectorizer)
    model = keras.models.load_model(ckpt)
    loss, acc = model.evaluate(test_ds, verbose=2)
    print(f"test loss = {loss:.4f}  test accuracy = {acc:.4f}")
    (artifacts / "evaluation.json").write_text(
        json.dumps({"test_loss": float(loss), "test_acc": float(acc)}, indent=2)
    )


if __name__ == "__main__":
    main()
