"""Reload the best Jena Climate model and report test MAE."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import build_datasets  # noqa: E402
from shared.utils import ensure_dir, set_seed  # noqa: E402


def main():
    set_seed(42)
    from tensorflow import keras

    artifacts = ensure_dir(HERE / "artifacts")
    ckpt = artifacts / "best_model.keras"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}\nrun train.py first")

    _, _, test_ds, _ = build_datasets()
    model = keras.models.load_model(ckpt)
    loss, mae = model.evaluate(test_ds, verbose=2)
    print(f"test loss = {loss:.4f}  test MAE = {mae:.4f}")
    (artifacts / "evaluation.json").write_text(
        json.dumps({"test_loss": float(loss), "test_mae": float(mae)}, indent=2)
    )


if __name__ == "__main__":
    main()
