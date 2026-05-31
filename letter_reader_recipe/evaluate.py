"""Reload checkpoints written by train.py and re-evaluate on the test split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_arrays  # noqa: E402
from shared.utils import ensure_dir, set_seed  # noqa: E402


def main():
    set_seed(42)
    from tensorflow import keras

    artifacts = ensure_dir(HERE / "artifacts")
    for name in ("best_recipe.keras", "final_recipe.keras"):
        ckpt = artifacts / name
        if not ckpt.exists():
            print(f"missing {ckpt} - run train.py first")
            continue
        _, _, x_te, y_te = load_arrays()
        model = keras.models.load_model(ckpt)
        loss, acc = model.evaluate(x_te, y_te, verbose=2)
        print(f"{name}: loss={loss:.4f}  acc={acc:.4f}")
        (artifacts / f"eval_{ckpt.stem}.json").write_text(
            json.dumps({"loss": float(loss), "acc": float(acc)}, indent=2)
        )


if __name__ == "__main__":
    main()
