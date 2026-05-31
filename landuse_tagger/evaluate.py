"""Evaluate EuroSAT CNN: per-class precision/recall/F1 + confusion matrix.

Outputs under ``artifacts/``:
- ``evaluation.json``      overall + per-class metrics in machine-readable form
- ``per_class_metrics.csv``  one row per class (precision / recall / F1 / support / accuracy)
- ``per_class_accuracy.png`` bar chart of per-class accuracy
- ``confusion_matrix.png``   confusion matrix heatmap
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import CLASS_NAMES, load_tf_datasets  # noqa: E402
from shared.utils import bar_chart, ensure_dir, set_seed  # noqa: E402


def plot_confusion_matrix(cm: np.ndarray, classes, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("EuroSAT confusion matrix")
    thresh = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)


def main():
    set_seed(42)
    from sklearn.metrics import (
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from tensorflow import keras

    artifacts = ensure_dir(HERE / "artifacts")
    ckpt = artifacts / "eurosat_cnn.keras"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}\nrun train.py first")

    _, _, test_ds = load_tf_datasets()
    model = keras.models.load_model(ckpt)
    loss, acc = model.evaluate(test_ds, verbose=2)
    print(f"overall test accuracy = {acc:.4f}")

    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        probs = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(y_batch.numpy().tolist())
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(CLASS_NAMES))))
    plot_confusion_matrix(cm, CLASS_NAMES, artifacts / "confusion_matrix.png")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        n = int(support[i])
        correct = int(cm[i, i])
        per_class_acc = float(correct / n) if n else 0.0
        per_class.append({
            "class": name, "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]),
            "support": n, "accuracy": per_class_acc,
        })

    # CSV
    csv_path = artifacts / "per_class_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()))
        w.writeheader()
        w.writerows(per_class)

    bar_chart(
        [r["class"] for r in per_class],
        [r["accuracy"] for r in per_class],
        out_path=artifacts / "per_class_accuracy.png",
        title="Per-class test accuracy", ylabel="accuracy",
    )

    (artifacts / "evaluation.json").write_text(
        json.dumps({"test_loss": float(loss), "test_acc": float(acc),
                    "per_class": per_class}, indent=2)
    )


if __name__ == "__main__":
    main()
