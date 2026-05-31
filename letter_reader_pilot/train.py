"""Train logistic baseline vs shallow MLP on EMNIST Letters.

Experiments produced when this script runs (everything ends up under
``artifacts/``):

- ``logistic_baseline.json``   logistic regression accuracy on flattened pixels
- ``hidden_units_sweep.csv``   shallow MLP test accuracy for hidden_units
  ``in {32, 64, 128, 256, 512}`` averaged over ``--n-seeds`` seeds
- ``batch_size_sweep.csv``     test accuracy for batch_size ``in {32, 128, 512}``
- ``lr_sweep.csv``             test accuracy for lr ``in {1e-2, 1e-3, 1e-4}``
- ``baseline_vs_mlp.csv``      head-to-head logistic vs MLP table
- ``*.png``                    bar charts of each sweep (with std error bars)
- ``shallow_mlp.keras``        final shallow MLP trained on the best config

Use ``--quick`` to shrink the sweep grid to a single point per axis for
smoke-testing without paying the full sweep cost.

Usage:
    python letter_reader_pilot/train.py
    python letter_reader_pilot/train.py --n-seeds 3 --epochs 8
    python letter_reader_pilot/train.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_numpy, load_tf_datasets  # noqa: E402
from model import build_model  # noqa: E402
from shared.utils import (  # noqa: E402
    aggregate_mean_std,
    bar_chart,
    ensure_dir,
    multi_seed,
    plot_history,
    save_metric_table,
    set_seed,
    time_block,
)


def train_logistic_baseline(seed: int = 42,
                            max_train: int = 20_000,
                            max_test: int = 5_000) -> float:
    """Quick sklearn SGD logistic regression on flattened pixels.

    Capped training set keeps this honest as a baseline and CPU-friendly.
    """
    from sklearn.linear_model import SGDClassifier

    x_tr, y_tr, x_te, y_te = load_numpy(max_train=max_train, max_test=max_test)
    clf = SGDClassifier(loss="log_loss", max_iter=20, tol=1e-3, random_state=seed)
    clf.fit(x_tr, y_tr)
    acc = float(clf.score(x_te, y_te))
    print(f"[logistic baseline] seed={seed} test accuracy = {acc:.4f}")
    return acc


def _train_one(hidden_units: int,
               batch_size: int,
               learning_rate: float,
               epochs: int,
               seed: int) -> dict:
    set_seed(seed)
    train_ds, val_ds, test_ds = load_tf_datasets(batch_size=batch_size, seed=seed)
    model = build_model(hidden_units=hidden_units, learning_rate=learning_rate)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"test_acc": float(acc), "test_loss": float(loss),
            "hidden_units": hidden_units, "batch_size": batch_size,
            "learning_rate": learning_rate, "epochs": epochs}


def _sweep(axis_name: str,
           values: list,
           fixed: dict,
           epochs: int,
           n_seeds: int,
           artifacts: Path) -> list[dict]:
    rows: list[dict] = []
    for v in values:
        cfg = dict(fixed)
        cfg[axis_name] = v
        per_seed = multi_seed(
            lambda seed, cfg=cfg: _train_one(epochs=epochs, seed=seed, **cfg),
            n_seeds=n_seeds,
        )
        agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
        row = {axis_name: v, **agg, "n_seeds": n_seeds}
        rows.append(row)
        print(f"[sweep:{axis_name}={v}] acc={agg.get('test_acc_mean', float('nan')):.4f} "
              f"+/- {agg.get('test_acc_std', float('nan')):.4f}")
    save_metric_table(rows, artifacts / f"{axis_name}_sweep.csv")
    bar_chart(
        [str(r[axis_name]) for r in rows],
        [r["test_acc_mean"] for r in rows],
        errors=[r["test_acc_std"] for r in rows],
        out_path=artifacts / f"{axis_name}_sweep.png",
        title=f"Shallow MLP sweep over {axis_name}",
        ylabel="test accuracy",
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="shrink every sweep to a single point (smoke test)")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    set_seed(args.seed)

    # ---- 1. logistic baseline (multi-seed, capped train) ----
    baseline_rows: list[dict] = []
    if not args.skip_baseline:
        for i in range(args.n_seeds):
            seed = args.seed + i
            acc = train_logistic_baseline(seed=seed)
            baseline_rows.append({"seed": seed, "test_acc": acc})
        save_metric_table(baseline_rows, artifacts / "logistic_baseline.csv")
        agg = aggregate_mean_std(baseline_rows, ["test_acc"])
        (artifacts / "logistic_baseline.json").write_text(
            json.dumps({"per_seed": baseline_rows, **agg}, indent=2)
        )

    # ---- 2. sweeps ----
    hu_values = [args.hidden_units] if args.quick else [32, 64, 128, 256, 512]
    bs_values = [args.batch_size] if args.quick else [32, 128, 512]
    lr_values = [args.learning_rate] if args.quick else [1e-2, 1e-3, 1e-4]

    fixed = dict(hidden_units=args.hidden_units,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate)

    with time_block("hidden_units sweep"):
        hu_rows = _sweep("hidden_units", hu_values, fixed, args.epochs,
                         args.n_seeds, artifacts)
    with time_block("batch_size sweep"):
        bs_rows = _sweep("batch_size", bs_values, fixed, args.epochs,
                         args.n_seeds, artifacts)
    with time_block("learning_rate sweep"):
        lr_rows = _sweep("learning_rate", lr_values, fixed, args.epochs,
                         args.n_seeds, artifacts)

    # ---- 3. baseline vs MLP head-to-head ----
    h2h_rows: list[dict] = []
    if baseline_rows:
        b_agg = aggregate_mean_std(baseline_rows, ["test_acc"])
        h2h_rows.append({
            "model": "logistic_regression",
            "test_acc_mean": b_agg.get("test_acc_mean"),
            "test_acc_std": b_agg.get("test_acc_std"),
            "n_seeds": args.n_seeds,
        })
    # pick the central hidden-units run as a representative MLP point
    representative_hu = args.hidden_units
    mlp_pick = next((r for r in hu_rows if r["hidden_units"] == representative_hu),
                    hu_rows[len(hu_rows) // 2])
    h2h_rows.append({
        "model": f"shallow_mlp(hu={mlp_pick['hidden_units']})",
        "test_acc_mean": mlp_pick.get("test_acc_mean"),
        "test_acc_std": mlp_pick.get("test_acc_std"),
        "n_seeds": args.n_seeds,
    })
    save_metric_table(h2h_rows, artifacts / "baseline_vs_mlp.csv")
    bar_chart(
        [r["model"] for r in h2h_rows],
        [r["test_acc_mean"] or 0.0 for r in h2h_rows],
        errors=[r["test_acc_std"] or 0.0 for r in h2h_rows],
        out_path=artifacts / "baseline_vs_mlp.png",
        title="Logistic baseline vs shallow MLP (EMNIST Letters)",
        ylabel="test accuracy",
    )

    # ---- 4. final model on chosen config + saved checkpoint ----
    set_seed(args.seed)
    train_ds, val_ds, test_ds = load_tf_datasets(batch_size=args.batch_size, seed=args.seed)
    model = build_model(hidden_units=args.hidden_units,
                        learning_rate=args.learning_rate)
    model.summary()
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"[final shallow MLP] test accuracy = {test_acc:.4f}")
    model.save(artifacts / "shallow_mlp.keras")
    plot_history(history.history, artifacts / "history.png",
                 title="Shallow MLP on EMNIST Letters")

    summary = {
        "seed": args.seed,
        "n_seeds": args.n_seeds,
        "epochs": args.epochs,
        "final_shallow_mlp": {
            "hidden_units": args.hidden_units,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
        },
        "sweeps": {
            "hidden_units": hu_rows,
            "batch_size": bs_rows,
            "learning_rate": lr_rows,
        },
        "baseline_rows": baseline_rows,
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
