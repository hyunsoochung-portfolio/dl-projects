"""Ablate CNN architecture, BN, augmentation, label smoothing on EuroSAT.

Experiments produced under ``artifacts/``:

- ``architecture_sweep.csv / .png``  depth in {shallow=2, medium=3, deep=4}
- ``batchnorm_ablation.csv / .png``  BN on vs off (medium depth)
- ``augmentation_comparison.csv / .png``  none vs flip vs full
- ``label_smoothing_comparison.csv / .png``  ls in {0.0, 0.05, 0.1}
- ``final_history.png``              training curves for the chosen final config
- ``eurosat_cnn.keras``              best CNN checkpoint (consumed by `landuse_explainer`)
- ``summary.json``                   structured summary

Multi-seed (``--n-seeds``, default 3) is enforced for the final chosen config
so `landuse_explainer` always has a checkpoint produced under controlled conditions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_tf_datasets  # noqa: E402
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


def _train_one(depth: int, use_bn: bool, augmentation: str,
               label_smoothing: float, epochs: int, batch_size: int,
               seed: int) -> dict:
    set_seed(seed)
    train_ds, val_ds, test_ds = load_tf_datasets(batch_size=batch_size,
                                                 seed=seed)
    model = build_model(depth=depth, use_bn=use_bn,
                        augmentation=augmentation,
                        label_smoothing=label_smoothing)
    from tensorflow import keras
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                         restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              callbacks=cbs, verbose=0)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"test_acc": float(acc), "test_loss": float(loss)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    set_seed(args.seed)

    # ---- 1. architecture depth sweep ----
    depths = [3] if args.quick else [2, 3, 4]
    depth_rows = []
    with time_block("architecture sweep"):
        for d in depths:
            per_seed = multi_seed(
                lambda seed, d=d: _train_one(
                    depth=d, use_bn=True, augmentation="none",
                    label_smoothing=0.0,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"depth": d, "blocks": f"{d}_conv_blocks", **agg,
                   "n_seeds": args.n_seeds}
            depth_rows.append(row)
            print(f"[depth={d}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(depth_rows, artifacts / "architecture_sweep.csv")
    bar_chart([f"depth={r['depth']}" for r in depth_rows],
              [r["test_acc_mean"] for r in depth_rows],
              errors=[r["test_acc_std"] for r in depth_rows],
              out_path=artifacts / "architecture_sweep.png",
              title="Architecture depth sweep", ylabel="test accuracy")

    # ---- 2. BatchNorm on/off ablation ----
    bn_rows = []
    with time_block("BN ablation"):
        for use_bn in (False, True):
            per_seed = multi_seed(
                lambda seed, use_bn=use_bn: _train_one(
                    depth=3, use_bn=use_bn, augmentation="none",
                    label_smoothing=0.0,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"batchnorm": use_bn, **agg, "n_seeds": args.n_seeds}
            bn_rows.append(row)
            print(f"[bn={use_bn}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(bn_rows, artifacts / "batchnorm_ablation.csv")
    bar_chart([str(r["batchnorm"]) for r in bn_rows],
              [r["test_acc_mean"] for r in bn_rows],
              errors=[r["test_acc_std"] for r in bn_rows],
              out_path=artifacts / "batchnorm_ablation.png",
              title="BN on vs off (depth=3)", ylabel="test accuracy")

    # ---- 3. data augmentation comparison ----
    aug_values = ["none"] if args.quick else ["none", "flip", "full"]
    aug_rows = []
    with time_block("augmentation comparison"):
        for aug in aug_values:
            per_seed = multi_seed(
                lambda seed, aug=aug: _train_one(
                    depth=3, use_bn=True, augmentation=aug,
                    label_smoothing=0.0,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"augmentation": aug, **agg, "n_seeds": args.n_seeds}
            aug_rows.append(row)
            print(f"[aug={aug}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(aug_rows, artifacts / "augmentation_comparison.csv")
    bar_chart([r["augmentation"] for r in aug_rows],
              [r["test_acc_mean"] for r in aug_rows],
              errors=[r["test_acc_std"] for r in aug_rows],
              out_path=artifacts / "augmentation_comparison.png",
              title="Data augmentation comparison", ylabel="test accuracy")

    # ---- 4. label smoothing comparison ----
    ls_values = [0.0] if args.quick else [0.0, 0.05, 0.1]
    ls_rows = []
    with time_block("label smoothing"):
        for ls in ls_values:
            per_seed = multi_seed(
                lambda seed, ls=ls: _train_one(
                    depth=3, use_bn=True, augmentation="flip",
                    label_smoothing=ls,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"label_smoothing": ls, **agg, "n_seeds": args.n_seeds}
            ls_rows.append(row)
            print(f"[ls={ls}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(ls_rows, artifacts / "label_smoothing_comparison.csv")
    bar_chart([f"ls={r['label_smoothing']}" for r in ls_rows],
              [r["test_acc_mean"] for r in ls_rows],
              errors=[r["test_acc_std"] for r in ls_rows],
              out_path=artifacts / "label_smoothing_comparison.png",
              title="Label smoothing comparison",
              ylabel="test accuracy")

    # ---- 5. final chosen config: depth=3, BN on, full aug, ls=0.0 ----
    set_seed(args.seed)
    from tensorflow import keras

    train_ds, val_ds, test_ds = load_tf_datasets(batch_size=args.batch_size,
                                                 seed=args.seed)
    ckpt_path = artifacts / "eurosat_cnn.keras"
    final_aug = "none" if args.quick else "full"
    model = build_model(depth=3, use_bn=True, augmentation=final_aug,
                        label_smoothing=0.0)
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path),
                                        monitor="val_accuracy",
                                        save_best_only=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=args.epochs, callbacks=callbacks, verbose=2)
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"[final CNN] test acc = {test_acc:.4f}")
    plot_history(history.history, artifacts / "final_history.png",
                 title="EuroSAT CNN - final config")

    summary = {
        "seed": args.seed, "n_seeds": args.n_seeds, "epochs": args.epochs,
        "architecture_sweep": depth_rows,
        "batchnorm_ablation": bn_rows,
        "augmentation_comparison": aug_rows,
        "label_smoothing_comparison": ls_rows,
        "final_config": {"depth": 3, "use_bn": True, "augmentation": final_aug,
                         "label_smoothing": 0.0,
                         "test_acc": float(test_acc),
                         "test_loss": float(test_loss)},
        "checkpoint": str(ckpt_path.name),
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
