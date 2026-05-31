"""Benchmark depth, activations, optimizers and learning rates on EMNIST Letters.

Outputs under ``artifacts/``:

- ``depth_sweep.csv / .png``         test acc for depth in {1, 2, 3, 4}
- ``activations.csv / .png``         test acc for activations
                                     {sigmoid, tanh, relu, leaky_relu, elu, gelu, swish}
- ``optimizers.csv / .png``          all 8 optimizers test acc (3 seeds each)
- ``lr_opt_grid.csv / .png``         3 LRs x top-3 optimizers mini-grid
- ``final_comparison.csv / .png``    combined leaderboard
- ``best_model.keras``               best (optimizer, lr) trained model
- ``summary.json``                   full structured summary

Every cell of every sweep is run for ``--n-seeds`` consecutive seeds; we report
mean and std of test accuracy and the bar charts carry error bars.

Usage:
    python letter_reader_advanced/train.py
    python letter_reader_advanced/train.py --quick --n-seeds 1 --epochs 2
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
from model import (  # noqa: E402
    ACTIVATIONS,
    build_sequential_list,
    build_with_add,
    compile_with,
    depth_to_hidden,
    optimizer_factories,
)
from shared.utils import (  # noqa: E402
    aggregate_mean_std,
    bar_chart,
    ensure_dir,
    multi_seed,
    save_metric_table,
    set_seed,
    time_block,
)


def _train_once(model_builder, optimizer, train_ds, val_ds, test_ds, epochs):
    model = compile_with(model_builder(), optimizer)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return model, float(loss), float(acc)


def _seeded_run(model_builder_factory, optimizer_factory, train_ds, val_ds,
                test_ds, epochs):
    """Wrap _train_once with seed support so multi_seed can drive it."""
    def run(seed: int):
        set_seed(seed)
        _, loss, acc = _train_once(model_builder_factory, optimizer_factory(),
                                   train_ds, val_ds, test_ds, epochs)
        return {"test_acc": acc, "test_loss": loss}
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true",
                        help="single-point sweeps for smoke testing")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    set_seed(args.seed)
    train_ds, val_ds, test_ds = load_tf_datasets(batch_size=args.batch_size)

    # ---- show both Sequential construction styles + summary ----
    print(">>> Sequential([...]) style")
    build_sequential_list("relu", depth_to_hidden(3)).summary()
    print("\n>>> model = Sequential(); model.add(...) style")
    build_with_add("relu", depth_to_hidden(3)).summary()

    # ---- 1. depth sweep ----
    depth_values = [3] if args.quick else [1, 2, 3, 4]
    depth_rows = []
    with time_block("depth sweep"):
        for d in depth_values:
            from tensorflow.keras.optimizers import Adam

            per_seed = multi_seed(
                _seeded_run(
                    lambda d=d: build_sequential_list("relu", depth_to_hidden(d)),
                    lambda: Adam(),
                    train_ds, val_ds, test_ds, args.epochs,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"depth": d, "hidden_layers": str(depth_to_hidden(d)),
                   **agg, "n_seeds": args.n_seeds}
            depth_rows.append(row)
            print(f"[depth] {d} -> acc={agg.get('test_acc_mean'):.4f} +/- {agg.get('test_acc_std'):.4f}")
    save_metric_table(depth_rows, artifacts / "depth_sweep.csv")
    bar_chart([str(r["depth"]) for r in depth_rows],
              [r["test_acc_mean"] for r in depth_rows],
              errors=[r["test_acc_std"] for r in depth_rows],
              out_path=artifacts / "depth_sweep.png",
              title="Depth sweep (Adam, ReLU)",
              ylabel="test accuracy")

    # ---- 2. activation sweep (all 7 activations) ----
    act_values = ("relu",) if args.quick else ACTIVATIONS
    activation_rows = []
    with time_block("activation sweep"):
        for act in act_values:
            from tensorflow.keras.optimizers import Adam

            per_seed = multi_seed(
                _seeded_run(
                    lambda act=act: build_sequential_list(act, depth_to_hidden(3)),
                    lambda: Adam(),
                    train_ds, val_ds, test_ds, args.epochs,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"activation": act, **agg, "n_seeds": args.n_seeds}
            activation_rows.append(row)
            print(f"[activation] {act:10s} -> acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(activation_rows, artifacts / "activations.csv")
    bar_chart([r["activation"] for r in activation_rows],
              [r["test_acc_mean"] for r in activation_rows],
              errors=[r["test_acc_std"] for r in activation_rows],
              out_path=artifacts / "activations.png",
              title="Activation comparison (depth=3, Adam)",
              ylabel="test accuracy")

    # ---- 3. all 8 optimizers benchmark ----
    opt_factories = optimizer_factories()
    if args.quick:
        opt_factories = {k: v for k, v in list(opt_factories.items())[:2]}
    opt_rows = []
    best = {"name": None, "acc": -1.0, "model": None}
    with time_block("optimizer benchmark"):
        for name, factory in opt_factories.items():
            per_seed = multi_seed(
                _seeded_run(
                    lambda: build_sequential_list("relu", depth_to_hidden(3)),
                    factory,
                    train_ds, val_ds, test_ds, args.epochs,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"optimizer": name, **agg, "n_seeds": args.n_seeds}
            opt_rows.append(row)
            print(f"[optimizer] {name:14s} -> acc={agg.get('test_acc_mean'):.4f}")
            # train one more time at default seed to keep a model for saving
            if agg.get("test_acc_mean", 0.0) > best["acc"]:
                set_seed(args.seed)
                model, _, _ = _train_once(
                    lambda: build_sequential_list("relu", depth_to_hidden(3)),
                    factory(),
                    train_ds, val_ds, test_ds, args.epochs,
                )
                best = {"name": name, "acc": float(agg["test_acc_mean"]),
                        "model": model}
    save_metric_table(opt_rows, artifacts / "optimizers.csv")
    bar_chart([r["optimizer"] for r in opt_rows],
              [r["test_acc_mean"] for r in opt_rows],
              errors=[r["test_acc_std"] for r in opt_rows],
              out_path=artifacts / "optimizers.png",
              title="Optimizer comparison (depth=3, ReLU)",
              ylabel="test accuracy")
    if best["model"] is not None:
        best["model"].save(artifacts / "best_model.keras")

    # ---- 4. LR x optimizer mini-grid: top-3 optimizers x 3 LRs ----
    top_opts = sorted(opt_rows, key=lambda r: r["test_acc_mean"], reverse=True)
    top_names = [r["optimizer"] for r in top_opts[:3]]
    lr_values = [1e-3] if args.quick else [1e-2, 1e-3, 1e-4]
    grid_rows = []
    with time_block("lr x optimizer grid"):
        for opt_name in top_names:
            for lr in lr_values:
                factory = optimizer_factories(learning_rate=lr)[opt_name]
                per_seed = multi_seed(
                    _seeded_run(
                        lambda: build_sequential_list("relu", depth_to_hidden(3)),
                        factory,
                        train_ds, val_ds, test_ds, args.epochs,
                    ),
                    n_seeds=args.n_seeds, base_seed=args.seed,
                )
                agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
                row = {"optimizer": opt_name, "lr": lr, **agg,
                       "n_seeds": args.n_seeds}
                grid_rows.append(row)
                print(f"[grid] {opt_name:14s} lr={lr:.0e} -> acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(grid_rows, artifacts / "lr_opt_grid.csv")
    bar_chart(
        [f"{r['optimizer']}@{r['lr']:.0e}" for r in grid_rows],
        [r["test_acc_mean"] for r in grid_rows],
        errors=[r["test_acc_std"] for r in grid_rows],
        out_path=artifacts / "lr_opt_grid.png",
        title="LR x optimizer mini-grid",
        ylabel="test accuracy",
    )

    # ---- 5. consolidated leaderboard ----
    leaderboard = sorted(
        [{"name": f"{r['optimizer']}@{r['lr']:.0e}", **r} for r in grid_rows],
        key=lambda r: r["test_acc_mean"], reverse=True,
    )
    save_metric_table(leaderboard, artifacts / "final_comparison.csv")
    bar_chart(
        [r["name"] for r in leaderboard],
        [r["test_acc_mean"] for r in leaderboard],
        errors=[r["test_acc_std"] for r in leaderboard],
        out_path=artifacts / "final_comparison.png",
        title="Final leaderboard (sorted by mean test accuracy)",
        ylabel="test accuracy",
    )

    summary = {
        "seed": args.seed, "n_seeds": args.n_seeds, "epochs": args.epochs,
        "best_optimizer": best["name"], "best_test_acc": best["acc"],
        "depth_sweep": depth_rows,
        "activations": activation_rows,
        "optimizers": opt_rows,
        "lr_opt_grid": grid_rows,
        "leaderboard_top3": leaderboard[:3],
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
