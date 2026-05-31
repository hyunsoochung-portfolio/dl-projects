"""Ablate regularization, normalization, callbacks and LR schedules.

Experiments produced under ``artifacts/``:

- ``dropout_sweep.csv / .png``       dropout rate in {0.0, 0.2, 0.4, 0.6}
- ``batchnorm_ablation.csv / .png``  BN on vs off
- ``early_stopping_sweep.csv / .png``  EarlyStopping patience in {3, 5, 10}
- ``lr_scheduler_comparison.csv / .png``  constant vs StepDecay vs CosineDecay
                                          vs ReduceLROnPlateau
- ``l2_sweep.csv / .png``            L2 weight-decay strength sweep
- ``gradclip_ablation.csv / .png``   gradient clipping on vs off
- ``recipe_history.png``             curves for the best-practice recipe
                                     (EarlyStopping + ModelCheckpoint +
                                      ReduceLROnPlateau + TensorBoard)
- ``best_recipe.keras``              final saved checkpoint of the recipe
- ``summary.json``                   structured summary of every experiment

Run with ``--quick`` for a single-point smoke test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_arrays  # noqa: E402
from model import (  # noqa: E402
    build_batchnorm_mlp,
    build_clip_mlp,
    build_dropout_mlp,
    build_l2_mlp,
    build_overfit_mlp,
    build_recipe_mlp,
)
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


def _evaluate(model, x_te, y_te) -> tuple[float, float]:
    loss, acc = model.evaluate(x_te, y_te, verbose=0)
    return float(loss), float(acc)


def _run(model_builder, x_tr, y_tr, x_te, y_te,
         epochs: int, batch_size: int,
         callbacks=None, seed: int = 0) -> dict:
    set_seed(seed)
    model = model_builder()
    model.fit(x_tr, y_tr, validation_split=0.1, epochs=epochs,
              batch_size=batch_size, callbacks=callbacks, verbose=0)
    loss, acc = _evaluate(model, x_te, y_te)
    return {"test_acc": acc, "test_loss": loss}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    tb_dir = ensure_dir(artifacts / "tb_logs")
    set_seed(args.seed)

    from tensorflow import keras

    x_tr, y_tr, x_te, y_te = load_arrays()

    # ---- 1. dropout rate sweep ----
    dropout_values = [0.4] if args.quick else [0.0, 0.2, 0.4, 0.6]
    dropout_rows = []
    with time_block("dropout sweep"):
        for rate in dropout_values:
            per_seed = multi_seed(
                lambda seed, rate=rate: _run(
                    lambda: (build_overfit_mlp() if rate == 0.0
                             else build_dropout_mlp(rate=rate)),
                    x_tr, y_tr, x_te, y_te,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"dropout": rate, **agg, "n_seeds": args.n_seeds}
            dropout_rows.append(row)
            print(f"[dropout={rate}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(dropout_rows, artifacts / "dropout_sweep.csv")
    bar_chart([str(r["dropout"]) for r in dropout_rows],
              [r["test_acc_mean"] for r in dropout_rows],
              errors=[r["test_acc_std"] for r in dropout_rows],
              out_path=artifacts / "dropout_sweep.png",
              title="Dropout rate sweep", ylabel="test accuracy")

    # ---- 2. BatchNorm on/off ablation ----
    bn_rows = []
    with time_block("batchnorm ablation"):
        for use_bn in (False, True):
            per_seed = multi_seed(
                lambda seed, use_bn=use_bn: _run(
                    lambda: build_batchnorm_mlp(use_bn=use_bn, dropout=0.3),
                    x_tr, y_tr, x_te, y_te,
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
              title="BatchNorm on vs off", ylabel="test accuracy")

    # ---- 3. EarlyStopping patience sweep ----
    es_values = [5] if args.quick else [3, 5, 10]
    es_rows = []
    with time_block("early stopping sweep"):
        for patience in es_values:
            cbs = [keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True)]
            per_seed = multi_seed(
                lambda seed, cbs=cbs: _run(
                    lambda: build_dropout_mlp(rate=0.3),
                    x_tr, y_tr, x_te, y_te,
                    epochs=args.epochs * 2,  # let EarlyStopping actually act
                    batch_size=args.batch_size,
                    callbacks=cbs, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"patience": patience, **agg, "n_seeds": args.n_seeds}
            es_rows.append(row)
            print(f"[EarlyStopping patience={patience}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(es_rows, artifacts / "early_stopping_sweep.csv")
    bar_chart([f"p={r['patience']}" for r in es_rows],
              [r["test_acc_mean"] for r in es_rows],
              errors=[r["test_acc_std"] for r in es_rows],
              out_path=artifacts / "early_stopping_sweep.png",
              title="EarlyStopping patience sweep", ylabel="test accuracy")

    # ---- 4. LR scheduler comparison ----
    def step_decay(epoch, lr):
        return lr * (0.5 if (epoch and epoch % 3 == 0) else 1.0)

    def cosine_decay(epoch, lr):
        # smooth half-cosine from 1.0 -> 0.1 over args.epochs
        progress = min(1.0, epoch / max(1, args.epochs - 1))
        scale = 0.1 + 0.5 * (1 - 0.1) * (1 + math.cos(math.pi * progress))
        return 1e-3 * scale

    schedulers = {
        "constant": [],
        "step_decay": [keras.callbacks.LearningRateScheduler(step_decay)],
        "cosine_decay": [keras.callbacks.LearningRateScheduler(cosine_decay)],
        "reduce_on_plateau": [keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)],
    }
    if args.quick:
        schedulers = {k: v for k, v in list(schedulers.items())[:2]}
    sched_rows = []
    with time_block("lr scheduler comparison"):
        for name, cbs in schedulers.items():
            per_seed = multi_seed(
                lambda seed, cbs=cbs: _run(
                    lambda: build_dropout_mlp(rate=0.3),
                    x_tr, y_tr, x_te, y_te,
                    epochs=args.epochs, batch_size=args.batch_size,
                    callbacks=cbs, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"scheduler": name, **agg, "n_seeds": args.n_seeds}
            sched_rows.append(row)
            print(f"[scheduler={name}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(sched_rows, artifacts / "lr_scheduler_comparison.csv")
    bar_chart([r["scheduler"] for r in sched_rows],
              [r["test_acc_mean"] for r in sched_rows],
              errors=[r["test_acc_std"] for r in sched_rows],
              out_path=artifacts / "lr_scheduler_comparison.png",
              title="LR scheduler comparison", ylabel="test accuracy")

    # ---- 5. L2 weight-decay sweep ----
    l2_values = [1e-4] if args.quick else [0.0, 1e-5, 1e-4, 1e-3]
    l2_rows = []
    with time_block("l2 sweep"):
        for l2 in l2_values:
            per_seed = multi_seed(
                lambda seed, l2=l2: _run(
                    lambda: (build_overfit_mlp() if l2 == 0.0
                             else build_l2_mlp(l2=l2)),
                    x_tr, y_tr, x_te, y_te,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"l2": l2, **agg, "n_seeds": args.n_seeds}
            l2_rows.append(row)
            print(f"[l2={l2}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(l2_rows, artifacts / "l2_sweep.csv")
    bar_chart([f"l2={r['l2']}" for r in l2_rows],
              [r["test_acc_mean"] for r in l2_rows],
              errors=[r["test_acc_std"] for r in l2_rows],
              out_path=artifacts / "l2_sweep.png",
              title="L2 weight-decay sweep", ylabel="test accuracy")

    # ---- 6. gradient clipping on/off ablation ----
    clip_rows = []
    with time_block("grad clip ablation"):
        for grad_clip in (None, 1.0):
            per_seed = multi_seed(
                lambda seed, grad_clip=grad_clip: _run(
                    lambda: (build_overfit_mlp() if grad_clip is None
                             else build_clip_mlp(grad_clip=grad_clip)),
                    x_tr, y_tr, x_te, y_te,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"clipnorm": grad_clip if grad_clip is not None else "off",
                   **agg, "n_seeds": args.n_seeds}
            clip_rows.append(row)
            print(f"[clipnorm={grad_clip}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(clip_rows, artifacts / "gradclip_ablation.csv")
    bar_chart([str(r["clipnorm"]) for r in clip_rows],
              [r["test_acc_mean"] for r in clip_rows],
              errors=[r["test_acc_std"] for r in clip_rows],
              out_path=artifacts / "gradclip_ablation.png",
              title="Gradient clipping on vs off", ylabel="test accuracy")

    # ---- 7. best-practice recipe: all 4 callbacks together ----
    ckpt_path = artifacts / "best_recipe.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=str(ckpt_path),
                                        monitor="val_loss",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=2, min_lr=1e-5),
        keras.callbacks.TensorBoard(log_dir=str(tb_dir)),
    ]
    set_seed(args.seed)
    recipe = build_recipe_mlp()
    recipe.summary()
    recipe_hist = recipe.fit(x_tr, y_tr, validation_split=0.1,
                             epochs=args.epochs * 2,
                             batch_size=args.batch_size,
                             callbacks=callbacks, verbose=2)
    plot_history(recipe_hist.history, artifacts / "recipe_history.png",
                 title="Best-practice recipe (BN + Dropout + L2 + clipnorm)")
    recipe_loss, recipe_acc = _evaluate(recipe, x_te, y_te)
    print(f"[recipe] test acc = {recipe_acc:.4f}")

    # save/load round-trip
    saved_path = artifacts / "final_recipe.keras"
    recipe.save(saved_path)
    reloaded = keras.models.load_model(saved_path)
    rl_loss, rl_acc = _evaluate(reloaded, x_te, y_te)
    print(f"[reloaded] test acc = {rl_acc:.4f} (should match recipe above)")

    summary = {
        "seed": args.seed, "n_seeds": args.n_seeds, "epochs": args.epochs,
        "dropout_sweep": dropout_rows,
        "batchnorm_ablation": bn_rows,
        "early_stopping_sweep": es_rows,
        "lr_scheduler_comparison": sched_rows,
        "l2_sweep": l2_rows,
        "gradclip_ablation": clip_rows,
        "recipe_test_acc": float(recipe_acc),
        "recipe_test_loss": float(recipe_loss),
        "reloaded_test_acc": float(rl_acc),
        "best_checkpoint": str(ckpt_path.name),
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
