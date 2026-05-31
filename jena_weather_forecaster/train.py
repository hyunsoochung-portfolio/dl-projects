"""Benchmark RNN/LSTM/GRU, window, horizon, recurrent dropout, depth, bidir.

Experiments produced under ``artifacts/``:

- ``architectures.csv / .png``       SimpleRNN / LSTM / GRU head-to-head (3 seeds)
- ``window_sweep.csv / .png``        past window in {24h, 72h, 120h}
- ``horizon_sweep.csv / .png``       predict +1h / +6h / +24h ahead
- ``recurrent_dropout.csv / .png``   LSTM recurrent_dropout in {0.0, 0.2, 0.4}
- ``stacked_vs_single.csv / .png``   1-layer vs 2-layer stacked LSTM
- ``bidirectional_lstm.csv / .png``  unidir vs Bidirectional LSTM
- ``history_<arch>.png``             training curves for each top-line model
- ``best_model.keras``               lowest-MAE model across the architectures benchmark
- ``summary.json``                   full structured summary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import HORIZON, WINDOW_PAST, build_datasets  # noqa: E402
from model import (  # noqa: E402
    build_gru,
    build_lstm,
    build_lstm_recurrent_dropout,
    build_simple_rnn,
    build_stacked_lstm,
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


def _fit_eval(model_factory, datasets, epochs: int):
    train_ds, val_ds, test_ds, _ = datasets
    model = model_factory()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=epochs, verbose=0)
    loss, mae = model.evaluate(test_ds, verbose=0)
    return model, hist, float(loss), float(mae)


def _seeded_run(model_factory, datasets, epochs):
    def run(seed):
        set_seed(seed)
        _, _, loss, mae = _fit_eval(model_factory, datasets, epochs)
        return {"test_mae": mae, "test_loss": loss}
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    set_seed(args.seed)

    # Build the default-window datasets once and reuse across the per-arch
    # / dropout / depth / bidir sweeps.
    base_datasets = build_datasets(batch_size=args.batch_size,
                                   window_past=WINDOW_PAST, horizon=HORIZON)
    n_features = base_datasets[-1]

    # ---- 1. SimpleRNN / LSTM / GRU head-to-head ----
    arch_factories = {
        "SimpleRNN": lambda: build_simple_rnn(n_features),
        "LSTM": lambda: build_lstm(n_features),
        "GRU": lambda: build_gru(n_features),
    }
    if args.quick:
        arch_factories = {"LSTM": arch_factories["LSTM"]}
    arch_rows = []
    best = {"name": None, "mae": float("inf"), "model": None}
    with time_block("architectures benchmark"):
        for name, factory in arch_factories.items():
            per_seed = multi_seed(
                _seeded_run(factory, base_datasets, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            row = {"architecture": name, **agg, "n_seeds": args.n_seeds}
            arch_rows.append(row)
            print(f"[arch={name}] MAE={agg.get('test_mae_mean'):.4f}")
            # also do one full fit at base seed so we can save the best model
            set_seed(args.seed)
            model, hist, loss, mae = _fit_eval(factory, base_datasets, args.epochs)
            plot_history(hist.history,
                         artifacts / f"history_{name.replace('+', '_')}.png",
                         title=name)
            if mae < best["mae"]:
                best = {"name": name, "mae": float(mae), "model": model}
    save_metric_table(arch_rows, artifacts / "architectures.csv")
    bar_chart([r["architecture"] for r in arch_rows],
              [r["test_mae_mean"] for r in arch_rows],
              errors=[r["test_mae_std"] for r in arch_rows],
              out_path=artifacts / "architectures.png",
              title="SimpleRNN / LSTM / GRU - test MAE",
              ylabel="MAE (standardized units)")
    if best["model"] is not None:
        best["model"].save(artifacts / "best_model.keras")

    # ---- 2. window-size sweep ----
    window_values = [WINDOW_PAST] if args.quick else [24, 72, 120]
    window_rows = []
    with time_block("window sweep"):
        for w in window_values:
            ds = build_datasets(batch_size=args.batch_size,
                                window_past=w, horizon=HORIZON)
            per_seed = multi_seed(
                _seeded_run(lambda w=w: build_lstm(ds[-1], window_past=w),
                            ds, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            row = {"window_past_hours": w, **agg, "n_seeds": args.n_seeds}
            window_rows.append(row)
            print(f"[window={w}h] MAE={agg.get('test_mae_mean'):.4f}")
    save_metric_table(window_rows, artifacts / "window_sweep.csv")
    bar_chart([f"{r['window_past_hours']}h" for r in window_rows],
              [r["test_mae_mean"] for r in window_rows],
              errors=[r["test_mae_std"] for r in window_rows],
              out_path=artifacts / "window_sweep.png",
              title="Window-size sweep (LSTM)", ylabel="MAE")

    # ---- 3. horizon sweep ----
    horizon_values = [HORIZON] if args.quick else [1, 6, 24]
    horizon_rows = []
    with time_block("horizon sweep"):
        for h in horizon_values:
            ds = build_datasets(batch_size=args.batch_size,
                                window_past=WINDOW_PAST, horizon=h)
            per_seed = multi_seed(
                _seeded_run(lambda: build_lstm(ds[-1]),
                            ds, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            row = {"horizon_hours": h, **agg, "n_seeds": args.n_seeds}
            horizon_rows.append(row)
            print(f"[horizon=+{h}h] MAE={agg.get('test_mae_mean'):.4f}")
    save_metric_table(horizon_rows, artifacts / "horizon_sweep.csv")
    bar_chart([f"+{r['horizon_hours']}h" for r in horizon_rows],
              [r["test_mae_mean"] for r in horizon_rows],
              errors=[r["test_mae_std"] for r in horizon_rows],
              out_path=artifacts / "horizon_sweep.png",
              title="Forecast horizon sweep (LSTM)", ylabel="MAE")

    # ---- 4. recurrent dropout sweep on LSTM ----
    drop_values = [0.0] if args.quick else [0.0, 0.2, 0.4]
    drop_rows = []
    with time_block("recurrent dropout sweep"):
        for rd in drop_values:
            factory = (lambda rd=rd:
                       build_lstm(n_features) if rd == 0.0
                       else build_lstm_recurrent_dropout(n_features, rec_dropout=rd))
            per_seed = multi_seed(
                _seeded_run(factory, base_datasets, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            row = {"recurrent_dropout": rd, **agg, "n_seeds": args.n_seeds}
            drop_rows.append(row)
            print(f"[rec_dropout={rd}] MAE={agg.get('test_mae_mean'):.4f}")
    save_metric_table(drop_rows, artifacts / "recurrent_dropout.csv")
    bar_chart([f"rd={r['recurrent_dropout']}" for r in drop_rows],
              [r["test_mae_mean"] for r in drop_rows],
              errors=[r["test_mae_std"] for r in drop_rows],
              out_path=artifacts / "recurrent_dropout.png",
              title="LSTM recurrent_dropout sweep", ylabel="MAE")

    # ---- 5. stacked vs single LSTM ----
    stack_rows = []
    with time_block("stacked vs single"):
        for name, factory in (
            ("lstm_1layer", lambda: build_lstm(n_features)),
            ("lstm_2layer_stacked", lambda: build_stacked_lstm(n_features)),
        ):
            per_seed = multi_seed(
                _seeded_run(factory, base_datasets, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            stack_rows.append({"model": name, **agg, "n_seeds": args.n_seeds})
            print(f"[{name}] MAE={agg.get('test_mae_mean'):.4f}")
    save_metric_table(stack_rows, artifacts / "stacked_vs_single.csv")
    bar_chart([r["model"] for r in stack_rows],
              [r["test_mae_mean"] for r in stack_rows],
              errors=[r["test_mae_std"] for r in stack_rows],
              out_path=artifacts / "stacked_vs_single.png",
              title="1-layer vs 2-layer stacked LSTM", ylabel="MAE")

    # ---- 6. unidirectional vs Bidirectional LSTM ----
    bi_rows = []
    with time_block("bidirectional LSTM"):
        for bi in (False, True):
            per_seed = multi_seed(
                _seeded_run(lambda bi=bi: build_lstm(n_features, bidirectional=bi),
                            base_datasets, args.epochs),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_mae", "test_loss"])
            bi_rows.append({"bidirectional": bi, **agg,
                            "n_seeds": args.n_seeds})
            print(f"[bi={bi}] MAE={agg.get('test_mae_mean'):.4f}")
    save_metric_table(bi_rows, artifacts / "bidirectional_lstm.csv")
    bar_chart([str(r["bidirectional"]) for r in bi_rows],
              [r["test_mae_mean"] for r in bi_rows],
              errors=[r["test_mae_std"] for r in bi_rows],
              out_path=artifacts / "bidirectional_lstm.png",
              title="Unidirectional vs Bidirectional LSTM",
              ylabel="MAE")

    summary = {
        "seed": args.seed, "n_seeds": args.n_seeds, "epochs": args.epochs,
        "architectures": arch_rows,
        "window_sweep": window_rows,
        "horizon_sweep": horizon_rows,
        "recurrent_dropout": drop_rows,
        "stacked_vs_single": stack_rows,
        "bidirectional_lstm": bi_rows,
        "best_architecture": best["name"], "best_mae": best["mae"],
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
