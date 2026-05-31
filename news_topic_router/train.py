"""Sweep embedding dim, RNN units, bidirectionality, and padding on AG News.

Experiments produced under ``artifacts/``:

- ``embed_dim_sweep.csv / .png``     embed dim in {32, 64, 128, 256}
- ``rnn_units_sweep.csv / .png``     rnn units in {32, 64, 128}
- ``bidirectional_ablation.csv / .png``  uni- vs bi-directional SimpleRNN
- ``padding_ablation.csv / .png``    pre- vs post-padding
- ``input_encoding_comparison.csv / .png``  one-hot vs embedding (same RNN size)
- ``final_history.png``              training curves for the chosen final config
- ``embedding_rnn.keras``            final embedding RNN checkpoint
- ``summary.json``                   structured summary

Every sweep cell is run for ``--n-seeds`` seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import build_vectorizer, load_raw_datasets, vectorize_ds  # noqa: E402
from model import build_embedding_rnn, build_onehot_rnn  # noqa: E402
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


def _prepare(batch_size: int, padding: str):
    raw_train, raw_val, raw_test = load_raw_datasets(batch_size=batch_size)
    vectorizer = build_vectorizer(raw_train)
    train_ds = vectorize_ds(raw_train, vectorizer, padding=padding)
    val_ds = vectorize_ds(raw_val, vectorizer, padding=padding)
    test_ds = vectorize_ds(raw_test, vectorizer, padding=padding)
    return train_ds, val_ds, test_ds


def _run_embedding(embed_dim: int, rnn_units: int, bidirectional: bool,
                   padding: str, epochs: int, batch_size: int,
                   seed: int) -> dict:
    set_seed(seed)
    train_ds, val_ds, test_ds = _prepare(batch_size, padding)
    model = build_embedding_rnn(embed_dim=embed_dim, rnn_units=rnn_units,
                                bidirectional=bidirectional)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"test_acc": float(acc), "test_loss": float(loss)}


def _run_onehot(rnn_units: int, epochs: int, batch_size: int,
                padding: str, seed: int) -> dict:
    set_seed(seed)
    train_ds, val_ds, test_ds = _prepare(batch_size, padding)
    model = build_onehot_rnn(rnn_units=rnn_units)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"test_acc": float(acc), "test_loss": float(loss)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    artifacts = ensure_dir(args.output_dir)
    set_seed(args.seed)

    # ---- 1. embedding-dim sweep ----
    embed_values = [64] if args.quick else [32, 64, 128, 256]
    embed_rows = []
    with time_block("embedding dim sweep"):
        for ed in embed_values:
            per_seed = multi_seed(
                lambda seed, ed=ed: _run_embedding(
                    embed_dim=ed, rnn_units=64, bidirectional=False,
                    padding="post",
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"embed_dim": ed, **agg, "n_seeds": args.n_seeds}
            embed_rows.append(row)
            print(f"[embed_dim={ed}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(embed_rows, artifacts / "embed_dim_sweep.csv")
    bar_chart([str(r["embed_dim"]) for r in embed_rows],
              [r["test_acc_mean"] for r in embed_rows],
              errors=[r["test_acc_std"] for r in embed_rows],
              out_path=artifacts / "embed_dim_sweep.png",
              title="Embedding dim sweep (rnn=64)", ylabel="test accuracy")

    # ---- 2. RNN units sweep ----
    units_values = [64] if args.quick else [32, 64, 128]
    units_rows = []
    with time_block("rnn units sweep"):
        for u in units_values:
            per_seed = multi_seed(
                lambda seed, u=u: _run_embedding(
                    embed_dim=64, rnn_units=u, bidirectional=False,
                    padding="post",
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"rnn_units": u, **agg, "n_seeds": args.n_seeds}
            units_rows.append(row)
            print(f"[rnn_units={u}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(units_rows, artifacts / "rnn_units_sweep.csv")
    bar_chart([str(r["rnn_units"]) for r in units_rows],
              [r["test_acc_mean"] for r in units_rows],
              errors=[r["test_acc_std"] for r in units_rows],
              out_path=artifacts / "rnn_units_sweep.png",
              title="RNN units sweep (embed=64)", ylabel="test accuracy")

    # ---- 3. bidirectional ablation ----
    bi_rows = []
    with time_block("bidirectional ablation"):
        for bi in (False, True):
            per_seed = multi_seed(
                lambda seed, bi=bi: _run_embedding(
                    embed_dim=64, rnn_units=64, bidirectional=bi,
                    padding="post",
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"bidirectional": bi, **agg, "n_seeds": args.n_seeds}
            bi_rows.append(row)
            print(f"[bi={bi}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(bi_rows, artifacts / "bidirectional_ablation.csv")
    bar_chart([str(r["bidirectional"]) for r in bi_rows],
              [r["test_acc_mean"] for r in bi_rows],
              errors=[r["test_acc_std"] for r in bi_rows],
              out_path=artifacts / "bidirectional_ablation.png",
              title="Bidirectional vs unidirectional SimpleRNN",
              ylabel="test accuracy")

    # ---- 4. padding strategy ablation ----
    pad_rows = []
    with time_block("padding ablation"):
        for pad in ("pre", "post"):
            per_seed = multi_seed(
                lambda seed, pad=pad: _run_embedding(
                    embed_dim=64, rnn_units=64, bidirectional=False,
                    padding=pad,
                    epochs=args.epochs, batch_size=args.batch_size, seed=seed,
                ),
                n_seeds=args.n_seeds, base_seed=args.seed,
            )
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            row = {"padding": pad, **agg, "n_seeds": args.n_seeds}
            pad_rows.append(row)
            print(f"[padding={pad}] acc={agg.get('test_acc_mean'):.4f}")
    save_metric_table(pad_rows, artifacts / "padding_ablation.csv")
    bar_chart([r["padding"] for r in pad_rows],
              [r["test_acc_mean"] for r in pad_rows],
              errors=[r["test_acc_std"] for r in pad_rows],
              out_path=artifacts / "padding_ablation.png",
              title="Pre- vs post-padding", ylabel="test accuracy")

    # ---- 5. one-hot vs Embedding head-to-head ----
    enc_rows = []
    with time_block("input encoding comparison"):
        oh_per_seed = multi_seed(
            lambda seed: _run_onehot(
                rnn_units=64,
                epochs=max(1, args.epochs // 2),  # one-hot is heavy
                batch_size=args.batch_size, padding="post", seed=seed,
            ),
            n_seeds=args.n_seeds, base_seed=args.seed,
        )
        emb_per_seed = multi_seed(
            lambda seed: _run_embedding(
                embed_dim=64, rnn_units=64, bidirectional=False,
                padding="post",
                epochs=args.epochs, batch_size=args.batch_size, seed=seed,
            ),
            n_seeds=args.n_seeds, base_seed=args.seed,
        )
        for name, per_seed in (("onehot_rnn", oh_per_seed),
                               ("embedding_rnn", emb_per_seed)):
            agg = aggregate_mean_std(per_seed, ["test_acc", "test_loss"])
            enc_rows.append({"model": name, **agg, "n_seeds": args.n_seeds})
    save_metric_table(enc_rows, artifacts / "input_encoding_comparison.csv")
    bar_chart([r["model"] for r in enc_rows],
              [r["test_acc_mean"] for r in enc_rows],
              errors=[r["test_acc_std"] for r in enc_rows],
              out_path=artifacts / "input_encoding_comparison.png",
              title="One-hot vs Embedding (same RNN size)",
              ylabel="test accuracy")

    # ---- 6. final chosen config + saved checkpoint ----
    set_seed(args.seed)
    train_ds, val_ds, test_ds = _prepare(args.batch_size, padding="post")
    model = build_embedding_rnn(embed_dim=64, rnn_units=64, bidirectional=True)
    model.summary()
    hist = model.fit(train_ds, validation_data=val_ds,
                     epochs=args.epochs, verbose=2)
    plot_history(hist.history, artifacts / "final_history.png",
                 title="Embedding bi-SimpleRNN (final)")
    fl_loss, fl_acc = model.evaluate(test_ds, verbose=2)
    model.save(artifacts / "embedding_rnn.keras")
    print(f"[final] test acc = {fl_acc:.4f}")

    summary = {
        "seed": args.seed, "n_seeds": args.n_seeds, "epochs": args.epochs,
        "embed_dim_sweep": embed_rows,
        "rnn_units_sweep": units_rows,
        "bidirectional_ablation": bi_rows,
        "padding_ablation": pad_rows,
        "input_encoding_comparison": enc_rows,
        "final_test_acc": float(fl_acc),
        "final_test_loss": float(fl_loss),
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
