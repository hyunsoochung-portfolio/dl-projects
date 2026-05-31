"""Shared helpers used across all modules.

Keep this file dependency-light: only stdlib + numpy + matplotlib + tensorflow.
"""
from __future__ import annotations

import contextlib
import csv
import os
import random
import time
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow RNGs for reproducibility.

    TensorFlow is imported lazily so importing this module stays cheap.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        # TensorFlow might not be importable in some constrained environments
        # (e.g. doc-building). Seed what we can and continue.
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create the directory if it does not exist; return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def multi_seed(func: Callable[[int], dict],
               n_seeds: int = 3,
               base_seed: int = 0) -> list[dict]:
    """Run `func(seed)` for `n_seeds` consecutive seeds and return all results.

    Each call must return a dict of numeric metrics keyed by name. The caller
    is responsible for aggregating mean / std across the returned list. We keep
    aggregation external because not every metric is numeric (e.g. some return
    histories or filenames alongside scalars).
    """
    out: list[dict] = []
    for i in range(n_seeds):
        seed = base_seed + i
        set_seed(seed)
        result = dict(func(seed))
        result.setdefault("seed", seed)
        out.append(result)
    return out


def aggregate_mean_std(rows: Sequence[Mapping[str, float]],
                       keys: Sequence[str]) -> dict[str, float]:
    """Return {key_mean, key_std} aggregates over the given numeric keys."""
    agg: dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        if not vals:
            continue
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals, ddof=0))
    return agg


def save_metric_table(rows: Sequence[Mapping[str, object]],
                      out_path: str | os.PathLike) -> Path:
    """Write a list of row-dicts to CSV. Header is the union of all keys."""
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if not rows:
        out_path.write_text("")
        return out_path

    # Preserve ordering: first row's keys, then any additional keys after.
    seen: list[str] = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in seen:
                seen.append(k)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seen)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in seen})
    return out_path


def save_history_plot(history_dict: Mapping[str, Sequence[float]],
                      out_path: str | os.PathLike,
                      title: str = "Training history") -> Path:
    """Alias of :func:`plot_history` returning the output Path."""
    plot_history(history_dict, out_path, title=title)
    return Path(out_path)


@contextlib.contextmanager
def time_block(label: str = "block"):
    """Context manager that prints elapsed wall-clock time on exit."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[time] {label}: {elapsed:.2f}s")


def plot_history(history_dict: Mapping[str, Sequence[float]],
                 out_path: str | os.PathLike,
                 title: str = "Training history") -> None:
    """Plot loss + a metric (accuracy/mae) from a keras History.history dict.

    Saves a side-by-side PNG with training and validation curves.
    """
    import matplotlib.pyplot as plt

    metric_key = None
    for candidate in ("accuracy", "acc", "mae", "mean_absolute_error"):
        if candidate in history_dict:
            metric_key = candidate
            break

    n_panels = 2 if metric_key else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(history_dict["loss"], label="train")
    if "val_loss" in history_dict:
        ax.plot(history_dict["val_loss"], label="val")
    ax.set_title(f"{title} - loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)

    if metric_key:
        ax = axes[1]
        ax.plot(history_dict[metric_key], label=f"train {metric_key}")
        val_key = f"val_{metric_key}"
        if val_key in history_dict:
            ax.plot(history_dict[val_key], label=f"val {metric_key}")
        ax.set_title(f"{title} - {metric_key}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_key)
        ax.legend()
        ax.grid(alpha=0.3)

    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def bar_chart(labels: Sequence[str],
              values: Sequence[float],
              out_path: str | os.PathLike,
              title: str = "Comparison",
              ylabel: str = "value",
              errors: Sequence[float] | None = None) -> None:
    """Simple bar chart helper with optional error bars."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    bars = ax.bar(labels, values, color="#4C72B0",
                  yerr=errors if errors is not None else None,
                  capsize=4 if errors is not None else 0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def grid_image(images: Iterable[np.ndarray],
               out_path: str | os.PathLike,
               cols: int = 8,
               titles: Sequence[str] | None = None,
               cmap: str | None = "viridis") -> None:
    """Save a grid of small images (filters / feature maps)."""
    import matplotlib.pyplot as plt

    images = list(images)
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        ax = axes[idx // cols, idx % cols]
        ax.axis("off")
        if idx >= n:
            continue
        img = images[idx]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        ax.imshow(img, cmap=cmap)
        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=7)

    ensure_dir(Path(out_path).parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
