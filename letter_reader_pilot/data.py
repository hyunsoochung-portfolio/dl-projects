"""EMNIST Letters loader shared by the EMNIST-Letters projects.

Uses tensorflow_datasets. Labels in `emnist/letters` are 1..26; we shift
them to 0..25 so they line up with a standard softmax of size 26.

The raw EMNIST images come in transposed/flipped orientation relative to
how a human would read them. We rotate -90 deg + horizontal flip so the
letters look right-side-up - it doesn't matter for the model, but it
makes any visualization sane.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

NUM_CLASSES = 26
IMG_SHAPE = (28, 28)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tfds"


def _fix_orientation(image: tf.Tensor) -> tf.Tensor:
    # image is (28, 28, 1) uint8
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)
    return image


def _preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = _fix_orientation(image)
    image = tf.cast(image, tf.float32) / 255.0
    label = label - 1  # shift 1..26 -> 0..25
    return image, label


def load_tf_datasets(batch_size: int = 128,
                     val_fraction: float = 0.1,
                     shuffle_buffer: int = 10_000,
                     seed: int = 42) -> Tuple[tf.data.Dataset,
                                              tf.data.Dataset,
                                              tf.data.Dataset]:
    """Return (train_ds, val_ds, test_ds) as tf.data pipelines.

    Splits a small validation set off the training split.
    """
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (train_raw, test_raw), info = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=str(DATA_DIR),
    )

    n_train_total = info.splits["train"].num_examples
    n_val = int(n_train_total * val_fraction)

    val_raw = train_raw.take(n_val)
    train_raw = train_raw.skip(n_val)

    def _prep(ds, training: bool) -> tf.data.Dataset:
        ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(shuffle_buffer, seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return _prep(train_raw, True), _prep(val_raw, False), _prep(test_raw, False)


def load_numpy(max_train: int | None = None,
               max_test: int | None = None) -> Tuple[np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
    """Materialize the dataset into flat NumPy arrays (for sklearn baselines)."""
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_raw, test_raw = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        data_dir=str(DATA_DIR),
    )

    def to_xy(ds, cap):
        xs, ys = [], []
        for i, (img, lbl) in enumerate(ds):
            if cap is not None and i >= cap:
                break
            img = _fix_orientation(img).numpy().reshape(-1).astype(np.float32) / 255.0
            xs.append(img)
            ys.append(int(lbl.numpy()) - 1)
        return np.asarray(xs), np.asarray(ys)

    x_tr, y_tr = to_xy(train_raw, max_train)
    x_te, y_te = to_xy(test_raw, max_test)
    return x_tr, y_tr, x_te, y_te
