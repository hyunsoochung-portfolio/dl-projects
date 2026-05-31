"""AG News dataset pipeline.

We use TextVectorization with output_mode='int' and a fixed output length.
A small val split is sliced off the training data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tfds"

VOCAB_SIZE = 20_000
SEQ_LEN = 80


def _extract(example):
    # ag_news_subset returns dict with 'description' and 'label'
    text = example["description"]
    label = example["label"]
    return text, label


def load_raw_datasets(batch_size: int = 128,
                      val_fraction: float = 0.05,
                      shuffle_buffer: int = 5_000,
                      seed: int = 42) -> Tuple[tf.data.Dataset,
                                               tf.data.Dataset,
                                               tf.data.Dataset]:
    """Return (train_ds, val_ds, test_ds) of (text, label) batches."""
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = tfds.load(
        "ag_news_subset",
        split=["train", "test"],
        data_dir=str(DATA_DIR),
    )
    train_raw, test_raw = ds

    train_raw = train_raw.map(_extract, num_parallel_calls=tf.data.AUTOTUNE)
    test_raw = test_raw.map(_extract, num_parallel_calls=tf.data.AUTOTUNE)

    n_train_total = sum(1 for _ in train_raw)
    n_val = int(n_train_total * val_fraction)
    val_raw = train_raw.take(n_val)
    train_raw = train_raw.skip(n_val)

    train_raw = train_raw.shuffle(shuffle_buffer, seed=seed).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_raw = val_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_raw = test_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_raw, val_raw, test_raw


def build_vectorizer(train_ds: tf.data.Dataset,
                     vocab_size: int = VOCAB_SIZE,
                     seq_len: int = SEQ_LEN) -> tf.keras.layers.TextVectorization:
    from tensorflow.keras import layers

    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
    )
    # adapt on text only
    text_only = train_ds.map(lambda text, label: text)
    vectorizer.adapt(text_only)
    return vectorizer


def vectorize_ds(ds: tf.data.Dataset,
                 vectorizer: tf.keras.layers.TextVectorization,
                 padding: str = "post") -> tf.data.Dataset:
    """Vectorize a (text, label) dataset.

    TextVectorization defaults to *post*-padding (zeros after real tokens).
    For "pre" padding we re-pack each row so that zeros sit at the front,
    which is the classic Keras pad_sequences default and the standard
    setup for plain RNNs (real signal closest to the final timestep).
    """
    def _map(text, label):
        ids = vectorizer(text)
        if padding == "post":
            return ids, label
        # pre: shift non-zero ids to the right
        nonzero_mask = tf.cast(ids != 0, tf.int32)
        n_nonzero = tf.reduce_sum(nonzero_mask, axis=-1, keepdims=True)
        seq_len = tf.shape(ids)[-1]
        # roll each row so the trailing nonzero block lands at the end
        # tf.roll on batch is not vectorized over different shifts, so use
        # a gather trick: build per-row indices.
        idx = tf.range(seq_len)[None, :]
        shift = seq_len - n_nonzero
        gathered_idx = (idx - shift) % seq_len
        rolled = tf.gather(ids, gathered_idx, axis=-1, batch_dims=1)
        return rolled, label

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
