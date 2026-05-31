"""EuroSAT RGB dataset pipeline.

EuroSAT only ships a single 'train' split, so we slice it 80/10/10 for
train / val / test using TFDS split slicing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf

NUM_CLASSES = 10
IMG_SIZE = 64
CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tfds"


def _preprocess(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def load_tf_datasets(batch_size: int = 64,
                     shuffle_buffer: int = 5_000,
                     seed: int = 42,
                     **_ignored) -> Tuple[tf.data.Dataset,
                                          tf.data.Dataset,
                                          tf.data.Dataset]:
    """Return (train_ds, val_ds, test_ds) for EuroSAT RGB."""
    import tensorflow_datasets as tfds

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (train_raw, val_raw, test_raw) = tfds.load(
        "eurosat/rgb",
        split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
        as_supervised=True,
        data_dir=str(DATA_DIR),
    )

    def _prep(ds, training: bool) -> tf.data.Dataset:
        ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(shuffle_buffer, seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return _prep(train_raw, True), _prep(val_raw, False), _prep(test_raw, False)
