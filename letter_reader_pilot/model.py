"""Shallow MLP for EMNIST Letters."""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data import IMG_SHAPE, NUM_CLASSES


def build_model(hidden_units: int = 128,
                learning_rate: float = 1e-3) -> keras.Model:
    """Single hidden-layer MLP with ReLU + softmax."""
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMG_SHAPE, 1)),
            layers.Flatten(),
            layers.Dense(hidden_units, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="shallow_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
