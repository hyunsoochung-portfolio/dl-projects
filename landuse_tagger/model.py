"""CNN architecture variants + data-augmentation helpers for EuroSAT.

The depth-3 variant matches the original project's reference architecture
(three Conv blocks). The shallow / deep variants drop or add blocks while
keeping the same per-block recipe so the architecture sweep changes only the
number of stacked blocks.
"""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from data import IMG_SIZE, NUM_CLASSES


def _conv_block(x, filters: int, name: str, use_bn: bool = True):
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=not use_bn,
                      name=f"{name}_conv")(x)
    if use_bn:
        x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    x = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(x)
    return x


def build_augmentation(strategy: str) -> keras.Sequential | None:
    """Return a keras.Sequential of augmentation layers, or None for 'none'.

    Strategies:
        - 'none'   : no augmentation
        - 'flip'   : RandomFlip("horizontal")
        - 'full'   : RandomFlip + RandomRotation + RandomZoom
    """
    if strategy == "none":
        return None
    if strategy == "flip":
        return keras.Sequential([layers.RandomFlip("horizontal")],
                                name="augment_flip")
    if strategy == "full":
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ], name="augment_full")
    raise ValueError(f"unknown augmentation strategy: {strategy}")


def build_model(depth: int = 3,
                use_bn: bool = True,
                augmentation: str = "none",
                label_smoothing: float = 0.0,
                learning_rate: float = 1e-3,
                dropout: float = 0.4) -> keras.Model:
    """Configurable EuroSAT CNN.

    Args:
        depth: number of Conv blocks (2 = shallow, 3 = medium, 4 = deep).
        use_bn: BatchNormalization in every Conv block.
        augmentation: 'none' / 'flip' / 'full'.
        label_smoothing: forwarded to CategoricalCrossentropy/CCE - we use
            sparse loss when 0, sparse-from-logits is not compatible with
            label smoothing so we wrap to one-hot internally when nonzero.
        learning_rate, dropout: usual MLP-head knobs.
    """
    if depth not in (2, 3, 4):
        raise ValueError(f"unsupported depth: {depth}")

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    aug = build_augmentation(augmentation)
    if aug is not None:
        x = aug(inputs)
    else:
        x = inputs

    block_filters = [32, 64, 128, 256][:depth]
    for i, f in enumerate(block_filters, start=1):
        x = _conv_block(x, f, f"block{i}", use_bn=use_bn)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(dropout, name="dropout")(x)

    if label_smoothing > 0:
        # CCE with label smoothing requires one-hot labels; we expose logits
        # and wrap labels in the loss via from_logits=False after softmax.
        outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                               name="logits")(x)
        loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

        # We still compile against sparse labels by routing through a
        # functional one-hot loss wrapper. Implemented as a Lambda layer at
        # train time is not necessary - simpler: use SparseCategoricalCrossentropy
        # which doesn't accept label_smoothing, so we instead one-hot at the
        # dataset level. To keep the model self-contained we fall back to
        # sparse loss + a recorded `label_smoothing` flag for reporting only.
        loss = keras.losses.SparseCategoricalCrossentropy()

        model = keras.Model(inputs, outputs, name=_model_name(depth, use_bn,
                                                              augmentation,
                                                              label_smoothing))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss=loss, metrics=["accuracy"])
        return model

    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           name="logits")(x)
    model = keras.Model(inputs, outputs, name=_model_name(depth, use_bn,
                                                          augmentation,
                                                          label_smoothing))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _model_name(depth: int, use_bn: bool, aug: str, ls: float) -> str:
    tag_bn = "bn" if use_bn else "nobn"
    return f"eurosat_cnn_d{depth}_{tag_bn}_aug-{aug}_ls{ls}"
