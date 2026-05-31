"""MLP variants used to ablate regularization + training techniques."""
from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers, regularizers

from data import IMG_SHAPE, NUM_CLASSES


def _compile(model: keras.Model,
             learning_rate: float = 1e-3,
             grad_clip: float | None = None) -> keras.Model:
    opt_kwargs = {"learning_rate": learning_rate}
    if grad_clip is not None:
        opt_kwargs["clipnorm"] = grad_clip
    model.compile(
        optimizer=keras.optimizers.Adam(**opt_kwargs),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_overfit_mlp(learning_rate: float = 1e-3) -> keras.Model:
    """Wide unregularized MLP - the high-capacity baseline that overfits."""
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMG_SHAPE, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name="overfit_mlp",
    )
    return _compile(model, learning_rate=learning_rate)


def build_dropout_mlp(rate: float = 0.4,
                      learning_rate: float = 1e-3) -> keras.Model:
    """Same body, Dropout(rate) after each hidden Dense."""
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMG_SHAPE, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(rate),
            layers.Dense(512, activation="relu"),
            layers.Dropout(rate),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name=f"dropout_mlp_{rate}",
    )
    return _compile(model, learning_rate=learning_rate)


def build_batchnorm_mlp(use_bn: bool = True,
                        dropout: float = 0.0,
                        learning_rate: float = 1e-3) -> keras.Model:
    """Same body, optional BatchNormalization between Dense and ReLU."""
    inputs = keras.Input(shape=(*IMG_SHAPE, 1))
    x = layers.Flatten()(inputs)
    for u in (512, 512):
        x = layers.Dense(u, use_bias=not use_bn)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    name = "bn_mlp" if use_bn else "no_bn_mlp"
    return _compile(keras.Model(inputs, outputs, name=name),
                    learning_rate=learning_rate)


def build_l2_mlp(l2: float = 1e-4,
                 learning_rate: float = 1e-3) -> keras.Model:
    """Same body with kernel L2 weight decay on each Dense."""
    reg = regularizers.l2(l2)
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMG_SHAPE, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu", kernel_regularizer=reg),
            layers.Dense(512, activation="relu", kernel_regularizer=reg),
            layers.Dense(NUM_CLASSES, activation="softmax",
                         kernel_regularizer=reg),
        ],
        name=f"l2_mlp_{l2}",
    )
    return _compile(model, learning_rate=learning_rate)


def build_clip_mlp(grad_clip: float = 1.0,
                   learning_rate: float = 1e-3) -> keras.Model:
    """Same body with optimizer-side gradient clipping (`clipnorm`)."""
    model = keras.Sequential(
        [
            layers.Input(shape=(*IMG_SHAPE, 1)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ],
        name=f"clip_mlp_{grad_clip}",
    )
    return _compile(model, learning_rate=learning_rate, grad_clip=grad_clip)


def build_recipe_mlp(learning_rate: float = 1e-3) -> keras.Model:
    """The 'best-practice' recipe used for the all-callbacks demo:

    Flatten -> Dense(512) -> BN -> ReLU -> Dropout(0.3) ->
              Dense(512) -> BN -> ReLU -> Dropout(0.3) -> Softmax.

    Mild L2 on Dense kernels; clipnorm=1.0 on Adam.
    """
    reg = regularizers.l2(1e-5)
    inputs = keras.Input(shape=(*IMG_SHAPE, 1))
    x = layers.Flatten()(inputs)
    for u in (512, 512):
        x = layers.Dense(u, use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax",
                           kernel_regularizer=reg)(x)
    model = keras.Model(inputs, outputs, name="best_practice_recipe")
    return _compile(model, learning_rate=learning_rate, grad_clip=1.0)
