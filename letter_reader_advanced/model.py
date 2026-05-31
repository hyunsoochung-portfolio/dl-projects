"""Deep MLP definitions used by the benchmark loop in train.py.

Provides:
- build_sequential_list(activation, hidden_layers): Sequential([...]) style
- build_with_add(activation, hidden_layers): imperative Sequential().add(...) style
- compile_with(model, optimizer): attach loss + metric
- optimizer_factories(): all 8 optimizers used in the benchmark
- ACTIVATIONS: full list of activations swept in the activation experiment
- depth_to_hidden(n): canonical width recipe for a given depth
"""
from __future__ import annotations

from typing import Callable

from tensorflow import keras
from tensorflow.keras import layers

from data import IMG_SHAPE, NUM_CLASSES

# Activations swept in train.py. `leaky_relu` and `swish` aren't strings on
# every Keras version, so we wrap them as proper Activation layers when needed.
ACTIVATIONS = ("sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu", "swish")


def _activation_layer(name: str) -> layers.Layer:
    """Return an Activation/LeakyReLU layer for the given name (Keras-version-safe)."""
    if name == "leaky_relu":
        return layers.LeakyReLU(negative_slope=0.1)
    if name == "swish":
        return layers.Activation("swish")
    return layers.Activation(name)


def _dense_block(units: int, activation: str) -> list[layers.Layer]:
    """A Dense (no built-in activation) + separate Activation layer.

    Using a separate Activation layer lets us swap in LeakyReLU / GELU / swish
    consistently across all the swept activations.
    """
    return [layers.Dense(units), _activation_layer(activation)]


def depth_to_hidden(depth: int) -> tuple[int, ...]:
    """Canonical widths used in the depth sweep.

    Depth 1 -> (128,), 2 -> (256,128), 3 -> (256,128,64), 4 -> (512,256,128,64).
    Keeping a single recipe means the depth sweep varies *only* the number of
    layers, not the architectural family per layer count.
    """
    table = {
        1: (128,),
        2: (256, 128),
        3: (256, 128, 64),
        4: (512, 256, 128, 64),
    }
    if depth not in table:
        raise ValueError(f"unsupported depth: {depth}")
    return table[depth]


def build_sequential_list(activation: str = "relu",
                          hidden_layers=(256, 128, 64)) -> keras.Model:
    """`Sequential([...])` constructor style."""
    layer_stack: list[layers.Layer] = [
        layers.Input(shape=(*IMG_SHAPE, 1)),
        layers.Flatten(),
    ]
    for u in hidden_layers:
        layer_stack.extend(_dense_block(u, activation))
    layer_stack.append(layers.Dense(NUM_CLASSES, activation="softmax"))
    return keras.Sequential(layer_stack, name=f"deep_mlp_list_{activation}")


def build_with_add(activation: str = "relu",
                   hidden_layers=(256, 128, 64)) -> keras.Model:
    """`model = Sequential(); model.add(layer)` constructor style."""
    model = keras.Sequential(name=f"deep_mlp_add_{activation}")
    model.add(layers.Input(shape=(*IMG_SHAPE, 1)))
    model.add(layers.Flatten())
    for u in hidden_layers:
        model.add(layers.Dense(u))
        model.add(_activation_layer(activation))
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    return model


def compile_with(model: keras.Model,
                 optimizer: keras.optimizers.Optimizer) -> keras.Model:
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def optimizer_factories(learning_rate: float | None = None
                        ) -> dict[str, Callable[[], keras.optimizers.Optimizer]]:
    """All optimizers used in the benchmark loop.

    If `learning_rate` is provided, it is forwarded to every optimizer so the
    LR x optimizer mini-grid can be built from the same registry.
    """
    def lr(opt_cls, **kwargs):
        if learning_rate is not None:
            kwargs.setdefault("learning_rate", learning_rate)
        return opt_cls(**kwargs)

    return {
        "SGD": lambda: lr(keras.optimizers.SGD),
        "SGD+momentum": lambda: lr(keras.optimizers.SGD, momentum=0.9),
        "SGD+nesterov": lambda: lr(keras.optimizers.SGD, momentum=0.9, nesterov=True),
        "RMSprop": lambda: lr(keras.optimizers.RMSprop),
        "Adam": lambda: lr(keras.optimizers.Adam),
        "Adagrad": lambda: lr(keras.optimizers.Adagrad),
        "Adadelta": lambda: lr(keras.optimizers.Adadelta),
        "Nadam": lambda: lr(keras.optimizers.Nadam),
    }
