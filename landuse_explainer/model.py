"""Helpers for opening a trained CNN and exposing intermediate outputs."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from tensorflow import keras


def load_trained_model(checkpoint_path: Path) -> keras.Model:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"checkpoint missing: {checkpoint_path}\n"
            "run landuse_tagger/train.py first"
        )
    return keras.models.load_model(checkpoint_path)


def conv_layers(model: keras.Model) -> List[keras.layers.Layer]:
    return [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]


def build_intermediate_model(base: keras.Model,
                             layer_names: list[str]) -> keras.Model:
    """Functional API trick:

    intermediate = Model(inputs=base.input, outputs=[layer.output, ...])
    """
    outputs = [base.get_layer(name).output for name in layer_names]
    return keras.Model(inputs=base.input, outputs=outputs,
                       name="intermediate_outputs")


def first_filter_weights(model: keras.Model) -> np.ndarray:
    """Return (n_filters, H, W, 3) filter weights from the first Conv2D layer.

    For visualization we normalize each filter independently to [0, 1].
    """
    conv = conv_layers(model)[0]
    w = conv.get_weights()[0]  # (H, W, in_ch, out_ch)
    w = np.transpose(w, (3, 0, 1, 2))  # (out_ch, H, W, in_ch)
    out = []
    for f in w:
        lo, hi = f.min(), f.max()
        f = (f - lo) / (hi - lo + 1e-8)
        out.append(f)
    return np.stack(out, axis=0)


def last_conv_layer_name(model: keras.Model) -> str:
    """Name of the last Conv2D layer (the canonical Grad-CAM target)."""
    convs = conv_layers(model)
    if not convs:
        raise ValueError("model has no Conv2D layers")
    return convs[-1].name


def grad_cam(model: keras.Model,
             image: np.ndarray,
             class_index: int | None = None,
             layer_name: str | None = None) -> tuple[np.ndarray, int]:
    """Grad-CAM heatmap for `image` w.r.t. `class_index`.

    Standard recipe (Selvaraju et al. 2017): take the gradient of the score
    for `class_index` w.r.t. the feature maps of the target Conv layer,
    average over spatial dims for the channel weights, then apply a ReLU
    to the channel-weighted activations.

    Returns (heatmap [H, W] in [0, 1], predicted_class_index).
    """
    import tensorflow as tf

    if layer_name is None:
        layer_name = last_conv_layer_name(model)

    grad_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    inp = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(inp, training=False)
        pred_class = int(tf.argmax(predictions[0]).numpy())
        target_class = pred_class if class_index is None else int(class_index)
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, conv_out)            # (1, H, W, C)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_out = conv_out[0]                            # (H, W, C)
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)  # (H, W)
    heatmap = tf.nn.relu(heatmap).numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap, pred_class


def saliency_map(model: keras.Model,
                 image: np.ndarray,
                 class_index: int | None = None) -> tuple[np.ndarray, int]:
    """Vanilla saliency: |d score / d input| averaged across channels.

    Returns (saliency [H, W] in [0, 1], predicted_class_index).
    """
    import tensorflow as tf

    inp = tf.Variable(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(inp)
        predictions = model(inp, training=False)
        pred_class = int(tf.argmax(predictions[0]).numpy())
        target_class = pred_class if class_index is None else int(class_index)
        loss = predictions[:, target_class]

    grads = tape.gradient(loss, inp).numpy()[0]   # (H, W, C)
    saliency = np.max(np.abs(grads), axis=-1)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency, pred_class
