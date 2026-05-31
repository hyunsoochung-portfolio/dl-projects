"""Visualize the trained CNN: filters, feature maps, Grad-CAM, saliency.

Reads the EuroSAT checkpoint produced by `landuse_tagger` and writes a set of PNGs
under ``artifacts/``:

- ``first_conv_filters.png``       grid of first-layer Conv2D filter weights
- ``feature_maps_<layer>.png``     per-layer feature-map grids for one input
- ``sample_input.png``             the input used for feature-map/Grad-CAM
- ``gradcam_<i>_<label>.png``      Grad-CAM overlay on N test samples
- ``saliency_<i>_<label>.png``     saliency map overlay on the same samples
- ``correct_vs_misclassified.png`` side-by-side Grad-CAM for a correctly
                                   classified sample and a misclassified one
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import CHECKPOINT_PATH, CLASS_NAMES, load_tf_datasets  # noqa: E402
from model import (  # noqa: E402
    build_intermediate_model,
    conv_layers,
    first_filter_weights,
    grad_cam,
    last_conv_layer_name,
    load_trained_model,
    saliency_map,
)
from shared.utils import ensure_dir, grid_image, set_seed  # noqa: E402


def visualize_filters(model, out_dir: Path) -> Path:
    filters = first_filter_weights(model)
    out = out_dir / "first_conv_filters.png"
    grid_image(
        [f for f in filters],
        out_path=out,
        cols=8,
        titles=[f"f{i}" for i in range(len(filters))],
        cmap=None,
    )
    print(f"saved {out}")
    return out


def visualize_feature_maps(model, sample_image: np.ndarray,
                           sample_label: int, out_dir: Path) -> list[Path]:
    layer_names = [l.name for l in conv_layers(model)]
    intermediate = build_intermediate_model(model, layer_names)

    activations = intermediate.predict(sample_image[None, ...], verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    saved: list[Path] = []
    for name, act in zip(layer_names, activations):
        maps = act[0]
        channels = min(32, maps.shape[-1])
        images = [maps[:, :, c] for c in range(channels)]
        out_path = out_dir / f"feature_maps_{name}.png"
        grid_image(images, out_path=out_path, cols=8,
                   titles=[f"ch{c}" for c in range(channels)])
        saved.append(out_path)
        print(f"saved {out_path}")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(sample_image)
    ax.set_title(f"input: {CLASS_NAMES[sample_label]}")
    ax.axis("off")
    fig.tight_layout()
    inp_path = out_dir / "sample_input.png"
    fig.savefig(inp_path, dpi=140)
    plt.close(fig)
    saved.append(inp_path)
    return saved


def _overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Upsample heatmap to image size and overlay as a soft colormap."""
    import matplotlib.cm as cm
    import tensorflow as tf

    h, w = image.shape[:2]
    hm = tf.image.resize(heatmap[..., None], (h, w)).numpy()[..., 0]
    cmap = cm.get_cmap("jet")
    overlay = cmap(hm)[..., :3]
    return np.clip(0.55 * image + 0.45 * overlay, 0, 1)


def visualize_gradcam_and_saliency(model, samples, out_dir: Path,
                                   layer_name: str) -> None:
    import matplotlib.pyplot as plt

    for i, (img, true_lbl) in enumerate(samples):
        heatmap, pred = grad_cam(model, img, layer_name=layer_name)
        sal, _ = saliency_map(model, img)

        overlay = _overlay(img, heatmap)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(img); axes[0].set_title(f"input ({CLASS_NAMES[true_lbl]})")
        axes[1].imshow(overlay); axes[1].set_title(f"Grad-CAM -> {CLASS_NAMES[pred]}")
        axes[2].imshow(sal, cmap="hot"); axes[2].set_title("saliency |dscore/dx|")
        for a in axes: a.axis("off")
        fig.tight_layout()
        out = out_dir / f"gradcam_{i}_{CLASS_NAMES[true_lbl]}.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"saved {out}")


def visualize_correct_vs_misclassified(model, samples, out_dir: Path,
                                       layer_name: str) -> Path | None:
    """Find one correct + one misclassified test sample; overlay Grad-CAM side-by-side."""
    import matplotlib.pyplot as plt

    correct = None
    wrong = None
    for img, true_lbl in samples:
        _, pred = grad_cam(model, img, layer_name=layer_name)
        if pred == true_lbl and correct is None:
            correct = (img, true_lbl, pred)
        elif pred != true_lbl and wrong is None:
            wrong = (img, true_lbl, pred)
        if correct is not None and wrong is not None:
            break

    if correct is None or wrong is None:
        print("[gradcam] could not find both a correct and an incorrect sample - "
              "showing whichever was found")
        return None

    def overlay_for(img):
        hm, _ = grad_cam(model, img, layer_name=layer_name)
        return _overlay(img, hm)

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    axes[0, 0].imshow(correct[0]); axes[0, 0].set_title(
        f"CORRECT input ({CLASS_NAMES[correct[1]]})")
    axes[0, 1].imshow(overlay_for(correct[0])); axes[0, 1].set_title(
        f"Grad-CAM -> {CLASS_NAMES[correct[2]]}")
    axes[1, 0].imshow(wrong[0]); axes[1, 0].set_title(
        f"MISCLASSIFIED input ({CLASS_NAMES[wrong[1]]})")
    axes[1, 1].imshow(overlay_for(wrong[0])); axes[1, 1].set_title(
        f"Grad-CAM -> {CLASS_NAMES[wrong[2]]}")
    for a in axes.flat: a.axis("off")
    fig.tight_layout()
    out = out_dir / "correct_vs_misclassified.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"saved {out}")
    return out


def main():
    set_seed(42)
    artifacts = ensure_dir(HERE / "artifacts")
    model = load_trained_model(CHECKPOINT_PATH)
    model.summary()

    visualize_filters(model, artifacts)

    # collect a handful of test samples up-front
    _, _, test_ds = load_tf_datasets(batch_size=1)
    samples: list[tuple[np.ndarray, int]] = []
    for x, y in test_ds.take(8):
        samples.append((x.numpy()[0], int(y.numpy()[0])))

    visualize_feature_maps(model, samples[0][0], samples[0][1], artifacts)

    layer_name = last_conv_layer_name(model)
    visualize_gradcam_and_saliency(model, samples[:4], artifacts,
                                   layer_name=layer_name)
    visualize_correct_vs_misclassified(model, samples, artifacts,
                                       layer_name=layer_name)


if __name__ == "__main__":
    main()
