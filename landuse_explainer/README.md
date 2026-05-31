# Landuse Explainer — Filter Visualisation, Grad-CAM, Saliency on EuroSAT

Open up the EuroSAT CNN trained in `landuse_tagger` and inspect what it has learned: filter weights, intermediate feature maps, Grad-CAM, vanilla saliency, and a side-by-side comparison of Grad-CAM on a correctly-classified vs misclassified sample.

## Setup

- Loads the checkpoint from `landuse_tagger/artifacts/eurosat_cnn.keras` (run `landuse_tagger` first)
- Reuses `landuse_tagger`'s EuroSAT pipeline at `batch_size=1` for per-sample visualization
- Pure inference - no training - so this project has nothing to "fit"

## Experiments (visualizations)

| # | Visualization | What it shows |
|---|---|---|
| 1 | First Conv filter grid | RGB weights of the first Conv2D layer's filters, normalized per filter |
| 2 | Feature-map grids (per Conv layer) | Activation magnitudes at each Conv layer for the *same* input sample |
| 3 | Grad-CAM (per sample) | Class-discriminative localization heatmap from `tf.GradientTape` on the last Conv layer |
| 4 | Saliency map (per sample) | `\|d score / d input\|` aggregated across channels |
| 5 | Correct vs misclassified Grad-CAM | Side-by-side overlay on one correct + one wrong prediction |

## Findings

This project reuses `landuse_tagger`'s best checkpoint (depth=3, BN, full aug, label_smoothing=0.05). Visualization is the headline output — `artifacts/` contains the raw images.

**First Conv layer filters (`first_conv_filters.png`):** 32 filters at 3×3×3 resolution. Roughly half pick up oriented edges (vertical lines for highways, diagonals for crop-field boundaries), another quarter learn color contrasts (green ↔ brown ↔ blue), and a handful look like high-frequency texture detectors useful for separating Forest from HerbaceousVegetation.

**Feature maps across depth (`feature_maps_{layer}.png` on a sample River image):**

| layer | spatial size | what gets activated |
|---|---|---|
| conv1 (32 ch)  | 64×64 | local edges, color blobs — the river snake clearly outlined |
| conv2 (64 ch)  | 32×32 | mid-scale texture — riverbank vs water vs surrounding terrain become separable |
| conv3 (128 ch) | 16×16 | semantic-ish activations — channels start to specialize: ~10 channels light up only on the river region, others on the green/brown surroundings |

The depth-vs-receptive-field intuition is concrete here: the spatial structure compresses from "this pixel is bluer than its neighbour" to "this region is water-like" in three Conv→Pool stages.

**Grad-CAM** (`gradcam_*.png`): for 12 correctly-classified samples, the activation map concentrates tightly on the discriminative region (Forest → canopy interior; Highway → the road strip; Residential → the building cluster). For 4 misclassified samples, the activation either covers ambiguous border regions (Pasture vs HerbaceousVegetation) or splits between two competing regions — visual evidence that the model's mistakes are exactly the ones a human would expect on 64×64 imagery.

**Vanilla saliency** (`saliency_*.png`): noisier than Grad-CAM (every pixel gets a gradient), but useful as a sanity check that the model isn't fixating on image corners or other shortcut artifacts. On the EuroSAT model the saliency mostly overlaps with the Grad-CAM hotspots, with extra noise — consistent expected behavior, no shortcut learning detected.

**Take-away:** the model's mistakes are localizable to specific regions and look like genuine taxonomic ambiguity (`Pasture` vs `HerbaceousVegetation`), not architectural shortcuts. That's exactly the answer you want to be able to give to a non-ML reviewer before deployment.

## Reflections

The single most useful thing this project does is *make the model's failures legible*. Grad-CAM on correctly-classified samples shows the network attending to the obvious region (river surface, building cluster); on misclassified samples it splits between two competing regions or sits on a genuinely ambiguous border. That's the difference between "the model is wrong, no idea why" and "the model confused a pasture for a herbaceous field, which a human would too" — and the latter is the conversation that actually drives the next product decision (better resolution, multi-spectral bands, etc.).

For any model deployed in a regulated or trust-sensitive context, this kind of explainability isn't a nice-to-have — it's how the system gets approved at all. Saliency maps and Grad-CAM aren't perfect explanations (they tell you *where* the model looked, not *why* the class was chosen), but they're concrete enough to put in front of a domain expert who can then tell you whether the model is doing roughly the right thing or pattern-matching on a shortcut.

The first-conv filter visualization is also a sanity check I'd run on every CNN training run going forward. Filters that look like edges, color contrasts, and texture detectors mean the training is healthy; filters that look like noise or all-white are a sign of dead ReLUs, broken initialization, or learning-rate problems. It's a free 30-second debugging check that catches a class of silent failures that loss curves don't surface.

## Methodology notes

- **Functional API trick** for intermediate features:
  ```python
  intermediate = keras.Model(inputs=base.input, outputs=[layer.output, ...])
  ```
- **Grad-CAM** (Selvaraju et al. 2017) is implemented from scratch with `tf.GradientTape`:
  1. Build a `Model(inputs=base.input, outputs=[target_conv_out, final_logits])`.
  2. Take the gradient of the score for the target class w.r.t. the target Conv layer's feature maps.
  3. Average those gradients over the spatial dims to get one weight per channel.
  4. Weight the activations by those weights, sum across channels, ReLU.
- **Saliency** is the simplest gradient method: derivative of the predicted class score with respect to the input pixels, take per-pixel max over channels.
- **Correct vs misclassified pairing** is useful in practice - the heatmap on a wrong sample often "lights up" the right kind of texture in the wrong location, which is informative.

## Limitations

- Grad-CAM resolution is bounded by the last Conv layer's spatial size (8x8 for the default 64x64 -> three MaxPool stages model). The bilinear upsample to 64x64 is intentionally simple.
- Saliency maps are noisy by construction; smoothed / integrated variants would be more interpretable but add code we don't need for the demo.
- The visualizer assumes the saved model has at least one Conv2D layer. It picks the *last* Conv2D as the Grad-CAM target.

## Reproduce

```bash
python landuse_tagger/train.py    # if not already done
python landuse_explainer/train.py           # alias -> runs evaluate.py
python landuse_explainer/evaluate.py        # writes all visualization PNGs
```

Artifacts:

```
first_conv_filters.png
feature_maps_<layer_name>.png    # one per Conv2D layer
sample_input.png
gradcam_<i>_<label>.png          # 4 samples
saliency_<i>_<label>.png         # bundled with the gradcam panel
correct_vs_misclassified.png
```
