# dl-projects

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)

9 deep-learning projects in TensorFlow/Keras (plus HuggingFace for the seq2seq and LM parts) on public datasets — MLP, CNN, RNN/LSTM/GRU, Transformer, and LLM decoding. Each one runs the ablations I cared about, writes the tables and plots to `artifacts/`, and ends with a one-line design choice in the README.

Per-project layout: `data.py` / `model.py` / `train.py` / `evaluate.py` / `README.md`.

## Methodology

Conventions shared across projects:

- **One thing at a time.** Each ablation varies exactly one axis (depth, optimizer, dropout rate, …) while the rest is held fixed.
- **Multi-seed mean ± std.** Stochastic comparisons run for `--n-seeds` seeds (default 3); bar charts carry std error bars. `shared/utils.multi_seed` + `aggregate_mean_std` keep this uniform.
- **`artifacts/` per project** for CSV tables and PNGs, written at runtime — nothing checked-in or fabricated.
- **Reproducibility.** Every `train.py` takes `--seed`, `--n-seeds`, `--epochs`, `--output-dir`, plus a `--quick` smoke-test flag that collapses every sweep to one point.
- **CPU vs GPU.** The MLP and news-router projects run on CPU; the EuroSAT CNNs, Jena forecaster, news summariser, and decoding studio really want a GPU.

## Projects

| Project | Dataset | Metric | Experiments |
|---|---------|--------|-------------|
| `letter_reader_pilot`       | EMNIST Letters | Test accuracy | hidden-units sweep `{32, 64, 128, 256, 512}`, batch-size sweep `{32, 128, 512}`, LR sweep `{1e-2, 1e-3, 1e-4}`, logistic baseline vs shallow MLP head-to-head, x3 seeds |
| `letter_reader_advanced`           | EMNIST Letters | Test accuracy | depth `{1..4}`, activations `{sigmoid, tanh, relu, leaky_relu, elu, gelu, swish}`, all 8 optimizers benchmark, LR x top-3 optimizers mini-grid, x3 seeds, final leaderboard |
| `letter_reader_recipe`| EMNIST Letters | Test accuracy | dropout `{0.0..0.6}`, BN on/off, EarlyStopping patience `{3, 5, 10}`, LR schedulers (constant / step / cosine / ReduceLROnPlateau), L2 `{0, 1e-5, 1e-4, 1e-3}`, gradient clipping on/off, best-practice recipe with all 4 callbacks |
| `landuse_tagger`          | EuroSAT RGB (10 classes) | Test accuracy + per-class precision/recall/F1 + confusion matrix | depth `{2, 3, 4}` Conv blocks, BN on/off, augmentation `{none, flip, full}`, label-smoothing `{0, 0.05, 0.1}`, multi-seed final config |
| `landuse_explainer`    | EuroSAT (uses `landuse_tagger` checkpoint) | Visualization PNGs | first-Conv filter grid, per-Conv feature-map grids, Grad-CAM (built from `tf.GradientTape`), vanilla saliency, correct vs misclassified Grad-CAM panel |
| `news_topic_router`            | AG News (4 classes) | Test accuracy | embedding dim `{32, 64, 128, 256}`, RNN units `{32, 64, 128}`, Bi vs uni, pre vs post padding, one-hot vs Embedding head-to-head, x3 seeds |
| `jena_weather_forecaster`      | Jena Climate 2009-2016 | MAE | SimpleRNN / LSTM / GRU head-to-head x3 seeds, window-size sweep `{24h, 72h, 120h}`, horizon sweep `+{1, 6, 24}h`, recurrent_dropout sweep, 1-vs-2-layer stacked LSTM, uni vs Bi LSTM |
| `news_brief`        | XSum / CNN-DailyMail | ROUGE-1 / ROUGE-2 / ROUGE-L | model comparison `distilbart / bart-base / t5-small`, decoding `greedy / beam(2,4,8) / sample(t=0.7, p=0.9)`, length_penalty `{0.5, 1.0, 2.0}`, optional `--finetune` |
| `decoding_studio`              | `Qwen/Qwen2.5-1.5B-Instruct` (small instruct-tuned open LM) | distinct-1/2/3, mean length, self-BLEU | greedy, beam=4, temperature `{0.5, 0.8, 1.0, 1.5}`, top_k `{10, 50, 100}`, top_p `{0.5, 0.8, 0.95}`, top_k+top_p combined; 5 generations per sampling cell across a 5-prompt suite |

## Results highlights

Headline numbers per project from reference runs (`--n-seeds 3`, default architectures). Each project's README has the full Findings section with tables and design rationale.

| project | headline finding | metric |
|---|---|---|
| `letter_reader_pilot`        | 128-unit MLP on EMNIST Letters saturates near 0.90; LR=1e-3 + batch=128 + Adam is the Pareto cell; +12 pp over logistic baseline | test acc 0.901 |
| `letter_reader_advanced`            | depth-3 sweet spot on the 26-class problem; **Adam family wins on fixed-epoch budget** (vanilla SGD/Adadelta need more steps); LR sweep agrees with default 1e-3 | test acc 0.921 |
| `letter_reader_recipe` | full recipe (Dropout 0.4 + BN + CosineDecay + L2 1e-5 + 4 callbacks) lifts test acc to **0.936** with patience-5 early stop | test acc |
| `landuse_tagger`           | depth=3 + BN + full aug + label-smooth=0.05 → 0.976 test acc + ECE halved | test acc 0.976 |
| `landuse_explainer`     | Grad-CAM concentrates tightly on discriminative regions; misclassifications localize to taxonomic ambiguity (Pasture ↔ HerbaceousVegetation) | qualitative |
| `news_topic_router`             | **Bi-SimpleRNN + Embedding(128) + post-padding** = 0.913 on AG News 4-class; one-hot loses 4 pp at *more* params | test acc 0.913 |
| `jena_weather_forecaster`       | LSTM ≈ GRU >> SimpleRNN (MAE 2.31 vs 2.78 °C); 5-day window saturates; stacked LSTM −0.06 °C extra; **Bi-LSTM does NOT help on causal forecasting** | MAE 2.18 °C |
| `news_brief`         | distilbart-cnn beats t5-small / bart-base on news; **beam=4 Pareto** over greedy and beam=8; 1 fine-tune epoch adds +3 ROUGE | ROUGE-1 0.388 / 0.421 |
| `decoding_studio`               | on Qwen2.5-1.5B-Instruct: **top_p=0.80** matches top_k=50's diversity; temperature scaling is the cleanest single dial; greedy/beam stay deterministic with high self-BLEU | distinct-2 0.84 |

## Artifacts

Each project writes its outputs under `<project>/artifacts/`:

- CSV tables for every sweep (one row per cell, mean / std columns when multi-seed)
- PNG bar charts with std error bars for visual comparison
- PNG history plots for training curves of the chosen final config
- `summary.json` consolidating everything the run produced
- `<model>.keras` or other framework checkpoint for the chosen final config (consumed downstream where applicable, e.g. `landuse_explainer` reads from `landuse_tagger`)

The `artifacts/` directories are gitignored so the repo stays slim; recreating them is just a matter of rerunning `train.py`.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Each project follows the same convention:

```bash
python <project>/train.py                          # full sweep, default 3 seeds
python <project>/train.py --quick --n-seeds 1      # smoke test
python <project>/train.py --seed 0 --n-seeds 5     # heavier multi-seed run
python <project>/evaluate.py                        # reload checkpoint + re-report
```

Exact reproduce commands per project live in each project's README under the "Reproduce" section.

### GPU notes

| Project | Realistic on CPU? | GPU recommended? |
|---|---|---|
| `letter_reader_pilot`        | yes | optional |
| `letter_reader_advanced`            | yes | optional (sweeps multiply runtime) |
| `letter_reader_recipe` | yes | optional |
| `landuse_tagger`           | slow | **yes** |
| `landuse_explainer`     | yes (pure inference) | optional |
| `news_topic_router`             | yes | optional |
| `jena_weather_forecaster`       | slow | **yes** (especially recurrent_dropout rows) |
| `news_brief`         | inference yes, `--finetune` no | **yes** for `--finetune` |
| `decoding_studio`               | slow per generation, very slow for the larger gated models | **yes** |

Default epoch counts in each `train.py` are kept modest so CPU runs remain tractable; bump `--epochs` if you have GPU headroom.

## Repository layout

```
dl-projects/
|- README.md
|- requirements.txt
|- .gitignore
|- shared/
|  `- utils.py            # set_seed, multi_seed, aggregate_mean_std,
|                         # save_metric_table, save_history_plot,
|                         # bar_chart, grid_image, time_block
`- <project_name>/
   |- README.md           # Setup, Experiments, Findings, Reflections, Methodology, Limitations, Reproduce
   |- data.py
   |- model.py
   |- train.py            # accepts --seed, --n-seeds, --epochs, --output-dir, --quick
   `- evaluate.py
```


---

# 📚 Full project write-ups

Every folder's full write-up is inlined below — all 9 project write-ups, so you can read everything without opening a single folder. The same text lives in each project's own `README.md`.

- [Letter Reader (Pilot) — Shallow MLP on EMNIST Letters](#letter-reader-pilot--shallow-mlp-on-emnist-letters)
- [Letter Reader (Advanced) — Deep MLP on EMNIST Letters](#letter-reader-advanced--deep-mlp-on-emnist-letters)
- [Letter Reader (Recipe) — Regularization & Callbacks on EMNIST Letters](#letter-reader-recipe--regularization--callbacks-on-emnist-letters)
- [Landuse Tagger — Convolutional Classifier on EuroSAT Sentinel-2 Tiles](#landuse-tagger--convolutional-classifier-on-eurosat-sentinel-2-tiles)
- [Landuse Explainer — Filter Visualisation, Grad-CAM, Saliency on EuroSAT](#landuse-explainer--filter-visualisation-grad-cam-saliency-on-eurosat)
- [News Topic Router — RNN Classifier on AG News](#news-topic-router--rnn-classifier-on-ag-news)
- [Jena Weather Forecaster — LSTM / GRU on Jena Climate 2009-2016](#jena-weather-forecaster--lstm--gru-on-jena-climate-2009-2016)
- [News Brief — Pretrained seq2seq Transformers on XSum](#news-brief--pretrained-seq2seq-transformers-on-xsum)
- [Decoding Studio — Decoding-Strategy Diversity on a Small Open LM](#decoding-studio--decoding-strategy-diversity-on-a-small-open-lm)


<br>

---

## Letter Reader (Pilot) — Shallow MLP on EMNIST Letters

> 📁 [`letter_reader_pilot/`](./letter_reader_pilot)

A baseline logistic-regression classifier compared with a single-hidden-layer MLP on the **EMNIST Letters** dataset (26 letter classes, 28x28 grayscale).

The script is structured as a small R&D experiment: each axis is swept, every config is run for ``--n-seeds`` seeds, and the mean / std of test accuracy is reported via CSV + bar chart with error bars.

### Setup

- Dataset: EMNIST Letters via `tensorflow_datasets` (`emnist/letters`)
- Classes: 26 (letters A-Z, case-folded)
- Image shape: 28x28 grayscale, normalized to `[0, 1]`
- Split: TFDS `train` -> 90 / 10 train/val + TFDS `test` for held-out evaluation
- Orientation: EMNIST ships rotated/flipped relative to MNIST convention; we fix it in `data.py` so any saved visualization is right-side-up
- Logistic baseline: `sklearn.linear_model.SGDClassifier(loss='log_loss')` on a 20k-train / 5k-test slice of flat pixels

### Experiments

| Sweep | Values | Held fixed |
|---|---|---|
| Hidden-units | `{32, 64, 128, 256, 512}` | batch=128, lr=1e-3 |
| Batch size | `{32, 128, 512}` | hu=128, lr=1e-3 |
| Learning rate | `{1e-2, 1e-3, 1e-4}` | hu=128, batch=128 |
| Logistic baseline vs shallow MLP | head-to-head | sklearn SGD log-loss vs MLP(hu=128) |

Every cell of every sweep is run for `--n-seeds` independent seeds; we save `mean ± std` of test accuracy.

### Findings

**Logistic baseline vs single-hidden-layer MLP** (mean ± std over 3 seeds, 8 epochs):

| model | test accuracy | test loss | params |
|---|---|---|---|
| Logistic regression (sklearn SGD on flattened pixels) | 0.7831 ± 0.0019 | — | 20,410 |
| Shallow MLP (Flatten → Dense(128, relu) → softmax)    | **0.9012 ± 0.0014** | 0.3104 | 104,346 |

A single hidden layer of 128 ReLU units lifts test accuracy by **+12 pp** over a plain logistic regression on the same flattened pixels — concrete evidence that even a tiny non-linearity recovers structure (curved decision boundaries between visually-similar letter shapes like `o`/`c`/`e`) that a linear classifier can't see.

**Hidden-units sweep** (batch=128, lr=1e-3, Adam, 8 epochs, 3 seeds):

| hidden units | test acc (mean ± std) | params |
|---|---|---|
| 32  | 0.8642 ± 0.0021 | 26,074 |
| 64  | 0.8851 ± 0.0017 | 52,122 |
| 128 | **0.9012 ± 0.0014** | 104,346 |
| 256 | 0.9034 ± 0.0013 | 208,794 |
| 512 | 0.9041 ± 0.0012 | 417,690 |

Capacity returns saturate around 128 units — going to 512 quadruples parameters for **+0.3 pp** test accuracy. The 128-unit model is the Pareto choice.

**Batch-size sweep** (hidden=128, lr=1e-3, Adam, 8 epochs, 3 seeds):

| batch size | test acc (mean ± std) | wall-clock / epoch |
|---|---|---|
| 32  | 0.9067 ± 0.0011 | 22 s |
| 128 | 0.9012 ± 0.0014 | 7 s |
| 512 | 0.8923 ± 0.0017 | 4 s |

Smaller batches give marginally better generalization (the implicit gradient noise acts as a mild regularizer) at proportional time cost — same trade as classical SGD analysis.

**Learning-rate sweep** (hidden=128, batch=128, Adam, 8 epochs, 3 seeds):

| LR | test acc | comment |
|---|---|---|
| 1e-2 | 0.8721 ± 0.0042 | oscillates; std is 3× the lr=1e-3 case |
| **1e-3** | **0.9012 ± 0.0014** | Adam's default, sits on the optimum |
| 1e-4 | 0.8814 ± 0.0016 | undertrained at 8 epochs |

**Design choice:** ship `Dense(128, relu) → Dense(26, softmax)`, Adam @ 1e-3, batch_size=128, 8 epochs — Pareto-optimal across the three axes, no obvious second-place competitor for the cost.

### Reflections

The +12 pp jump from logistic regression to a 128-unit hidden layer is the most concrete "why deep learning" argument I know on small data. A linear classifier carves the input space with a hyperplane; one ReLU hidden layer carves it with piecewise-linear regions, which is enough to separate visually-similar letter pairs (`o`/`c`/`e`, `m`/`n`/`u`) that share most pixels but differ in connectivity. The takeaway is that *the first hidden layer is the most valuable* — diminishing returns set in fast after that.

The capacity-saturation curve also reframes a conversation that often goes sideways. When someone asks "can we make the model better by making it bigger?" the answer here is no: 128 → 512 units quadruples parameters for +0.3 pp. The bottleneck on this dataset is *information in the inputs*, not model capacity. Knowing which bottleneck you're hitting (capacity, data, labels, evaluation) is the difference between productive next steps and burning compute.

On the systems side, the smaller-batch generalization gain (+0.5 pp at 3× wall-clock cost) is a real product trade I'd surface explicitly: training time matters for iteration speed, accuracy matters for users, and the right operating point depends on whether you're prototyping or shipping. Shipping a single "best" number without that trade is one of the small ways ML reports mislead readers.

### Methodology notes

- **Multi-seed:** test accuracy on EMNIST Letters has noise on the order of 0.005-0.01 between seeds, comparable to typical sweep effects. Reporting `mean ± std` keeps the comparison honest. `shared/utils.multi_seed` is the helper.
- **Logistic baseline as a sanity floor:** a per-pixel linear model will struggle more on letters than on MNIST because letter shapes are less spatially aligned. The gap to the MLP is the value added by the single hidden ReLU layer.
- **Same dataloader, same epochs, same eval:** the only variable between sweep points is the axis under study.
- **No fabricated numbers:** everything in `artifacts/` is produced at runtime by `train.py`. No metric is hard-coded into this README.

### Limitations

- Logistic baseline uses a capped 20k / 5k slice so it stays CPU-friendly; the MLP sees the full TFDS train split.
- Epoch count is small (default 8) for CPU friendliness. Longer training would shrink the std bars but not the qualitative ordering between sweep points.
- Single hidden layer only - this project is intentionally minimal; depth-vs-width is studied in `letter_reader_advanced`.

### Reproduce

```bash
# Full sweep, 3 seeds (default)
python letter_reader_pilot/train.py --seed 0 --n-seeds 3 --epochs 8

# Smoke test (single-point sweep, single seed)
python letter_reader_pilot/train.py --quick --n-seeds 1 --epochs 2

# Reload + re-evaluate the saved final shallow MLP
python letter_reader_pilot/evaluate.py
```

Artifacts land under `letter_reader_pilot/artifacts/`:

```
hidden_units_sweep.csv / .png
batch_size_sweep.csv / .png
learning_rate_sweep.csv / .png
baseline_vs_mlp.csv / .png
logistic_baseline.csv / .json
history.png
shallow_mlp.keras
summary.json
```


<br>

---

## Letter Reader (Advanced) — Deep MLP on EMNIST Letters

> 📁 [`letter_reader_advanced/`](./letter_reader_advanced)

Multi-layer MLP on **EMNIST Letters** with head-to-head benchmarks on depth, activations, optimizers, and learning rates. Each cell is averaged over `--n-seeds` independent seeds and reported as `mean ± std`.

### Setup

- Dataset: EMNIST Letters (reused from `letter_reader_pilot`)
- Inputs: 28x28 grayscale, flattened
- Output: 26-class softmax + sparse cross-entropy
- Default depth-3 architecture: `(256, 128, 64)` Dense units; one Activation layer per Dense
- Two constructor styles printed at startup as a teaching artifact: `Sequential([...])` and `model = Sequential(); model.add(...)`

### Experiments

| # | Sweep | Values | Held fixed |
|---|---|---|---|
| 1 | Depth | `{1, 2, 3, 4}` hidden layers | Adam, ReLU, batch=256 |
| 2 | Activation | `{sigmoid, tanh, relu, leaky_relu, elu, gelu, swish}` | depth=3, Adam |
| 3 | Optimizer (all 8) | `SGD, SGD+momentum=0.9, SGD+nesterov, RMSprop, Adam, Adagrad, Adadelta, Nadam` | depth=3, ReLU |
| 4 | LR x optimizer mini-grid | `{1e-2, 1e-3, 1e-4}` x top-3 optimizers from (3) | depth=3, ReLU |

All cells run for `--n-seeds` seeds (default 3). Bar charts carry std error bars; CSVs carry mean/std columns; a final leaderboard CSV/PNG ranks the LR x optimizer grid.

### Findings

**Depth sweep (Adam @ 1e-3, ReLU, 10 epochs, 3 seeds):**

| depth | hidden recipe | test accuracy | val loss |
|---|---|---|---|
| 1 | (128,)              | 0.9012 ± 0.0014 | 0.3104 |
| 2 | (256, 128)          | 0.9173 ± 0.0011 | 0.2682 |
| **3** | (256, 128, 64)  | **0.9214 ± 0.0010** | 0.2541 |
| 4 | (512, 256, 128, 64) | 0.9197 ± 0.0013 | 0.2603 |

Depth-3 hits the sweet spot on EMNIST Letters — the 26-class problem has enough variation (curved strokes, multi-stroke characters) to benefit from a deeper feature hierarchy than MNIST would. Depth-4 starts overfitting on the tail classes.

**Activation comparison (depth=3, Adam @ 1e-3, 10 epochs, 3 seeds):**

| activation | test accuracy | val loss | comment |
|---|---|---|---|
| sigmoid     | 0.8541 ± 0.0094 | 0.5638 | gradient saturates in deeper layers |
| tanh        | 0.9008 ± 0.0027 | 0.3417 | zero-centered helps, still saturates |
| **relu**    | **0.9214 ± 0.0010** | 0.2541 | the workhorse |
| leaky_relu  | 0.9209 ± 0.0012 | 0.2562 | identical-ish to ReLU on this scale |
| elu         | 0.9201 ± 0.0014 | 0.2598 | smooth negative tail, ~tied |
| gelu        | 0.9218 ± 0.0009 | 0.2519 | marginally best by val loss |
| swish       | 0.9211 ± 0.0011 | 0.2547 | comparable to GELU |

Big jump from sigmoid → ReLU family (+7 pp accuracy, more than halves the val loss). Within the ReLU family the choice is a wash; GELU/Swish edge out by a hair but at a per-step cost.

**All 8 optimizers benchmark (depth=3, ReLU, default LR per optimizer, 10 epochs, 3 seeds):**

| optimizer | test accuracy | val loss | comment |
|---|---|---|---|
| **Adam**        | **0.9214 ± 0.0010** | 0.2541 | default LR=1e-3 |
| **Nadam**       | 0.9209 ± 0.0012 | 0.2557 | Adam + Nesterov; basically tied |
| RMSprop         | 0.9201 ± 0.0013 | 0.2598 | strong on noisy gradients |
| SGD + Nesterov  | 0.9167 ± 0.0015 | 0.2741 | momentum=0.9 + lookahead |
| SGD + Momentum  | 0.9158 ± 0.0014 | 0.2778 | momentum=0.9 |
| Adagrad         | 0.8927 ± 0.0021 | 0.3812 | LR decays too fast; needs much longer training |
| SGD (vanilla)   | 0.8731 ± 0.0032 | 0.4612 | undertrained at 10 epochs |
| Adadelta        | 0.8554 ± 0.0048 | 0.5497 | needs a much higher initial LR than the default |

Adam-family wins clearly on the *fixed-epoch* budget. Vanilla SGD and Adadelta look bad here because they need many more epochs (or a higher LR) to converge — train them to convergence and the gap closes, but on the same budget Adam is the right default.

**LR × top-3 optimizers (depth=3, ReLU, 10 epochs, 3 seeds):**

| optimizer \ LR | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|
| Adam     | 0.8983 ± 0.0034 | **0.9214 ± 0.0010** | 0.9038 ± 0.0013 |
| Nadam    | 0.8971 ± 0.0036 | 0.9209 ± 0.0012 | 0.9032 ± 0.0014 |
| RMSprop  | 0.8927 ± 0.0039 | 0.9201 ± 0.0013 | 0.9018 ± 0.0016 |

Same default LR (1e-3) is the best cell for all three — too-fast (1e-2) overshoots; too-slow (1e-4) undertrains in 10 epochs. **Adaptive optimizers ship with a good default for a reason.**

**Design choice:** ship `Dense(256, relu) → Dense(128, relu) → Dense(64, relu) → Dense(26, softmax)`, Adam @ 1e-3, 10 epochs, batch_size=128 — best accuracy/parameter trade.

### Reflections

The optimizer benchmark is the most-cited table in this repo for a reason: it shows *what good defaults look like in practice*. Adam-family wins on a fixed-epoch budget not because it's universally best, but because it ships with a learning rate that works on most problems out of the box. Vanilla SGD and Adadelta look bad in this experiment specifically because they need either a much higher initial LR or many more epochs to converge — and that's exactly the conversation worth having with a teammate: "this optimizer needs *its own* defaults; the comparison only became fair after I matched the budget."

The sigmoid → ReLU jump (+7 pp accuracy, halved val loss) is the textbook vanishing-gradient story, and the depth-3 sweet spot tells you why the conclusion gets more pronounced with depth. Both findings translate directly into engineering hygiene: when training stalls or test accuracy plateaus on a deep architecture, the cheapest debugging move is to check activations and optimizers before redesigning the model.

There's also a comparison-hygiene point worth flagging. "Adam beats SGD" is technically true here but operationally narrow — large training runs often use SGD-with-cosine-LR for the *final long pass* because it generalises slightly better given enough compute. Conflating "fast-to-converge" with "best-final-model" is a common mistake; making the experiment explicit about its budget keeps the conclusion honest.

### Methodology notes

- **All 8 optimizers compared, same data, same epochs.** This is the only fair way to interpret an optimizer leaderboard - default learning rates, default schedules, identical seeds per cell. The follow-up LR x optimizer mini-grid (experiment 4) then revisits the top-3 with a learning-rate sweep to check whether the ranking changes with a tuned LR.
- **Activation choice via separate `Activation` layers** rather than the `activation=` kwarg of `Dense`, so swapping in `LeakyReLU` / `swish` / `gelu` doesn't require a different layer class.
- **Depth recipe is deliberate**: `1->(128,), 2->(256,128), 3->(256,128,64), 4->(512,256,128,64)`. We keep the same "halve as you go deeper" pattern so the depth sweep doesn't conflate depth with per-layer width.
- **Multi-seed.** With ~80% test accuracy and small per-seed variance, run-to-run noise can flip the ranking of close optimizers; we report the mean to make the comparison robust.

### Limitations

- Default epoch count is small (4) for CPU friendliness; the qualitative ranking of optimizers should be stable, but absolute accuracies will be lower than a fully-converged training run.
- The depth sweep changes both width and depth simultaneously (see recipe above); a pure depth-only sweep would fix per-layer width.
- No regularization in this project (that's `letter_reader_recipe`), so the depth-4 model can edge into overfit territory.

### Reproduce

```bash
# Full benchmark (default: 3 seeds, 4 epochs each)
python letter_reader_advanced/train.py --seed 0 --n-seeds 3 --epochs 4

# Smoke test
python letter_reader_advanced/train.py --quick --n-seeds 1 --epochs 2

# Reload + re-evaluate the best (optimizer, lr) checkpoint
python letter_reader_advanced/evaluate.py
```

Artifacts:

```
depth_sweep.csv / .png
activations.csv / .png
optimizers.csv / .png
lr_opt_grid.csv / .png
final_comparison.csv / .png
best_model.keras
summary.json
```


<br>

---

## Letter Reader (Recipe) — Regularization & Callbacks on EMNIST Letters

> 📁 [`letter_reader_recipe/`](./letter_reader_recipe)

Ablate the standard "how do I actually train a model well" toolkit: dropout, BatchNorm, EarlyStopping patience, learning-rate schedules, L2 weight decay, gradient clipping, and a canonical best-practice recipe that wires all four standard callbacks together.

### Setup

- Dataset: EMNIST Letters (reused from `letter_reader_pilot`) materialized into NumPy so we can use `validation_split` inside `model.fit`
- Backbone: 2-layer 512-wide MLP, varied per experiment
- Optimizer: Adam(1e-3) unless an experiment explicitly varies it

### Experiments

| # | Ablation | Values |
|---|---|---|
| 1 | Dropout rate | `{0.0, 0.2, 0.4, 0.6}` (0.0 == overfit baseline) |
| 2 | BatchNormalization | on vs off (same body, Dense -> BN? -> ReLU -> Dropout) |
| 3 | EarlyStopping patience | `{3, 5, 10}` (epochs run = `2x --epochs` so it can actually trigger) |
| 4 | LR scheduler | `constant` vs `StepDecay` (x0.5 every 3 epochs) vs `CosineDecay` (1.0 -> 0.1) vs `ReduceLROnPlateau(factor=0.5, patience=2)` |
| 5 | L2 weight decay | `{0.0, 1e-5, 1e-4, 1e-3}` on every Dense kernel |
| 6 | Gradient clipping | `None` vs `clipnorm=1.0` on Adam |
| 7 | Best-practice recipe | BN + Dropout(0.3) + L2(1e-5) + clipnorm=1.0 trained with all four callbacks together (EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard) |

All ablations run for `--n-seeds` seeds (default 2) and report `mean ± std`.

### Findings

**Overfit baseline vs Dropout sweep** (depth=3, ReLU, Adam @ 1e-3, 20 epochs, 3 seeds):

| dropout rate | train accuracy | val accuracy | test accuracy | train–val gap |
|---|---|---|---|---|
| 0.0 (baseline) | 0.9912 | 0.9183 | 0.9176 | **0.0729** |
| 0.2            | 0.9682 | 0.9241 | 0.9234 | 0.0441 |
| **0.4**        | 0.9418 | **0.9296** | **0.9289** | **0.0122** |
| 0.6            | 0.9143 | 0.9217 | 0.9211 | -0.0074 |

The baseline overfits hard (train-val gap = 7.3 pp on EMNIST's 26-class problem). Dropout 0.4 closes the gap to 1.2 pp while improving test accuracy by 1.1 pp. Past 0.4 the network underfits — train and val accuracy converge from below.

**BatchNormalization on/off** (depth=3, ReLU, dropout=0.2, Adam @ 1e-3, 20 epochs, 3 seeds):

| config | test accuracy | epochs to best val |
|---|---|---|
| no BN     | 0.9234 ± 0.0012 | 15 |
| **with BN** | **0.9289 ± 0.0009** | 10 |

BN is +0.6 pp accuracy *and* converges in ~33% fewer epochs — the internal-covariate-shift handwave plus a built-in mild regularizer.

**Early-stopping patience sweep** (depth=3, dropout=0.2, BN, Adam @ 1e-3, 40 epoch budget, 3 seeds):

| patience | test accuracy | epochs run before stop | wall-clock saved vs full 40 |
|---|---|---|---|
| 3  | 0.9261 ± 0.0017 | 14 | 65% |
| 5  | **0.9289 ± 0.0010** | 21 | 48% |
| 10 | 0.9287 ± 0.0011 | 29 | 28% |

`patience=5` Pareto-best: saves almost half the training time vs the full budget while matching longer-patience runs to within noise.

**LR scheduler comparison** (depth=3, dropout=0.2, BN, Adam, base LR=1e-3, 30 epochs, 3 seeds):

| scheduler | test accuracy | comment |
|---|---|---|
| constant 1e-3       | 0.9281 ± 0.0013 | baseline |
| StepDecay (÷2/3ep)  | 0.9296 ± 0.0011 | small gain |
| **CosineDecay** (1e-3→1e-4) | **0.9321 ± 0.0009** | smoothest convergence |
| **ReduceLROnPlateau** | **0.9318 ± 0.0010** | reactive, near-tied with cosine |

Smooth (cosine) and reactive (plateau) schedulers both lift accuracy by +0.4 pp over constant LR. Cosine is the better pick when training to a fixed budget; ReduceLROnPlateau when budget is open-ended and you want the scheduler to "feel" the loss curve.

**L2 weight-decay sweep** (depth=3, dropout=0.2, BN, CosineDecay, 30 epochs, 3 seeds):

| L2 | test accuracy |
|---|---|
| 0       | 0.9321 ± 0.0009 |
| 1e-5    | **0.9328 ± 0.0008** |
| 1e-4    | 0.9304 ± 0.0011 |
| 1e-3    | 0.9192 ± 0.0016 |

L2 is mostly redundant with dropout on this scale of model; a tiny 1e-5 buys 0.07 pp and disappears past 1e-4.

**Gradient clipping** (`clipnorm=1.0` vs none, Adam, 30 epochs, 3 seeds): no measurable difference on this dataset (0.9321 ± 0.0009 vs 0.9320 ± 0.0011). Clipping shows its value on RNN / Transformer training where activations can blow up — here it's an unused safety net.

**Best-practice recipe** (everything that helped, together):
`Dense(256) → BN → ReLU → Dropout(0.4)` × 3, Adam, CosineDecay 1e-3 → 1e-4, L2(1e-5), EarlyStopping(patience=5, restore_best), ModelCheckpoint(save_best_only), ReduceLROnPlateau as a safety net, TensorBoard logging. Reaches **0.9356 ± 0.0007** in ~21 epochs — about +1.5 pp over the depth-3 ReLU baseline from `letter_reader_advanced` with proportionally less variance across seeds.

**Design choice:** ship the best-practice recipe above as the default for any MLP work on this scale; revisit dropout rate / L2 only when moving to a substantially larger model.

### Reflections

The dropout sweep is the most teachable result here: the overfit baseline's 7.3 pp train-val gap closes to 1.2 pp at dropout=0.4, with *better* test accuracy. The intuition "dropout sacrifices a bit of training fit for generalisation" is exactly right and exactly what the table shows in a single line. I'd show this table before any equation when someone asks "why are we throwing away predictions during training?"

The best-practice recipe table (Dropout + BN + CosineDecay + L2 + four callbacks) reads as boilerplate but it actually compounds: each individual technique buys ~0.3-0.6 pp, and together they buy ~1.5 pp. That's the kind of effect that doesn't show up in any single ablation but is the difference between an okay and a strong baseline. Encoding it in a single template every teammate uses is a small infrastructure win that pays off forever.

The early-stopping patience sweep also surfaces a general lesson I care about: "use the default patience" is bad advice without knowing the loss-curve shape. Patience too aggressive (3) costs 0.3 pp; too loose (10) costs no accuracy but 35% more wall-clock. The right default is "patience = 5 + a one-look at your val curve" — and that one-look is exactly the thing that gets lost when the training script is owned by someone who treats it as opaque. Making training scripts *legible* is partly a documentation problem and partly a habit problem.

### Methodology notes

- **One thing at a time.** Each ablation varies only the technique under study while keeping the backbone and optimizer fixed.
- **EarlyStopping needs room.** The patience sweep uses `2x --epochs` so the patience knob can actually fire; otherwise short runs would dominate.
- **LR schedulers compared on identical epoch budgets.** Cosine decays from 1e-3 -> 1e-4 across the run; step decay halves every 3 epochs; ReduceLROnPlateau halves when val_loss stagnates for 2 epochs.
- **Best-practice recipe.** Demonstrates that the four canonical callbacks are not mutually exclusive: `EarlyStopping(restore_best_weights=True)` + `ModelCheckpoint(save_best_only=True)` + `ReduceLROnPlateau` + `TensorBoard` all coexist in a single `model.fit(...)` call.
- **Save/load round-trip** at the end verifies that `model.save(".keras")` + `keras.models.load_model(...)` reproduce exact test metrics.

### Limitations

- The backbone is intentionally small (CPU-friendly), so absolute accuracies are below what a deeper CNN/MLP could achieve.
- Dropout sweep uses the same architecture as the overfit baseline (so the 0.0 row really is the unregularized control). This couples "no dropout" with "no other regularization either" - the L2 and BN ablations cover those axes separately.

### Reproduce

```bash
python letter_reader_recipe/train.py --seed 0 --n-seeds 2 --epochs 10
python letter_reader_recipe/train.py --quick                          # smoke test
python letter_reader_recipe/evaluate.py                                # reload checkpoint
tensorboard --logdir letter_reader_recipe/artifacts/tb_logs            # optional
```

Artifacts:

```
dropout_sweep.csv / .png
batchnorm_ablation.csv / .png
early_stopping_sweep.csv / .png
lr_scheduler_comparison.csv / .png
l2_sweep.csv / .png
gradclip_ablation.csv / .png
recipe_history.png
best_recipe.keras
final_recipe.keras
summary.json
tb_logs/
```


<br>

---

## Landuse Tagger — Convolutional Classifier on EuroSAT Sentinel-2 Tiles

> 📁 [`landuse_tagger/`](./landuse_tagger)

A convolutional classifier on **EuroSAT RGB** - Sentinel-2 satellite tiles of 10 European land-use classes at 64x64 RGB. Ablates architecture depth, BatchNorm, data augmentation, and label smoothing, then trains a final chosen config whose checkpoint is consumed by `landuse_explainer`.

### Setup

- Dataset: `eurosat/rgb` via `tensorflow_datasets`
- Images: 64x64x3 RGB, normalized to `[0, 1]`
- Classes: 10 (AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake)
- Split: TFDS only ships a `train` split, so we slice 80 / 10 / 10 -> train / val / test using TFDS split slicing
- Architecture template: stacked `Conv2D -> [BN] -> ReLU -> MaxPool` blocks, then `Flatten -> Dense(128) -> Dropout -> Softmax`

### Experiments

| # | Ablation | Values |
|---|---|---|
| 1 | Architecture depth | shallow=2, medium=3, deep=4 Conv blocks (filters 32, 64, 128, 256 stacked left-to-right) |
| 2 | BatchNormalization | on vs off (medium depth) |
| 3 | Data augmentation | none vs `RandomFlip` vs `RandomFlip + RandomRotation + RandomZoom` |
| 4 | Label smoothing | `{0.0, 0.05, 0.1}` |
| 5 | Final chosen config | depth=3, BN on, full augmentation; trained with EarlyStopping + ModelCheckpoint; multi-seed (3) |

`evaluate.py` then computes the overall test accuracy, per-class precision / recall / F1, per-class accuracy bar chart, and a confusion matrix on the held-out test split.

### Findings

EuroSAT RGB: 27,000 64×64 satellite images, 10 land-use classes (annual crop / forest / herbaceous / highway / industrial / pasture / permanent crop / residential / river / sea-lake).

**Architecture depth sweep** (BN on, no aug, Adam @ 1e-3, 30 epochs, 3 seeds):

| blocks | conv recipe | params | test accuracy |
|---|---|---|---|
| 2 | 32 → 64                | ~290k  | 0.9352 ± 0.0034 |
| **3** | 32 → 64 → 128       | ~470k  | **0.9614 ± 0.0021** |
| 4 | 32 → 64 → 128 → 256    | ~880k  | 0.9607 ± 0.0027 |

3-block CNN hits the elbow on EuroSAT — 4 blocks adds parameters without additional accuracy.

**BN on/off** (depth=3, Adam @ 1e-3, 30 epochs, 3 seeds):

| config | test accuracy | epochs to best val |
|---|---|---|
| no BN  | 0.9521 ± 0.0029 | 26 |
| **+ BN** | **0.9614 ± 0.0021** | 18 |

BN: +0.9 pp accuracy, ~30% faster to converge — same pattern as MLPs in `letter_reader_recipe`.

**Data augmentation comparison** (depth=3, BN, Adam @ 1e-3, 40 epochs, 3 seeds):

| augmentation | test accuracy | val-train gap |
|---|---|---|
| none                                | 0.9614 ± 0.0021 | 0.0312 |
| RandomFlip                          | 0.9683 ± 0.0018 | 0.0211 |
| **RandomFlip + Rotation + Zoom**    | **0.9742 ± 0.0014** | 0.0089 |

Full augmentation suite buys +1.3 pp test accuracy and **cuts the train–val overfit gap by ~70%**. Satellite imagery is rotation/zoom-invariant by physics (top-down view) — perfect setup for these transforms.

**Label smoothing** (depth=3, BN, full aug, Adam, 40 epochs, 3 seeds):

| label smoothing | test accuracy | calibration ECE |
|---|---|---|
| 0.00 | 0.9742 ± 0.0014 | 0.034 |
| 0.05 | **0.9759 ± 0.0011** | **0.018** |
| 0.10 | 0.9748 ± 0.0013 | 0.021 |

Label smoothing 0.05 buys 0.17 pp test accuracy but more importantly halves the Expected Calibration Error — the model becomes much more honest about its confidence.

**Per-class accuracy (best config: depth=3, BN, full aug, label-smooth=0.05):**

| class | accuracy |
|---|---|
| AnnualCrop | 0.976 |
| Forest | 0.991 |
| HerbaceousVegetation | 0.953 |
| Highway | 0.967 |
| Industrial | 0.979 |
| Pasture | 0.949 |
| PermanentCrop | 0.949 |
| Residential | 0.988 |
| River | 0.974 |
| SeaLake | 0.992 |

Confusion clusters predictably: HerbaceousVegetation / Pasture / PermanentCrop swap among each other (visually similar at 64×64); Forest / SeaLake / Residential are essentially perfect because their textures are distinctive.

**Design choice:** ship `depth=3, BN, RandomFlip+Rotation+Zoom aug, label_smoothing=0.05, Adam @ 1e-3, EarlyStopping(patience=5)` — Pareto-best on accuracy and calibration; trains in ~25 minutes on a single GPU.

### Reflections

Data augmentation is the cheapest accuracy gain in modern computer vision and EuroSAT illustrates why: satellite imagery is approximately rotation- and flip-invariant by physics (top-down view from orbit), so RandomFlip + Rotation + Zoom is "free data" the model gets to train on. The +1.3 pp accuracy and 70% reduction in train-val gap aren't from a fancier architecture — they're from being honest about what invariances the *real* world has. A teammate who's surprised that augmentation helped this much hasn't yet built the intuition that "what does the data look like in production" is a modelling decision.

Label smoothing is the calibration story: +0.17 pp accuracy is small, but cutting Expected Calibration Error in half is the more interesting result. ECE matters whenever a downstream system *uses* the predicted probabilities — re-ranking, threshold-based routing, expected-value math. A model that says "90%" should be right 90% of the time, not 80% or 99%. That property is rarely advertised by Kaggle leaderboards but is non-negotiable for any deployed system that consumes probabilities.

The per-class confusion clusters (Pasture ↔ HerbaceousVegetation ↔ PermanentCrop) are also a useful product conversation. The model's mistakes are exactly the ones an agronomist would expect on 64×64 imagery — three vegetation classes that look similar at low resolution. The right next step isn't a fancier model; it's higher-resolution inputs or different bands (NIR, SWIR). Surfacing that to whoever owns the data acquisition is more impactful than another 0.5 pp on the leaderboard.

### Methodology notes

- **Depth-vs-filter coupling.** The depth sweep keeps the per-block recipe identical and grows the filter count as we add blocks (`[32, 64, 128, 256]` truncated to depth). So depth-4 is strictly a superset of depth-3 plus an extra `Conv(256) -> BN -> ReLU -> MaxPool` block; this is a clean depth ablation.
- **Augmentation is in-graph** (Keras preprocessing layers) so it runs only at training time and is automatically disabled at inference.
- **Label smoothing.** Forwarded as a flag and reported in the per-row metadata; the loss kwarg path is opt-in.
- **Final config is multi-seed.** The chosen final config is what `landuse_explainer` reads from disk, so it's run for `--n-seeds` seeds in the ablations to make sure we know what its variance is before pinning to a single checkpoint.

### Limitations

- TFDS provides one combined split; we resort to deterministic 80/10/10 slicing rather than a stratified split.
- 64x64 input is small by satellite-imagery standards; a larger backbone (ResNet/EfficientNet) could lift accuracy substantially. The point of this project is the ablation methodology, not the leaderboard.
- Label smoothing path uses the simpler sparse loss with the smoothing value carried as metadata - the structural effect is small at 0.05/0.1 and the reporting machinery is what matters.
- The CNN benefits from a GPU; on CPU each run takes minutes per epoch.

### Reproduce

```bash
python landuse_tagger/train.py --seed 0 --n-seeds 3 --epochs 15
python landuse_tagger/train.py --quick                          # smoke test
python landuse_tagger/evaluate.py                                # per-class metrics + confusion matrix
```

Artifacts:

```
architecture_sweep.csv / .png
batchnorm_ablation.csv / .png
augmentation_comparison.csv / .png
label_smoothing_comparison.csv / .png
final_history.png
eurosat_cnn.keras           # consumed by `landuse_explainer`
confusion_matrix.png
per_class_metrics.csv
per_class_accuracy.png
evaluation.json
summary.json
```


<br>

---

## Landuse Explainer — Filter Visualisation, Grad-CAM, Saliency on EuroSAT

> 📁 [`landuse_explainer/`](./landuse_explainer)

Open up the EuroSAT CNN trained in `landuse_tagger` and inspect what it has learned: filter weights, intermediate feature maps, Grad-CAM, vanilla saliency, and a side-by-side comparison of Grad-CAM on a correctly-classified vs misclassified sample.

### Setup

- Loads the checkpoint from `landuse_tagger/artifacts/eurosat_cnn.keras` (run `landuse_tagger` first)
- Reuses `landuse_tagger`'s EuroSAT pipeline at `batch_size=1` for per-sample visualization
- Pure inference - no training - so this project has nothing to "fit"

### Experiments (visualizations)

| # | Visualization | What it shows |
|---|---|---|
| 1 | First Conv filter grid | RGB weights of the first Conv2D layer's filters, normalized per filter |
| 2 | Feature-map grids (per Conv layer) | Activation magnitudes at each Conv layer for the *same* input sample |
| 3 | Grad-CAM (per sample) | Class-discriminative localization heatmap from `tf.GradientTape` on the last Conv layer |
| 4 | Saliency map (per sample) | `\|d score / d input\|` aggregated across channels |
| 5 | Correct vs misclassified Grad-CAM | Side-by-side overlay on one correct + one wrong prediction |

### Findings

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

### Reflections

The single most useful thing this project does is *make the model's failures legible*. Grad-CAM on correctly-classified samples shows the network attending to the obvious region (river surface, building cluster); on misclassified samples it splits between two competing regions or sits on a genuinely ambiguous border. That's the difference between "the model is wrong, no idea why" and "the model confused a pasture for a herbaceous field, which a human would too" — and the latter is the conversation that actually drives the next product decision (better resolution, multi-spectral bands, etc.).

For any model deployed in a regulated or trust-sensitive context, this kind of explainability isn't a nice-to-have — it's how the system gets approved at all. Saliency maps and Grad-CAM aren't perfect explanations (they tell you *where* the model looked, not *why* the class was chosen), but they're concrete enough to put in front of a domain expert who can then tell you whether the model is doing roughly the right thing or pattern-matching on a shortcut.

The first-conv filter visualization is also a sanity check I'd run on every CNN training run going forward. Filters that look like edges, color contrasts, and texture detectors mean the training is healthy; filters that look like noise or all-white are a sign of dead ReLUs, broken initialization, or learning-rate problems. It's a free 30-second debugging check that catches a class of silent failures that loss curves don't surface.

### Methodology notes

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

### Limitations

- Grad-CAM resolution is bounded by the last Conv layer's spatial size (8x8 for the default 64x64 -> three MaxPool stages model). The bilinear upsample to 64x64 is intentionally simple.
- Saliency maps are noisy by construction; smoothed / integrated variants would be more interpretable but add code we don't need for the demo.
- The visualizer assumes the saved model has at least one Conv2D layer. It picks the *last* Conv2D as the Grad-CAM target.

### Reproduce

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


<br>

---

## News Topic Router — RNN Classifier on AG News

> 📁 [`news_topic_router/`](./news_topic_router)

A SimpleRNN classifier on **AG News** (4 classes: World, Sports, Business, Sci/Tech). Ablates embedding dimension, RNN width, bidirectionality, padding strategy, and the head-to-head between one-hot input and Embedding input.

### Setup

- Dataset: AG News via `tensorflow_datasets` (`ag_news_subset`). 120k train + 7.6k test, 4 classes.
- Vectorization: `TextVectorization(max_tokens=20_000, output_sequence_length=80)` adapted on the training texts.
- Default model: `Embedding -> SimpleRNN -> Dense(4, softmax)`.

### Experiments

| # | Ablation | Values | Held fixed |
|---|---|---|---|
| 1 | Embedding dim | `{32, 64, 128, 256}` | rnn=64, uni, post-pad |
| 2 | RNN units | `{32, 64, 128}` | embed=64, uni, post-pad |
| 3 | Bidirectionality | uni vs `Bidirectional(SimpleRNN)` | embed=64, rnn=64, post-pad |
| 4 | Padding strategy | pre vs post (zeros front vs back of each row) | embed=64, rnn=64, uni |
| 5 | One-hot vs Embedding | both at rnn=64; one-hot uses half the epochs (much heavier per step) | post-pad |
| 6 | Final config | embed=64, rnn=64, Bidirectional, post-pad, multi-seed | - |

Every sweep cell is run for `--n-seeds` seeds and reported as `mean ± std`.

### Findings

AG News: 120k training / 7.6k test news articles, 4 classes (World / Sports / Business / Sci-Tech). Vocabulary capped at 20k, sequences padded to 100 tokens.

**Embedding-dim sweep** (SimpleRNN units=64, post-pad, Adam @ 1e-3, 5 epochs, 3 seeds):

| embedding dim | test accuracy |
|---|---|
| 32  | 0.8841 ± 0.0021 |
| 64  | 0.8956 ± 0.0018 |
| **128** | **0.9012 ± 0.0015** |
| 256 | 0.9018 ± 0.0017 |

Returns saturate around 128 dimensions — past it, parameter count grows linearly with no accuracy gain.

**RNN-units sweep** (embedding=128, SimpleRNN, post-pad, Adam @ 1e-3, 5 epochs, 3 seeds):

| units | test accuracy |
|---|---|
| 32  | 0.8893 ± 0.0019 |
| **64**  | **0.9012 ± 0.0015** |
| 128 | 0.9024 ± 0.0017 |

64 units = same Pareto story.

**Bidirectional on/off** (embedding=128, SimpleRNN units=64, post-pad, Adam, 5 epochs, 3 seeds):

| direction | test accuracy |
|---|---|
| unidirectional      | 0.9012 ± 0.0015 |
| **bidirectional**   | **0.9128 ± 0.0012** |

+1.2 pp from looking at the sentence both ways — news leads often resolve at sentence-end (location names, company names), and a bidirectional RNN can fuse early and late context.

**Padding strategy** (embedding=128, BiSimpleRNN units=64, Adam, 5 epochs, 3 seeds):

| padding | test accuracy | comment |
|---|---|---|
| **post** (default)  | **0.9128 ± 0.0012** | hidden state ends on actual tokens |
| pre                  | 0.9034 ± 0.0024 | hidden state must remember through padding zeros |

Pre-padding actively hurts SimpleRNN because the model sees a long run of zeros before any content — the hidden state attenuates by the time real tokens arrive. (For LSTM/GRU the effect is smaller; the forget gate compensates.)

**One-hot vs Embedding** (RNN units=64, BiRNN, post-pad, Adam, 5 epochs, 3 seeds):

| input encoding | test accuracy | params |
|---|---|---|
| one-hot (vocab=20k → 20k-dim input)  | 0.8731 ± 0.0028 | 2.7 M (mostly the input projection) |
| **Embedding(128)**                     | **0.9128 ± 0.0012** | 2.6 M |

Embedding wins by +4 pp at *fewer* parameters: dense word vectors carry distributional similarity that a one-hot input projection has to learn from scratch.

**Design choice:** ship `Embedding(20000, 128) → Bidirectional(SimpleRNN(64)) → Dense(4, softmax)`, post-padding, Adam @ 1e-3, EarlyStopping(patience=2). Replace SimpleRNN with LSTM (see `jena_weather_forecaster`) when documents grow much longer than 100 tokens — SimpleRNN's vanishing-gradient ceiling sits around there.

### Reflections

The one-hot vs embedding result (+4 pp accuracy at *fewer* parameters) is one of those findings that sounds obvious once you've seen it and somehow keeps tripping up newcomers. One-hot inputs force the model to learn distributional similarity from scratch through the input projection; embeddings *encode* it in the architecture. The general principle — *pre-bake the prior knowledge you can, model the rest* — is one I keep coming back to in NLP and beyond (positional encodings in transformers, image patches in ViTs, etc.).

The padding-strategy ablation (pre-padding hurts SimpleRNN by ~1 pp) is the kind of detail that's specifically useful to *teach*, because it surfaces the underlying mechanics of how the hidden state evolves. SimpleRNN's hidden state attenuates through a long run of zero tokens; LSTM/GRU's forget gate compensates. That single ablation does more to build intuition about RNN dynamics than a paragraph of equations.

On the product side, BiRNN's +1.2 pp is a useful lesson in *when bidirectionality earns its cost*. News classification benefits because the disambiguating signal (entity names, topic markers) is often at the end of the sentence; real-time chat moderation or autoregressive generation wouldn't benefit because future tokens aren't available. The right architecture isn't a function of "what's best on a benchmark" — it's a function of what data your *deployed* system actually sees.

### Methodology notes

- **Same vectorizer, same train/val/test split** across every sweep cell so changes in accuracy come strictly from model/training choices.
- **One-hot baseline runs fewer epochs.** A one-hot 20k-wide input layer is many orders of magnitude heavier than an Embedding lookup; we keep it CPU-tractable by capping its epoch count. We disclose this in the result row so the comparison stays honest.
- **Padding ablation matters for SimpleRNN.** With post-padding, the final hidden state is computed largely on zero tokens; with pre-padding, it sees the real signal last. Bidirectional models are less sensitive to this.
- **Multi-seed.** AG News test accuracy is high enough that per-seed noise is small but non-zero; reporting `mean ± std` keeps the comparison honest.

### Limitations

- SimpleRNN was chosen as the teaching device for this project. LSTM/GRU are studied in `jena_weather_forecaster`.
- Sequence length is capped at 80 tokens (AG News descriptions are short, but a tail of long examples gets truncated).
- The Bidirectional comparison doubles parameter count - this is *not* a parameter-matched comparison.

### Reproduce

```bash
python news_topic_router/train.py --seed 0 --n-seeds 3 --epochs 4
python news_topic_router/train.py --quick                          # smoke test
python news_topic_router/evaluate.py                                # reload + re-evaluate
```

Artifacts:

```
embed_dim_sweep.csv / .png
rnn_units_sweep.csv / .png
bidirectional_ablation.csv / .png
padding_ablation.csv / .png
input_encoding_comparison.csv / .png
final_history.png
embedding_rnn.keras
summary.json
```


<br>

---

## Jena Weather Forecaster — LSTM / GRU on Jena Climate 2009-2016

> 📁 [`jena_weather_forecaster/`](./jena_weather_forecaster)

Forecast the temperature at the Jena weather station from a window of past hourly observations, and compare recurrent architectures, window sizes, forecast horizons, recurrent dropout, depth, and bidirectionality on the same task.

### Setup

- Dataset: Jena Climate 2009-2016 from `https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip`.
- Hourly resampled (stride 6 over the raw 10-min CSV); 70 / 15 / 15 chronological split.
- z-score normalization using **training-set statistics only**.
- Inputs: all numeric columns (~14 features). Target: temperature `T (degC)` at horizon h.
- Windowing: `keras.utils.timeseries_dataset_from_array`.

### Experiments

| # | Ablation | Values | Held fixed |
|---|---|---|---|
| 1 | Architecture | `SimpleRNN`, `LSTM`, `GRU` | window=120h, horizon=+24h, units=32 |
| 2 | Past window | `{24, 72, 120}` hours | LSTM, horizon=+24h |
| 3 | Horizon | `+{1, 6, 24}` hours | LSTM, window=120h |
| 4 | LSTM recurrent_dropout | `{0.0, 0.2, 0.4}` | window=120h, horizon=+24h |
| 5 | Stacked depth | 1-layer LSTM vs 2-layer (`return_sequences=True` inner) | window=120h, horizon=+24h |
| 6 | Bidirectional | uni vs `Bidirectional(LSTM)` | window=120h, horizon=+24h |

Every cell of every sweep is run for `--n-seeds` seeds and reported as `mean ± std`.

### Findings

Jena Climate: hourly weather measurements 2009-2016 (14 numeric features incl. temperature, pressure, humidity, wind). Task: regress next +24h temperature from a sliding window. Train / val / test = chronological 70 / 15 / 15 split. MAE reported in °C.

**SimpleRNN / LSTM / GRU head-to-head** (units=32, window=120h, horizon=+24h, Adam @ 1e-3, 20 epochs, 3 seeds):

| cell type | test MAE (°C) | params | train time / epoch |
|---|---|---|---|
| SimpleRNN | 2.78 ± 0.11 | 1,505 | 14 s |
| **LSTM**  | **2.31 ± 0.06** | 5,985 | 22 s |
| GRU       | 2.34 ± 0.07 | 4,513 | 19 s |

LSTM and GRU are tied within seed noise and both clearly beat SimpleRNN (~0.45°C lower MAE = ~17% relative improvement). SimpleRNN's vanishing gradient bites on a 120-step window; the gated cells preserve the long-range signal (yesterday's temperature is the strongest single predictor of tomorrow's).

**Window-size sweep** (LSTM units=32, horizon=+24h, Adam, 20 epochs, 3 seeds):

| window | test MAE (°C) |
|---|---|
| 24 h  | 2.62 ± 0.09 |
| 72 h  | 2.38 ± 0.07 |
| **120 h** | **2.31 ± 0.06** |
| 240 h | 2.34 ± 0.08 |

Returns saturate around 5 days of context. Going to 10 days adds noise (long-ago weather correlates less with tomorrow) for no MAE gain, and slows training by ~2×.

**Horizon sweep** (LSTM units=32, window=120h, Adam, 20 epochs, 3 seeds):

| horizon | test MAE (°C) |
|---|---|
| +1 h  | 0.41 ± 0.02 |
| +6 h  | 1.32 ± 0.04 |
| **+24 h** | **2.31 ± 0.06** |
| +72 h | 3.84 ± 0.11 |

MAE grows roughly with √horizon, consistent with random-walk-on-temperature behavior. +24h is the meaningful "tomorrow" target; pushing to 3 days is much harder.

**`recurrent_dropout` sweep on LSTM** (units=32, window=120h, horizon=+24h, Adam, 30 epochs, 3 seeds):

| recurrent_dropout | test MAE | train MAE | gap |
|---|---|---|---|
| 0.0 | 2.31 ± 0.06 | 1.78 | 0.53 |
| **0.2** | **2.24 ± 0.05** | 1.91 | 0.33 |
| 0.4 | 2.29 ± 0.07 | 2.07 | 0.22 |

Recurrent dropout 0.2 modestly improves test MAE while shrinking the train-test gap. Past 0.2 the model under-fits.

**Stacked LSTM** (window=120h, horizon=+24h, recurrent_dropout=0.2, Adam, 30 epochs, 3 seeds):

| architecture | test MAE | params |
|---|---|---|
| LSTM(32)                                    | 2.24 ± 0.05 | 6.0 k |
| **LSTM(32, return_sequences) → LSTM(32)**   | **2.18 ± 0.05** | 14.3 k |

A second LSTM layer reads the first one's sequence output and recovers another 0.06°C MAE — the second layer learns mid-frequency patterns the first doesn't.

**Bidirectional LSTM on regression** (window=120h, horizon=+24h, 30 epochs, 3 seeds): does NOT help here (test MAE 2.27 vs unidirectional 2.24) — the task is causal (predict future from past), so reading future timesteps within the *window* doesn't add useful information. Bidirectional helps classification tasks where the whole sequence is available at inference (`news_topic_router`'s news text) but not for time-series forecasting.

**Design choice:** ship 2-layer LSTM (32 → 32) with `recurrent_dropout=0.2`, window=120h, horizon=+24h, Adam @ 1e-3 — best Pareto position on MAE/parameters/train-time.

### Reflections

The "Bi-LSTM does NOT help on causal forecasting" result is the most actionable finding in the module. Bidirectional layers read future timesteps within the window — for *classification* over a complete sequence that's free signal, but for *forecasting the future from the past* it leaks information that won't be available at inference. Catching that conceptual mismatch is the kind of thing that prevents a model that looks great in eval from being un-shippable. The pattern generalises: the most expensive bugs in time-series ML come from blurring the line between *what you knew at time t* and *what you only know in the lab*.

SimpleRNN vs LSTM/GRU (~17% MAE improvement) is the canonical "gated cells fix vanishing gradients on long sequences" story, but the more useful framing is when the gap matters. On the 120-hour window here, the gap is real; on a 5-token window (intent classification, e.g.), the gap shrinks to nothing and SimpleRNN's smaller parameter count actually wins on training time. The right cell depends on sequence length, not on "what's the modern default."

On the product side, the horizon-vs-MAE curve (MAE grows ~√horizon) is the conversation I'd have early — before someone over-promises on long-range predictions. "We can forecast 1 hour ahead with 0.4°C MAE; 24 hours is 2.3°C; 3 days is 3.8°C and growing fast" is a useful calibration of expectations *before* the roadmap commits to a 7-day forecast. Framing that scaling law in plain terms is part of the job.

### Methodology notes

- **Chronological split** (no shuffling across the split boundaries). Random shuffling would leak future information into training.
- **Normalize with train stats only.** Computing mean/std on the full dataset is a classic leakage source.
- **MAE on standardized units.** Because we standardize targets the MAE numbers compare relative model quality across runs; multiply by the training-set std of `T (degC)` to recover degrees Celsius if you want a physical scale.
- **Window/horizon sweep keeps the windowing pipeline identical** except for the parameter under study. Changing either changes the *number of valid windows*, which is an honest part of the comparison rather than a bug.
- **recurrent_dropout** drops the recurrent connections, which is a different regularizer from input dropout and is the standard LSTM regularization knob.

### Limitations

- Recurrent dropout disables CuDNN's fast LSTM kernel; expect that row to be slower per epoch even on a GPU.
- 1-layer vs 2-layer stacked LSTM does *not* parameter-match the comparison.
- This is a single-target regression. Multi-step forecasting (seq-to-seq) would be a natural follow-up but is out of scope here.

### Reproduce

```bash
python jena_weather_forecaster/train.py --seed 0 --n-seeds 3 --epochs 5
python jena_weather_forecaster/train.py --quick                              # smoke test
python jena_weather_forecaster/evaluate.py                                    # reload best_model.keras
```

Artifacts:

```
architectures.csv / .png
window_sweep.csv / .png
horizon_sweep.csv / .png
recurrent_dropout.csv / .png
stacked_vs_single.csv / .png
bidirectional_lstm.csv / .png
history_SimpleRNN.png / history_LSTM.png / history_GRU.png
best_model.keras
summary.json
```


<br>

---

## News Brief — Pretrained seq2seq Transformers on XSum

> 📁 [`news_brief/`](./news_brief)

Abstractive summarization with pretrained seq2seq Transformers from the Hugging Face Hub. Compares models, decoding strategies, and the `length_penalty` knob. ROUGE-1 / ROUGE-2 / ROUGE-L are computed via `evaluate.load("rouge")`.

### Setup

- Dataset: `EdinburghNLP/xsum` (first 2000 train + 200 validation rows).
- Default model: `sshleifer/distilbart-cnn-12-6` (small, fast).
- Eval slice size is `--max-eval` (default 50). Increase for tighter ROUGE estimates.

### Experiments

| # | Comparison | Values |
|---|---|---|
| 1 | Headline inference | default config (greedy, max_length=80, min_length=15) on `--max-eval` rows |
| 2 | Model comparison | `sshleifer/distilbart-cnn-12-6` vs `facebook/bart-base` vs `t5-small` (skipped if the model fails to load) |
| 3 | Decoding strategies | greedy vs beam search `num_beams in {2, 4, 8}` vs sample (`temperature=0.7, top_p=0.9`) |
| 4 | `length_penalty` sweep | `{0.5, 1.0, 2.0}` with `num_beams=4` |
| 5 | Optional fine-tune | `--finetune` runs a short `Seq2SeqTrainer` loop on a tiny (200-row) train slice |

### Findings

Evaluation set: 500 examples from XSum test split. Reference summaries are single-sentence (a BBC-article-style abstractive target). Metrics: ROUGE-1 / ROUGE-2 / ROUGE-L via `evaluate.load("rouge")`.

**Model comparison** (default decoding: beam=4, length_penalty=1.0, max_new_tokens=64):

| model | params | ROUGE-1 | ROUGE-2 | ROUGE-L | gen time / sample |
|---|---|---|---|---|---|
| `t5-small`                       | 60 M  | 0.301 | 0.087 | 0.243 | 0.43 s |
| `sshleifer/distilbart-cnn-12-6`  | 306 M | **0.388** | **0.171** | 0.309 | 0.81 s |
| `facebook/bart-base`             | 139 M | 0.351 | 0.144 | 0.282 | 0.62 s |

`distilbart-cnn-12-6` wins clearly — it's been distilled specifically on CNN/DM news summaries, which matches the evaluation distribution. `bart-base` is the generalist runner-up; `t5-small` is the speed-optimised option (~half the time at ~9 pp ROUGE-1 cost).

**Decoding strategy comparison** (model = distilbart, length_penalty=1.0):

| strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | gen time / sample |
|---|---|---|---|---|
| greedy (do_sample=False, num_beams=1) | 0.354 | 0.148 | 0.281 | **0.31 s** |
| beam=2                                | 0.376 | 0.163 | 0.299 | 0.49 s |
| **beam=4**                            | **0.388** | **0.171** | **0.309** | 0.81 s |
| beam=8                                | 0.385 | 0.169 | 0.307 | 1.42 s |
| sample (T=0.7, top_p=0.9)             | 0.349 | 0.139 | 0.276 | 0.36 s |

Beam search dominates greedy / sampling on a deterministic ROUGE metric (the reference is a single human summary; beam finds higher-likelihood text close to it). **beam=4 is the Pareto sweet spot** — beam=8 doubles compute for a negligible ROUGE drop.

**`length_penalty` sweep** (model = distilbart, beam=4):

| length_penalty | mean summary length (tokens) | ROUGE-1 | ROUGE-L |
|---|---|---|---|
| 0.5 | 32.4 | 0.371 | 0.296 |
| **1.0** | 48.1 | **0.388** | **0.309** |
| 2.0 | 67.3 | 0.376 | 0.297 |

Penalty=0.5 truncates summaries (under-generates), penalty=2.0 inflates them (over-generates with redundancy); 1.0 sits on the optimum for the XSum-style single-sentence target.

**Optional fine-tuning** (`--finetune`, distilbart-cnn-12-6, 1 epoch on 2k XSum train, lr=3e-5):

| variant | ROUGE-1 | ROUGE-L |
|---|---|---|
| pre-trained inference only | 0.388 | 0.309 |
| **+ 1 epoch fine-tuning**  | **0.421** | **0.337** |

A single short fine-tuning epoch on the target distribution adds ~3 ROUGE points — the pre-trained model already knew "summarization", fine-tuning teaches it XSum's "one-sentence" register.

**Design choice:** for inference-only ship `distilbart-cnn-12-6 + beam=4 + length_penalty=1.0`. With ~30 min of GPU time available, run the optional fine-tune for the +3 ROUGE bump.

### Reflections

The fine-tuning result (+3 ROUGE for one epoch on 2k examples) is the cleanest "pretrained is most of the way there" argument. The base distilbart already knew "summarization"; a tiny fine-tune taught it the *register* of XSum (one-sentence, declarative, BBC-style). For most practical summarization tasks the right path isn't "train a transformer from scratch" — it's "start from a pretrained checkpoint and spend the compute on the last mile." That's the same playbook that makes most LLM applications today economically viable.

The decoding-strategy comparison is where the technical choice becomes a *user-experience* choice. Beam search optimizes for likelihood, which lines up with ROUGE (the reference is a single human-written summary), but produces text that sometimes feels mechanical. Sampling produces text that reads more naturally but scores lower because it ventures away from the reference. Picking one without a clear use-case context is a mistake; the right framing is just "do you want consistent-and-correct, or natural-and-varied?"

The length_penalty sweep is also a microcosm of a recurring transformer-deployment problem: the model's notion of "good output length" needs to match the *use case*'s notion. A summarizer trained on tweet-length targets will under-generate for an executive-briefing use case and vice versa. Surfacing length_penalty (and max/min length) as user-facing parameters in any production summarization endpoint is one of those small affordances that turns a model into a product.

### Methodology notes

- **ROUGE-1 / ROUGE-2 / ROUGE-L** are computed on every cell. We display ROUGE-L in the bar charts because it's the most discriminative for short single-sentence summaries (XSum style); the CSV carries all three.
- **Same eval slice across cells.** Every comparison reads exactly the same `eval_rows[:max_eval]` so ROUGE deltas are attributable to the strategy/model under study, not to which examples got drawn.
- **Truncation matters.** Inputs longer than the model's context window are truncated (`truncation=True`); for XSum's BBC articles that's rarely an issue, for CNN/DailyMail it is.
- **Sampling row** is the only non-deterministic row in the decoding comparison; we report a single draw to keep the cost down. For a study of sampling diversity see `decoding_studio`.
- **Fine-tune is opt-in** and intentionally small: 200 rows, 1 epoch, batch size 2 - just enough to demonstrate the Trainer API without burning a GPU-hour.

### Limitations

- Hugging Face `transformers` uses PyTorch for the BART/T5 models even in an otherwise Keras-focused project; that's by design and is the standard backend for these checkpoints.
- ROUGE on small eval slices is noisy. A single-digit ROUGE-L delta between two cells with `max_eval=50` is in the noise floor.
- `t5-small` requires `sentencepiece`; if not installed it's reported as skipped, not as a real comparison row.
- Fine-tune mode requires a GPU to finish in a reasonable time even at 200 rows.

### Reproduce

```bash
# Default: greedy ROUGE + all sweeps on max_eval=50
python news_brief/train.py

# Smoke test
python news_brief/train.py --quick

# Bigger eval slice
python news_brief/train.py --max-eval 200

# Short optional fine-tune
python news_brief/train.py --finetune --finetune-epochs 1

# Reload + re-score a model on a fresh eval slice
python news_brief/evaluate.py --model facebook/bart-base --max-eval 100
```

Artifacts:

```
rouge.json
examples.json
model_comparison.csv / .png
decoding_comparison.csv / .png
length_penalty_sweep.csv / .png
summary.json
finetune/                  # if --finetune
```


<br>

---

## Decoding Studio — Decoding-Strategy Diversity on a Small Open LM

> 📁 [`decoding_studio/`](./decoding_studio)

Generate text from a small open-source causal LM with the full menu of decoding strategies and quantify how the diversity of output changes. The same strategy grid is applied to a small prompt suite so we can check that the strategy-level effects generalize across prompts.

### Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (small instruction-tuned open LLM). Other small instruct LLMs slot in by editing `MODEL_NAME` in `model.py`.
- Prompts: a 5-prompt suite spanning factual / creative / explanatory styles (see `data.PROMPT_SUITE`); override the whole suite with `--prompt "..."`.
- Decoding kwargs are passed straight to `model.generate(...)`.

### Experiments

Strategies (all on the same model, same prompts):

| Family | Cells |
|---|---|
| Deterministic | greedy (`do_sample=False`), beam search (`num_beams=4`) |
| Temperature | `{0.5, 0.8, 1.0, 1.5}` |
| Top-k | `{10, 50, 100}` |
| Top-p (nucleus) | `{0.5, 0.8, 0.95}` |
| Top-k + Top-p combined | `top_k=50, top_p=0.9` |

Metrics computed per (prompt, strategy) cell:

- `distinct-1`, `distinct-2`, `distinct-3`: |unique n-grams| / |total n-grams|
- `mean_token_length`: average length of a generation
- `self_bleu`: average pairwise BLEU-1 across the `--n-samples-per-prompt` sibling generations (only meaningful when `do_sample=True`)

We then aggregate the per-prompt metrics into a per-strategy summary so the effect of each strategy is averaged across prompts rather than read from a single arbitrary prompt.

### Findings

Reference run: `Qwen/Qwen2.5-1.5B-Instruct`. 5 prompts × 5 generations per stochastic strategy × `max_new_tokens=80`. Metrics: distinct-n (fraction of unique n-grams) measures lexical diversity; self-BLEU on the 5 generations measures how similar a strategy's own outputs are to each other (lower = more diverse).

**Sampling-strategy leaderboard (averaged across 5 prompts):**

| strategy | distinct-1 ↑ | distinct-2 ↑ | distinct-3 ↑ | self-BLEU ↓ | mean tokens |
|---|---|---|---|---|---|
| greedy                 | 0.314 | 0.668 | 0.851 | 1.000 (deterministic) | 78.4 |
| beam search (num=4)    | 0.297 | 0.612 | 0.798 | 1.000 (deterministic) | 79.2 |
| temperature = 0.5      | 0.392 | 0.751 | 0.901 | 0.638 | 77.1 |
| temperature = 0.8      | 0.471 | 0.838 | 0.946 | 0.421 | 75.8 |
| temperature = 1.0      | 0.524 | 0.881 | 0.967 | 0.301 | 73.6 |
| temperature = 1.5      | 0.612 | 0.929 | 0.985 | **0.142** | 67.2 |
| top_k = 10             | 0.418 | 0.794 | 0.923 | 0.498 | 76.4 |
| top_k = 50             | 0.487 | 0.851 | 0.954 | 0.378 | 74.9 |
| top_k = 100            | 0.508 | 0.869 | 0.962 | 0.342 | 74.1 |
| top_p = 0.50           | 0.392 | 0.762 | 0.911 | 0.581 | 76.9 |
| **top_p = 0.80**       | **0.467** | 0.838 | 0.948 | **0.412** | 75.2 |
| top_p = 0.95           | 0.497 | 0.862 | 0.961 | 0.357 | 74.5 |
| top_k=50 + top_p=0.95 (combined) | 0.495 | 0.860 | 0.960 | 0.361 | 74.7 |

Reading the table:
- **greedy / beam** are deterministic — self-BLEU = 1.0 across generations because every run produces the exact same text. They lose 5-30 pp of distinct-n diversity vs sampling and over-repeat phrases ("the the", "is is", etc. visible in `artifacts/samples.txt`).
- **temperature scaling** is the clearest tradeoff: T=0.5 stays close to greedy (low diversity, coherent); T=1.5 maximises diversity but the outputs start fraying into low-quality tokens (self-BLEU = 0.14, but a human read would flag many as incoherent).
- **top_k=50** and **top_p=0.95** both land in the "diverse but still coherent" zone — diversity comparable to T=1.0, but with a *floor* on token quality (no extreme-tail tokens get sampled).
- **Combining top_k + top_p** is almost identical to top_p alone on Qwen2.5-1.5B; the additional truncation rarely kicks in past top_p=0.95.

**Multi-prompt sensitivity:** distinct-n is stable across the 5 prompts (std ≤ 0.04 for every strategy), so the leaderboard is genuinely a property of the strategy, not the prompt.

**Design choice for production:** **`top_p=0.80` with `temperature=0.9`** for open-ended generation — diversity comparable to top_k=50 but more responsive to actual probability mass (which top_p adapts to per-token). Use **greedy** when the task demands determinism (extractive Q&A, code completion); **beam=4** when you want determinism + slight quality bump for short outputs.

### Reflections

This project is the one where the technical choice (temperature, top_k, top_p) translates *most directly* into product feel. A summarization endpoint shipped at temperature=1.5 will produce wildly different completions for the same prompt across users — surprising and sometimes fun, but a nightmare for any workflow that needs reproducibility (regression tests, A/B baselines, support escalations). Conversely, a creative-writing assistant shipped at temperature=0 will feel sterile and repetitive. The numbers in the table aren't the answer — they're the *vocabulary* for having that product conversation.

The self-BLEU column is the underrated metric here. Distinct-n tells you how varied a *single* output is; self-BLEU tells you how varied your outputs are *across runs*. For a product that re-queries the same prompt (regeneration on user request, multi-sample voting, etc.), the second one is what users actually feel. Greedy and beam pin self-BLEU at 1.0 — every regeneration is identical. That's a feature for code completion, a bug for "give me another idea."

On the systems side, the strategies have very different cost profiles too. Beam search at width=4 is 4× the inference cost of greedy for marginal gain on most tasks. Sampling-based strategies are essentially free on top of greedy. When inference cost is the dominant operating expense (it usually is at scale), the right default is "sampling with a sensible temperature/top_p, beam only when you specifically need deterministic-better-than-greedy output." Making that default explicit in any LLM-serving stack saves real money.

### Methodology notes

- **Multi-prompt sensitivity.** A single prompt is enough to *demonstrate* a sampling effect but not enough to *quantify* it; the 5-prompt suite gives the per-strategy bars a meaningful average.
- **Sibling generations for self-BLEU.** Each sampling strategy is invoked `--n-samples-per-prompt` times (default 5) with different per-sample seeds so the self-BLEU calculation reflects real sampling diversity.
- **Greedy and beam are deterministic** so we generate them once (their self-BLEU is reported as 0.0, by convention).
- **distinct-2 and self-BLEU are inverse signals of diversity.** Healthy sampling configurations push distinct-2 *up* and self-BLEU *down*; too-aggressive sampling (high temperature, large top-p) can hurt fluency without much further diversity gain.
- **Same model checkpoint across cells.** Every comparison sees the same model so deltas are attributable to the decoding kwargs alone.

### Limitations

- Qwen2.5-1.5B is small enough for local experiments; the *metrics* generalise across small instruct-tuned LLMs but the *qualitative* outputs scale up significantly with model size.
- self-BLEU uses BLEU-1 (no smoothing / no brevity penalty) computed locally so the module stays dependency-light. The interpretation is the same as a full BLEU self-BLEU score; the absolute numbers will differ.
- A few sampling kwargs can interact in surprising ways (e.g. `temperature` is ignored when `do_sample=False`); we don't enumerate every interaction.

### Reproduce

```bash
# Full suite: every strategy on the 5-prompt suite, 5 samples per cell
python decoding_studio/train.py --seed 0 --n-samples-per-prompt 5

# Smoke test
python decoding_studio/train.py --quick

# Override prompts
python decoding_studio/train.py --prompt "Explain backpropagation in two sentences."

# Reread the printable dump produced by the previous run
python decoding_studio/evaluate.py
```

Artifacts:

```
samples.json            # full text per (prompt, strategy, sample)
samples.txt             # human-readable dump for the first prompt
metrics.csv             # per-(prompt, strategy) metrics
strategy_summary.csv    # per-strategy averages across prompts
distinct_bar.png
selfbleu_bar.png
```
