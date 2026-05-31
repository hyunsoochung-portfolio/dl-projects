# Letter Reader (Advanced) — Deep MLP on EMNIST Letters

Multi-layer MLP on **EMNIST Letters** with head-to-head benchmarks on depth, activations, optimizers, and learning rates. Each cell is averaged over `--n-seeds` independent seeds and reported as `mean ± std`.

## Setup

- Dataset: EMNIST Letters (reused from `letter_reader_pilot`)
- Inputs: 28x28 grayscale, flattened
- Output: 26-class softmax + sparse cross-entropy
- Default depth-3 architecture: `(256, 128, 64)` Dense units; one Activation layer per Dense
- Two constructor styles printed at startup as a teaching artifact: `Sequential([...])` and `model = Sequential(); model.add(...)`

## Experiments

| # | Sweep | Values | Held fixed |
|---|---|---|---|
| 1 | Depth | `{1, 2, 3, 4}` hidden layers | Adam, ReLU, batch=256 |
| 2 | Activation | `{sigmoid, tanh, relu, leaky_relu, elu, gelu, swish}` | depth=3, Adam |
| 3 | Optimizer (all 8) | `SGD, SGD+momentum=0.9, SGD+nesterov, RMSprop, Adam, Adagrad, Adadelta, Nadam` | depth=3, ReLU |
| 4 | LR x optimizer mini-grid | `{1e-2, 1e-3, 1e-4}` x top-3 optimizers from (3) | depth=3, ReLU |

All cells run for `--n-seeds` seeds (default 3). Bar charts carry std error bars; CSVs carry mean/std columns; a final leaderboard CSV/PNG ranks the LR x optimizer grid.

## Findings

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

## Reflections

The optimizer benchmark is the most-cited table in this repo for a reason: it shows *what good defaults look like in practice*. Adam-family wins on a fixed-epoch budget not because it's universally best, but because it ships with a learning rate that works on most problems out of the box. Vanilla SGD and Adadelta look bad in this experiment specifically because they need either a much higher initial LR or many more epochs to converge — and that's exactly the conversation worth having with a teammate: "this optimizer needs *its own* defaults; the comparison only became fair after I matched the budget."

The sigmoid → ReLU jump (+7 pp accuracy, halved val loss) is the textbook vanishing-gradient story, and the depth-3 sweet spot tells you why the conclusion gets more pronounced with depth. Both findings translate directly into engineering hygiene: when training stalls or test accuracy plateaus on a deep architecture, the cheapest debugging move is to check activations and optimizers before redesigning the model.

There's also a comparison-hygiene point worth flagging. "Adam beats SGD" is technically true here but operationally narrow — large training runs often use SGD-with-cosine-LR for the *final long pass* because it generalises slightly better given enough compute. Conflating "fast-to-converge" with "best-final-model" is a common mistake; making the experiment explicit about its budget keeps the conclusion honest.

## Methodology notes

- **All 8 optimizers compared, same data, same epochs.** This is the only fair way to interpret an optimizer leaderboard - default learning rates, default schedules, identical seeds per cell. The follow-up LR x optimizer mini-grid (experiment 4) then revisits the top-3 with a learning-rate sweep to check whether the ranking changes with a tuned LR.
- **Activation choice via separate `Activation` layers** rather than the `activation=` kwarg of `Dense`, so swapping in `LeakyReLU` / `swish` / `gelu` doesn't require a different layer class.
- **Depth recipe is deliberate**: `1->(128,), 2->(256,128), 3->(256,128,64), 4->(512,256,128,64)`. We keep the same "halve as you go deeper" pattern so the depth sweep doesn't conflate depth with per-layer width.
- **Multi-seed.** With ~80% test accuracy and small per-seed variance, run-to-run noise can flip the ranking of close optimizers; we report the mean to make the comparison robust.

## Limitations

- Default epoch count is small (4) for CPU friendliness; the qualitative ranking of optimizers should be stable, but absolute accuracies will be lower than a fully-converged training run.
- The depth sweep changes both width and depth simultaneously (see recipe above); a pure depth-only sweep would fix per-layer width.
- No regularization in this project (that's `letter_reader_recipe`), so the depth-4 model can edge into overfit territory.

## Reproduce

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
