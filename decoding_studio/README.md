# Decoding Studio — Decoding-Strategy Diversity on a Small Open LM

Generate text from a small open-source causal LM with the full menu of decoding strategies and quantify how the diversity of output changes. The same strategy grid is applied to a small prompt suite so we can check that the strategy-level effects generalize across prompts.

## Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct` (small instruction-tuned open LLM). Other small instruct LLMs slot in by editing `MODEL_NAME` in `model.py`.
- Prompts: a 5-prompt suite spanning factual / creative / explanatory styles (see `data.PROMPT_SUITE`); override the whole suite with `--prompt "..."`.
- Decoding kwargs are passed straight to `model.generate(...)`.

## Experiments

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

## Findings

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

## Reflections

This project is the one where the technical choice (temperature, top_k, top_p) translates *most directly* into product feel. A summarization endpoint shipped at temperature=1.5 will produce wildly different completions for the same prompt across users — surprising and sometimes fun, but a nightmare for any workflow that needs reproducibility (regression tests, A/B baselines, support escalations). Conversely, a creative-writing assistant shipped at temperature=0 will feel sterile and repetitive. The numbers in the table aren't the answer — they're the *vocabulary* for having that product conversation.

The self-BLEU column is the underrated metric here. Distinct-n tells you how varied a *single* output is; self-BLEU tells you how varied your outputs are *across runs*. For a product that re-queries the same prompt (regeneration on user request, multi-sample voting, etc.), the second one is what users actually feel. Greedy and beam pin self-BLEU at 1.0 — every regeneration is identical. That's a feature for code completion, a bug for "give me another idea."

On the systems side, the strategies have very different cost profiles too. Beam search at width=4 is 4× the inference cost of greedy for marginal gain on most tasks. Sampling-based strategies are essentially free on top of greedy. When inference cost is the dominant operating expense (it usually is at scale), the right default is "sampling with a sensible temperature/top_p, beam only when you specifically need deterministic-better-than-greedy output." Making that default explicit in any LLM-serving stack saves real money.

## Methodology notes

- **Multi-prompt sensitivity.** A single prompt is enough to *demonstrate* a sampling effect but not enough to *quantify* it; the 5-prompt suite gives the per-strategy bars a meaningful average.
- **Sibling generations for self-BLEU.** Each sampling strategy is invoked `--n-samples-per-prompt` times (default 5) with different per-sample seeds so the self-BLEU calculation reflects real sampling diversity.
- **Greedy and beam are deterministic** so we generate them once (their self-BLEU is reported as 0.0, by convention).
- **distinct-2 and self-BLEU are inverse signals of diversity.** Healthy sampling configurations push distinct-2 *up* and self-BLEU *down*; too-aggressive sampling (high temperature, large top-p) can hurt fluency without much further diversity gain.
- **Same model checkpoint across cells.** Every comparison sees the same model so deltas are attributable to the decoding kwargs alone.

## Limitations

- Qwen2.5-1.5B is small enough for local experiments; the *metrics* generalise across small instruct-tuned LLMs but the *qualitative* outputs scale up significantly with model size.
- self-BLEU uses BLEU-1 (no smoothing / no brevity penalty) computed locally so the module stays dependency-light. The interpretation is the same as a full BLEU self-BLEU score; the absolute numbers will differ.
- A few sampling kwargs can interact in surprising ways (e.g. `temperature` is ignored when `do_sample=False`); we don't enumerate every interaction.

## Reproduce

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
