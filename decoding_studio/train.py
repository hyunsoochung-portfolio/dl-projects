"""Generate from a small open LLM with many sampling strategies and metrics.

There's nothing to "train" here; train.py runs the generation experiment and
writes:

- ``artifacts/samples.json``    full text for every (prompt, strategy, n) cell
- ``artifacts/metrics.csv``     distinct-1/2/3, mean token length, self-BLEU
- ``artifacts/strategy_summary.csv``  metrics averaged across prompts per strategy
- ``artifacts/distinct_bar.png``      bar chart of distinct-2 per strategy
- ``artifacts/selfbleu_bar.png``      bar chart of self-BLEU per strategy
- ``artifacts/samples.txt``    a human-readable dump of one prompt's outputs

Each sampling strategy is invoked with ``n_samples_per_prompt`` independent
generations when ``do_sample=True`` (so we can compute self-BLEU); greedy and
beam search are deterministic so we generate once for them.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import PROMPT_SUITE  # noqa: E402
from model import load_model  # noqa: E402
from shared.utils import (  # noqa: E402
    bar_chart,
    ensure_dir,
    save_metric_table,
    set_seed,
    time_block,
)


def distinct_n(tokens: list[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _bleu_1grams(a: list[str], b: list[str]) -> float:
    """Tiny BLEU-1 stand-in (no smoothing, no brevity penalty).

    Self-BLEU is conventionally a fuller BLEU score; computing it locally
    keeps the dependency footprint small. The interpretation is the same:
    higher means more overlap between sibling generations -> less diverse.
    """
    if not a or not b:
        return 0.0
    counts: dict[str, int] = {}
    for t in b:
        counts[t] = counts.get(t, 0) + 1
    matches = 0
    for t in a:
        if counts.get(t, 0) > 0:
            matches += 1
            counts[t] -= 1
    return matches / max(len(a), 1)


def self_bleu(token_lists: list[list[str]]) -> float:
    """Average pairwise BLEU-1 across sibling generations."""
    if len(token_lists) < 2:
        return 0.0
    pairs = list(combinations(range(len(token_lists)), 2))
    total = 0.0
    for i, j in pairs:
        total += _bleu_1grams(token_lists[i], token_lists[j])
    return total / max(len(pairs), 1)


def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             **gen_kwargs):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
    return text, tokens


def _strategies() -> dict[str, dict]:
    out: dict[str, dict] = {
        "greedy": dict(do_sample=False),
        "beam_4": dict(do_sample=False, num_beams=4),
    }
    for t in (0.5, 0.8, 1.0, 1.5):
        out[f"temperature_{t}"] = dict(do_sample=True, temperature=t)
    for k in (10, 50, 100):
        out[f"top_k_{k}"] = dict(do_sample=True, top_k=k)
    for p in (0.5, 0.8, 0.95):
        out[f"top_p_{p}"] = dict(do_sample=True, top_p=p)
    out["top_k50_top_p0.9"] = dict(do_sample=True, top_k=50, top_p=0.9)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="kept for CLI parity")
    parser.add_argument("--epochs", type=int, default=0,
                        help="kept for CLI parity")
    parser.add_argument("--prompt", default=None,
                        help="if set, replaces the entire prompt suite")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--n-samples-per-prompt", type=int, default=5,
                        help="independent samples per (prompt, strategy) "
                             "when do_sample=True; used for self-BLEU")
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    artifacts = ensure_dir(args.output_dir)
    model, tokenizer, model_name = load_model()

    prompts = [args.prompt] if args.prompt else list(PROMPT_SUITE)
    if args.quick:
        prompts = prompts[:1]
    strategies = _strategies()
    if args.quick:
        strategies = {k: strategies[k] for k in ("greedy", "temperature_0.8")}

    samples: dict = {"model": model_name, "prompts": prompts,
                     "strategies": {}}
    metric_rows: list[dict] = []
    strategy_agg: dict[str, list[dict]] = {s: [] for s in strategies}

    with time_block("LLM sampling"):
        for strat_name, kwargs in strategies.items():
            samples["strategies"][strat_name] = {"kwargs": kwargs,
                                                 "per_prompt": {}}
            for prompt_idx, prompt in enumerate(prompts):
                set_seed(args.seed + prompt_idx)
                if kwargs.get("do_sample"):
                    n_samples = args.n_samples_per_prompt
                else:
                    n_samples = 1
                gens = []
                token_lists = []
                for s in range(n_samples):
                    # vary the per-sample seed so a sampling strategy gets
                    # genuinely independent draws
                    set_seed(args.seed + prompt_idx * 100 + s)
                    text, tokens = generate(model, tokenizer, prompt,
                                            args.max_new_tokens, **kwargs)
                    gens.append(text)
                    token_lists.append(tokens)

                flat_tokens = [t for tl in token_lists for t in tl]
                d1 = distinct_n(flat_tokens, 1)
                d2 = distinct_n(flat_tokens, 2)
                d3 = distinct_n(flat_tokens, 3)
                mean_len = sum(len(tl) for tl in token_lists) / max(len(token_lists), 1)
                sb = self_bleu(token_lists) if n_samples > 1 else 0.0

                samples["strategies"][strat_name]["per_prompt"][str(prompt_idx)] = {
                    "prompt": prompt, "generations": gens,
                }
                cell = {
                    "strategy": strat_name, "prompt_idx": prompt_idx,
                    "n_samples": n_samples,
                    "distinct_1": d1, "distinct_2": d2, "distinct_3": d3,
                    "mean_token_length": mean_len,
                    "self_bleu": sb,
                }
                metric_rows.append(cell)
                strategy_agg[strat_name].append(cell)
                print(f"[{strat_name} | prompt {prompt_idx}] "
                      f"d1={d1:.3f} d2={d2:.3f} d3={d3:.3f} "
                      f"len={mean_len:.1f} self-bleu={sb:.3f}")

    save_metric_table(metric_rows, artifacts / "metrics.csv")

    # ---- per-strategy aggregates across prompts ----
    summary_rows = []
    for strat_name, cells in strategy_agg.items():
        if not cells:
            continue
        def avg(key):
            return sum(c[key] for c in cells) / len(cells)
        summary_rows.append({
            "strategy": strat_name,
            "n_prompts": len(cells),
            "distinct_1": avg("distinct_1"),
            "distinct_2": avg("distinct_2"),
            "distinct_3": avg("distinct_3"),
            "mean_token_length": avg("mean_token_length"),
            "self_bleu": avg("self_bleu"),
        })
    save_metric_table(summary_rows, artifacts / "strategy_summary.csv")

    bar_chart([r["strategy"] for r in summary_rows],
              [r["distinct_2"] for r in summary_rows],
              out_path=artifacts / "distinct_bar.png",
              title="distinct-2 per strategy (avg over prompts)",
              ylabel="distinct-2 (higher = more diverse)")
    bar_chart([r["strategy"] for r in summary_rows],
              [r["self_bleu"] for r in summary_rows],
              out_path=artifacts / "selfbleu_bar.png",
              title="self-BLEU per strategy (avg over prompts; sampling only)",
              ylabel="self-BLEU (lower = more diverse)")

    # ---- samples.json + a human-readable txt for the first prompt ----
    (artifacts / "samples.json").write_text(json.dumps(samples, indent=2))
    lines = [f"# LLM sampling demo", f"model: {model_name}",
             f"prompt: {prompts[0]}", ""]
    for strat_name in strategies:
        block = samples["strategies"][strat_name]["per_prompt"].get("0")
        if not block:
            continue
        lines.append(f"--- {strat_name} ---")
        for i, gen in enumerate(block["generations"]):
            lines.append(f"[#{i}] {gen.strip()}")
        lines.append("")
    (artifacts / "samples.txt").write_text("\n".join(lines))

    print(json.dumps({"model": model_name,
                      "summary_rows": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
