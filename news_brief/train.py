"""Summarization inference + ROUGE; optional short fine-tune via --finetune.

Experiments produced under ``artifacts/``:

- ``model_comparison.csv / .png``       distilbart vs bart-base (vs t5-small if loadable)
- ``decoding_comparison.csv / .png``    greedy / beam(2|4|8) / sample(temp=0.7,top_p=0.9)
- ``length_penalty_sweep.csv / .png``   length_penalty in {0.5, 1.0, 2.0}
- ``examples.json``                     a handful of (doc, ref, pred) triples
- ``rouge.json``                        ROUGE-1/2/L for the default-config eval
- ``summary.json``                      structured summary of every experiment

ROUGE uses ``evaluate.load("rouge")`` and reports ROUGE-1 / ROUGE-2 / ROUGE-L.

Pass ``--finetune`` to also run a tiny Seq2SeqTrainer fine-tune.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from data import load_summarization_split  # noqa: E402
from model import COMPARE_MODELS, DEFAULT_MODEL, build_pipeline, load_model_and_tokenizer  # noqa: E402
from shared.utils import (  # noqa: E402
    bar_chart,
    ensure_dir,
    save_metric_table,
    set_seed,
    time_block,
)


def compute_rouge(predictions, references):
    import evaluate

    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def _run_pipe(pipe, rows, gen_kwargs, max_eval):
    preds, refs = [], []
    for row in rows[:max_eval]:
        out = pipe(row["document"], max_length=80, min_length=15,
                   truncation=True, **gen_kwargs)
        preds.append(out[0]["summary_text"].strip())
        refs.append(row["summary"])
    return preds, refs


def run_inference(model_name: str, eval_rows, artifacts: Path,
                  max_eval: int = 50) -> dict:
    pipe = build_pipeline(model_name)
    sample = eval_rows[:max_eval]

    preds, refs = _run_pipe(pipe, sample, dict(do_sample=False), max_eval)

    examples = []
    for row, pred in zip(sample, preds):
        if len(examples) >= 5:
            break
        examples.append({
            "document": row["document"][:500] + ("..." if len(row["document"]) > 500 else ""),
            "reference": row["summary"],
            "prediction": pred,
        })

    rouge = compute_rouge(preds, refs)
    print("ROUGE:", rouge)

    (artifacts / "examples.json").write_text(json.dumps(examples, indent=2))
    (artifacts / "rouge.json").write_text(json.dumps(rouge, indent=2))
    return rouge


def run_model_comparison(eval_rows, artifacts: Path, max_eval: int,
                         model_names) -> list[dict]:
    rows = []
    with time_block("model comparison"):
        for name in model_names:
            try:
                pipe = build_pipeline(name)
                preds, refs = _run_pipe(pipe, eval_rows, dict(do_sample=False),
                                        max_eval)
                rouge = compute_rouge(preds, refs)
                row = {"model": name,
                       "rouge1": float(rouge.get("rouge1", 0.0)),
                       "rouge2": float(rouge.get("rouge2", 0.0)),
                       "rougeL": float(rouge.get("rougeL", 0.0))}
                rows.append(row)
                print(f"[model={name}] {row}")
            except Exception as e:
                print(f"[model={name}] skipped: {e}")
                rows.append({"model": name, "error": str(e)})
    save_metric_table(rows, artifacts / "model_comparison.csv")
    ok_rows = [r for r in rows if "rouge1" in r]
    if ok_rows:
        bar_chart([r["model"] for r in ok_rows],
                  [r["rougeL"] for r in ok_rows],
                  out_path=artifacts / "model_comparison.png",
                  title="Model comparison (ROUGE-L, greedy)",
                  ylabel="ROUGE-L")
    return rows


def run_decoding_comparison(model_name: str, eval_rows, artifacts: Path,
                            max_eval: int) -> list[dict]:
    pipe = build_pipeline(model_name)
    strategies = {
        "greedy": dict(do_sample=False),
        "beam_2": dict(do_sample=False, num_beams=2),
        "beam_4": dict(do_sample=False, num_beams=4),
        "beam_8": dict(do_sample=False, num_beams=8),
        "sample_t0.7_p0.9": dict(do_sample=True, temperature=0.7, top_p=0.9),
    }
    rows = []
    with time_block("decoding comparison"):
        for name, kwargs in strategies.items():
            preds, refs = _run_pipe(pipe, eval_rows, kwargs, max_eval)
            rouge = compute_rouge(preds, refs)
            row = {"strategy": name,
                   "rouge1": float(rouge.get("rouge1", 0.0)),
                   "rouge2": float(rouge.get("rouge2", 0.0)),
                   "rougeL": float(rouge.get("rougeL", 0.0))}
            rows.append(row)
            print(f"[decode={name}] {row}")
    save_metric_table(rows, artifacts / "decoding_comparison.csv")
    bar_chart([r["strategy"] for r in rows], [r["rougeL"] for r in rows],
              out_path=artifacts / "decoding_comparison.png",
              title="Decoding strategy comparison (ROUGE-L)",
              ylabel="ROUGE-L")
    return rows


def run_length_penalty(model_name: str, eval_rows, artifacts: Path,
                       max_eval: int) -> list[dict]:
    pipe = build_pipeline(model_name)
    rows = []
    with time_block("length penalty sweep"):
        for lp in (0.5, 1.0, 2.0):
            preds, refs = _run_pipe(
                pipe, eval_rows,
                dict(do_sample=False, num_beams=4, length_penalty=lp),
                max_eval,
            )
            rouge = compute_rouge(preds, refs)
            row = {"length_penalty": lp,
                   "rouge1": float(rouge.get("rouge1", 0.0)),
                   "rouge2": float(rouge.get("rouge2", 0.0)),
                   "rougeL": float(rouge.get("rougeL", 0.0))}
            rows.append(row)
            print(f"[length_penalty={lp}] {row}")
    save_metric_table(rows, artifacts / "length_penalty_sweep.csv")
    bar_chart([f"lp={r['length_penalty']}" for r in rows],
              [r["rougeL"] for r in rows],
              out_path=artifacts / "length_penalty_sweep.png",
              title="length_penalty sweep (beam=4)", ylabel="ROUGE-L")
    return rows


def run_finetune(model_name: str, train_rows, eval_rows, artifacts: Path,
                 epochs: int = 1, max_train: int = 200) -> None:
    from datasets import Dataset
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    model, tokenizer = load_model_and_tokenizer(model_name)

    train_rows = train_rows[:max_train]
    eval_rows = eval_rows[:min(len(eval_rows), 50)]

    def to_ds(rows):
        return Dataset.from_list(rows)

    def tokenize(batch):
        model_inputs = tokenizer(batch["document"], max_length=512,
                                 truncation=True)
        labels = tokenizer(text_target=batch["summary"], max_length=80,
                           truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = to_ds(train_rows).map(tokenize, batched=True,
                                      remove_columns=["document", "summary"])
    eval_tok = to_ds(eval_rows).map(tokenize, batched=True,
                                    remove_columns=["document", "summary"])

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(artifacts / "finetune"),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=3e-5,
        predict_with_generate=True,
        logging_steps=20,
        report_to=[],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(artifacts / "finetune" / "final"))
    print("fine-tune complete; model saved under", artifacts / "finetune" / "final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="kept for CLI parity; decoding is mostly "
                             "deterministic except the sampling row")
    parser.add_argument("--epochs", type=int, default=1)  # only matters for --finetune
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-eval", type=int, default=50)
    parser.add_argument("--output-dir", default=str(HERE / "artifacts"))
    parser.add_argument("--quick", action="store_true",
                        help="shrink eval to 5 rows for smoke testing")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--finetune-max-train", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)
    artifacts = ensure_dir(args.output_dir)

    train_rows, eval_rows, ds_name = load_summarization_split()
    print(f"loaded {ds_name}: {len(train_rows)} train, {len(eval_rows)} eval")

    max_eval = 5 if args.quick else args.max_eval

    # 1) headline default inference
    rouge = run_inference(args.model, eval_rows, artifacts, max_eval=max_eval)

    # 2) model comparison
    compare = ["distilbart-cnn-12-6"] if args.quick else COMPARE_MODELS
    model_rows = run_model_comparison(eval_rows, artifacts, max_eval, compare)

    # 3) decoding strategy comparison
    decode_rows = run_decoding_comparison(args.model, eval_rows, artifacts,
                                          max_eval)

    # 4) length_penalty sweep
    lp_rows = run_length_penalty(args.model, eval_rows, artifacts, max_eval)

    if args.finetune:
        run_finetune(args.model, train_rows, eval_rows, artifacts,
                     epochs=args.finetune_epochs,
                     max_train=args.finetune_max_train)

    summary = {
        "dataset": ds_name,
        "model": args.model,
        "default_rouge": rouge,
        "model_comparison": model_rows,
        "decoding_comparison": decode_rows,
        "length_penalty_sweep": lp_rows,
        "finetuned": bool(args.finetune),
        "max_eval": max_eval,
    }
    (artifacts / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
