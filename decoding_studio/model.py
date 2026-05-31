"""Load the open instruct-tuned LLM used for the sampling-strategy study."""
from __future__ import annotations

from typing import Tuple

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def load_model() -> Tuple[object, object, str]:
    """Return (model, tokenizer, name) for the configured LLM.

    Requires `HF_TOKEN` set in the environment if the target model is
    gated on the Hugging Face Hub.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[model] loaded {MODEL_NAME}")
    return model, tokenizer, MODEL_NAME
