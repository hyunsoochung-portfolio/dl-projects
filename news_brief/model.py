"""Load pretrained seq2seq Transformers for summarization."""
from __future__ import annotations

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"

# Models cycled through in the multi-model comparison experiment. Each entry
# is tried in order; failing entries are reported and skipped at runtime.
COMPARE_MODELS = (
    "sshleifer/distilbart-cnn-12-6",
    "facebook/bart-base",
    "t5-small",
)


def load_model_and_tokenizer(model_name: str = DEFAULT_MODEL):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer


def build_pipeline(model_name: str = DEFAULT_MODEL):
    from transformers import pipeline

    return pipeline("summarization", model=model_name, tokenizer=model_name)
