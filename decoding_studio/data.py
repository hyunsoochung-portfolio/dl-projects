"""Prompt source for the LLM sampling demo.

No external dataset is needed - the 'data' is a small fixed prompt suite that
spans factual / creative / explanatory styles so we can check that the
strategy effects generalize across prompts.
"""
from __future__ import annotations

DEFAULT_PROMPT = (
    "In a few sentences, describe how a transformer language model decides "
    "which token to generate next."
)

PROMPT_SUITE = [
    DEFAULT_PROMPT,
    "Write a short poem about a lighthouse in autumn.",
    "List three reasons why dropout regularizes neural networks.",
    "Explain what a kernel does in a convolutional layer.",
    "Tell a one-paragraph story about a robot learning to bake bread.",
]
