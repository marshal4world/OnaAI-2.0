"""Training pipeline for the OnaAI-2.0 model replica (SFT then RL)."""

from .data import (
    SFTExample,
    RLExample,
    load_sft_jsonl,
    load_rl_jsonl,
    sft_target_text,
    build_sft_messages,
    tokenize_sft_example,
)
from .reward import extract_answer, normalize_answer, verifiable_reward

__all__ = [
    "SFTExample",
    "RLExample",
    "load_sft_jsonl",
    "load_rl_jsonl",
    "sft_target_text",
    "build_sft_messages",
    "tokenize_sft_example",
    "extract_answer",
    "normalize_answer",
    "verifiable_reward",
]
