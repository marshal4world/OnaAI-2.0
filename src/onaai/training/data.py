"""Dataset loading + formatting for the VibeThinker-3B replica.

Pure-Python record handling (loading, validation, message/target construction)
is kept separate from tokenization so it can be unit-tested without torch or a
tokenizer. Tokenization (``tokenize_sft_example``) needs a tokenizer but no GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Structural markup the model is trained to emit (matches onaai.engine parsing).
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class SFTExample:
    problem: str
    reasoning: str
    answer: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SFTExample":
        missing = [k for k in ("problem", "reasoning", "answer") if k not in d or d[k] is None]
        if missing:
            raise ValueError(f"SFT record missing required field(s): {missing}: {d!r}")
        return cls(problem=str(d["problem"]), reasoning=str(d["reasoning"]), answer=str(d["answer"]))


@dataclass
class RLExample:
    problem: str
    answer: str

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLExample":
        missing = [k for k in ("problem", "answer") if k not in d or d[k] is None]
        if missing:
            raise ValueError(f"RL record missing required field(s): {missing}: {d!r}")
        return cls(problem=str(d["problem"]), answer=str(d["answer"]))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return rows


def load_sft_jsonl(path: str) -> List[SFTExample]:
    return [SFTExample.from_dict(r) for r in _read_jsonl(path)]


def load_rl_jsonl(path: str) -> List[RLExample]:
    return [RLExample.from_dict(r) for r in _read_jsonl(path)]


def sft_target_text(example: SFTExample) -> str:
    """The assistant completion the model is trained to produce."""
    return f"{THINK_OPEN}{example.reasoning}{THINK_CLOSE}\n\\boxed{{{example.answer}}}"


def build_sft_messages(example: SFTExample) -> List[Dict[str, str]]:
    """Full chat (user + assistant) used to build the SFT training sequence."""
    return [
        {"role": "user", "content": example.problem},
        {"role": "assistant", "content": sft_target_text(example)},
    ]


def build_prompt_messages(problem: str) -> List[Dict[str, str]]:
    """User-only chat used for RL rollouts and inference."""
    return [{"role": "user", "content": problem}]


def tokenize_sft_example(
    example: SFTExample,
    tokenizer,
    max_length: int = 1024,
) -> Dict[str, List[int]]:
    """Tokenize one SFT example with **prompt masking**.

    Returns ``input_ids``, ``attention_mask`` and ``labels`` where the prompt
    tokens have label ``-100`` (ignored by the loss) so the model only learns
    to produce the completion -- standard instruction-tuning practice.
    """
    # Prompt portion (user turn + assistant generation prefix).
    prompt_text = tokenizer.apply_chat_template(
        build_prompt_messages(example.problem),
        tokenize=False,
        add_generation_prompt=True,
    )
    # Full sequence (prompt + assistant completion).
    full_text = tokenizer.apply_chat_template(
        build_sft_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    full_ids = full_ids[:max_length]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100  # mask the prompt

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class SFTDataset:
    """A minimal map-style dataset of tokenized SFT examples (torch-free build)."""

    def __init__(self, examples: List[SFTExample], tokenizer, max_length: int = 1024):
        self.features = [tokenize_sft_example(ex, tokenizer, max_length) for ex in examples]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.features[idx]
