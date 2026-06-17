"""Verifiable reward for the RL phase.

The core of VibeThinker's "Signal" stage is a *verifiable* reward: a model
completion is rewarded only if its extracted final answer matches the
ground-truth answer. This module is pure-Python (no torch) and reuses the same
answer-extraction logic as the inference engine, so training and serving agree
on what "the answer" is.
"""

from __future__ import annotations

import re
from typing import Optional

from ..engine import _extract_boxed  # nested-brace-aware \boxed{...} extractor

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NON_ALNUM_EDGE = re.compile(r"^[\s$]+|[\s$.]+$")


def extract_answer(text: str) -> str:
    """Extract a candidate final answer from a model completion.

    Preference order:
      1. The content of the last ``\\boxed{...}``.
      2. Otherwise, the text after the last ``</think>`` (stripped).
      3. Otherwise, the whole stripped text.
    """
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()

    # Drop the thinking block(s) if present, keep what follows.
    after = _THINK_BLOCK.sub("", text).strip()
    return after or text.strip()


def normalize_answer(ans: str) -> str:
    """Normalize an answer for robust string comparison.

    Lowercase, strip surrounding whitespace/``$``/trailing period, collapse
    internal whitespace, and remove thousands separators in plain numbers.
    """
    s = ans.strip()
    s = _NON_ALNUM_EDGE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    # Normalize simple numbers like "1,000" -> "1000".
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
        # drop a trailing ".0" so "5" == "5.0"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def answers_match(prediction: str, ground_truth: str) -> bool:
    """True if the extracted prediction matches the ground-truth answer."""
    return normalize_answer(extract_answer(prediction)) == normalize_answer(ground_truth)


def verifiable_reward(
    completion: str,
    ground_truth: str,
    correct: float = 1.0,
    incorrect: float = 0.0,
) -> float:
    """Binary verifiable reward: ``correct`` if answers match, else ``incorrect``."""
    return correct if answers_match(completion, ground_truth) else incorrect
