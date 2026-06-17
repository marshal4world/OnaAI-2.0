"""High-level reasoning engine for OnaAI-2.0.

The engine sits on top of :class:`~onaai.model.VibeThinkerModel` and is
responsible for turning a raw model completion into a structured result:
separating the chain-of-thought ("reasoning") from the final answer.

VibeThinker-3B (like most reasoning models) tends to emit a long thinking
trace and then a final answer. We support two common conventions:

  * ``<think> ... </think>`` blocks wrapping the reasoning.
  * A ``\\boxed{...}`` final answer (common in math benchmarks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .config import Config, load_config
from .model import VibeThinkerModel

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{")


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the last ``\\boxed{...}``, respecting nested braces."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()  # position just after the opening brace
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


@dataclass
class ReasoningResult:
    """Structured output of a reasoning call."""

    prompt: str
    raw: str
    answer: str
    reasoning: str

    def __str__(self) -> str:
        return self.answer


class ReasoningEngine:
    """Orchestrates prompting VibeThinker-3B and parsing its output."""

    def __init__(self, model: VibeThinkerModel, config: Optional[Config] = None) -> None:
        self.model = model
        self.config = config or model.config

    @classmethod
    def from_default(cls, config_path: Optional[str] = None) -> "ReasoningEngine":
        """Build an engine from layered configuration (see :func:`load_config`)."""
        config = load_config(config_path)
        return cls(VibeThinkerModel(config), config)

    def solve(self, prompt: str, **overrides) -> ReasoningResult:
        """Run the model on ``prompt`` and return a parsed :class:`ReasoningResult`."""
        raw = self.model.generate(prompt, **overrides)
        return self.parse(prompt, raw)

    @staticmethod
    def parse(prompt: str, raw: str) -> ReasoningResult:
        """Split a raw completion into reasoning + final answer."""
        reasoning = ""
        answer = raw.strip()

        think_match = _THINK_RE.search(raw)
        if think_match:
            reasoning = think_match.group(1).strip()
            # Everything after the closing </think> is the answer body.
            answer = raw[think_match.end():].strip() or answer

        boxed = _extract_boxed(raw)
        if boxed is not None:
            answer = boxed

        return ReasoningResult(prompt=prompt, raw=raw, answer=answer, reasoning=reasoning)
