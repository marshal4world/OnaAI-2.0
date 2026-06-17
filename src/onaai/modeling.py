"""Architecture replica of VibeThinker-3B.

VibeThinker-3B is a **dense decoder-only Transformer** built on Qwen2.5-Coder-3B,
which uses the **Qwen2** architecture (RoPE, grouped-query attention, SwiGLU MLP,
RMSNorm). This module builds a *randomly-initialized* model with that exact
architecture at a chosen scale.

Two presets are provided:

* ``"vibethinker-3b"`` — the real ~3B dimensions (for training on real hardware).
  These mirror the public Qwen2.5-3B config; for an exact match, vendor the real
  ``config.json`` (``scripts/download_model.py``) and use
  :func:`config_from_pretrained`.
* ``"tiny"`` — a few-hundred-K-parameter model that trains on CPU in seconds,
  used by the end-to-end test.

Note: building from a preset produces **fresh random weights** (a replica of the
*architecture*, not the trained parameters). To start from the real pretrained
weights, point at the vendored model directory instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ArchPreset:
    """A Qwen2-architecture size preset."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


# Real VibeThinker-3B / Qwen2.5-3B dimensions (approximate public config).
# vocab_size is supplied at build time from the actual tokenizer.
VIBETHINKER_3B = ArchPreset(
    hidden_size=2048,
    intermediate_size=11008,
    num_hidden_layers=36,
    num_attention_heads=16,
    num_key_value_heads=2,
    max_position_embeddings=32768,
    rope_theta=1_000_000.0,
    tie_word_embeddings=True,
)

# Tiny preset: trains on CPU in seconds. Used for the end-to-end test.
TINY = ArchPreset(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    rope_theta=10_000.0,
    tie_word_embeddings=True,
)

PRESETS: Dict[str, ArchPreset] = {
    "vibethinker-3b": VIBETHINKER_3B,
    "tiny": TINY,
}


def build_config(preset: str = "tiny", vocab_size: int = 32000, **overrides):
    """Build a ``Qwen2Config`` from a named preset.

    ``vocab_size`` should match the tokenizer you intend to use.
    ``overrides`` override any individual config field.
    """
    from transformers import Qwen2Config

    if preset not in PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
    p = PRESETS[preset]

    cfg_kwargs: Dict[str, Any] = dict(
        vocab_size=vocab_size,
        hidden_size=p.hidden_size,
        intermediate_size=p.intermediate_size,
        num_hidden_layers=p.num_hidden_layers,
        num_attention_heads=p.num_attention_heads,
        num_key_value_heads=p.num_key_value_heads,
        max_position_embeddings=p.max_position_embeddings,
        rope_theta=p.rope_theta,
        tie_word_embeddings=p.tie_word_embeddings,
    )
    cfg_kwargs.update(p.extra)
    cfg_kwargs.update(overrides)
    return Qwen2Config(**cfg_kwargs)


def build_model(preset_or_config: Any = "tiny", vocab_size: int = 32000, **overrides):
    """Build a randomly-initialized ``Qwen2ForCausalLM``.

    ``preset_or_config`` may be a preset name (str) or a ready ``Qwen2Config``.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    if isinstance(preset_or_config, str):
        config = build_config(preset_or_config, vocab_size=vocab_size, **overrides)
    elif isinstance(preset_or_config, Qwen2Config):
        config = preset_or_config
    else:
        raise TypeError("preset_or_config must be a preset name or a Qwen2Config")

    model = Qwen2ForCausalLM(config)
    return model


def config_from_pretrained(path: str):
    """Load a ``Qwen2Config`` from a vendored model directory's ``config.json``.

    Use this for an *exact* architecture match against the real VibeThinker-3B
    after running ``scripts/download_model.py``.
    """
    from transformers import Qwen2Config

    cfg_path = Path(path)
    if cfg_path.is_dir():
        cfg_path = cfg_path / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"no config.json at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Qwen2Config(**data)


def count_parameters(model) -> int:
    """Total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())
