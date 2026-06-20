"""Configuration loading for OnaAI-2.0.

Resolution order (lowest -> highest precedence):
  1. Built-in defaults (this module).
  2. ``config/default.yaml`` shipped with the package.
  3. A user-supplied YAML file (``--config`` / ``load_config(path=...)``).
  4. Environment variables prefixed ``ONAAI_`` (e.g. ``ONAAI_MODEL_PATH``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency
    yaml = None

_ENV_PREFIX = "ONAAI_"

# Path to the YAML shipped with the package: <repo>/config/default.yaml
_PACKAGED_DEFAULT = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


@dataclass
class Config:
    """Runtime configuration for OnaAI-2.0."""

    model_path: str = "models/Qwen2.5-0.5B-Instruct"
    backend: str = "transformers"  # "transformers" | "vllm"
    dtype: str = "bfloat16"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = -1
    max_new_tokens: int = 40960

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.backend not in ("transformers", "vllm"):
            raise ValueError(
                f"backend must be 'transformers' or 'vllm', got {self.backend!r}"
            )
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature out of range [0, 2]: {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1]: {self.top_p}")
        # top_k == -1 disables top-k (vLLM/SGLang style); otherwise it must be
        # a positive integer. 0 and other negatives are invalid.
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1: {self.top_k}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive: {self.max_new_tokens}")

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _coerce(name: str, value: Any) -> Any:
    """Coerce a raw (string) value to the dataclass field's type."""
    field_types = {f.name: f.type for f in fields(Config)}
    target = field_types.get(name)
    if target in ("int", int):
        return int(value)
    if target in ("float", float):
        return float(value)
    return value


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to read configuration files")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping")
    return data


def load_config(path: Optional[str] = None) -> Config:
    """Build a :class:`Config`, layering packaged defaults, an optional file, and env."""
    valid = {f.name for f in fields(Config)}
    merged: Dict[str, Any] = {}

    # 2. packaged default yaml (best effort)
    if _PACKAGED_DEFAULT.is_file():
        merged.update({k: v for k, v in _read_yaml(_PACKAGED_DEFAULT).items() if k in valid})

    # 3. user-supplied yaml
    if path:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        merged.update({k: v for k, v in _read_yaml(p).items() if k in valid})

    # 4. environment overrides
    for key in valid:
        env_key = _ENV_PREFIX + key.upper()
        if env_key in os.environ:
            merged[key] = _coerce(key, os.environ[env_key])

    return Config(**merged)
