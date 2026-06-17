#!/usr/bin/env python3
"""Vendor the base reasoning model (tokenizer + weights) into a local directory.

This is part C of OnaAI-2.0: pull the tokenizer files (tokenizer.json,
tokenizer_config.json, special_tokens_map.json, vocab/merges, ...) and the
model weights (*.safetensors) from the Hugging Face Hub into ``models/`` so
everything lives on local disk.

Examples
--------
    # Download the default small, CPU-friendly model + tokenizer into
    # ./models/Qwen2.5-0.5B-Instruct
    python scripts/download_model.py

    # Tokenizer + config only (fast; no multi-GB weights) -- useful for
    # building the data pipeline before committing to a full download.
    python scripts/download_model.py --tokenizer-only

    # Custom repo / destination (e.g. a larger model on GPU hardware)
    python scripts/download_model.py --repo <org>/<model> --dest /data/models

Notes
-----
* Requires ``huggingface_hub`` (``pip install -e ".[train]"`` or
  ``pip install huggingface_hub``).
* Large weights are NOT committed to git (see .gitignore). Keep them on disk
  or in a shared cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Tokenizer + config files (small). Weight shards are added separately.
TOKENIZER_PATTERNS = [
    "tokenizer*.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "*.model",  # sentencepiece, if present
    "config.json",
    "generation_config.json",
    "chat_template*",
]

WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vendor the base reasoning model locally.")
    p.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF repo id")
    p.add_argument("--dest", default="models", help="destination root directory")
    p.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="download only tokenizer + config files (skip weights)",
    )
    p.add_argument("--revision", default=None, help="git revision / tag / commit")
    p.add_argument(
        "--token", default=None, help="HF access token (or set HF_TOKEN env var)"
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is required.\n"
            '  pip install huggingface_hub   (or: pip install -e ".[train]")',
            file=sys.stderr,
        )
        return 1

    patterns = list(TOKENIZER_PATTERNS)
    if not args.tokenizer_only:
        patterns += WEIGHT_PATTERNS

    target = Path(args.dest) / args.repo.split("/")[-1]
    target.mkdir(parents=True, exist_ok=True)

    kind = "tokenizer + config" if args.tokenizer_only else "tokenizer + weights"
    print(f"Downloading {kind} for {args.repo} -> {target}")

    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=str(target),
        allow_patterns=patterns,
        token=args.token,
    )

    print(f"\nDone. Files vendored at: {path}")
    print("Point OnaAI-2.0 at the local copy with:")
    print(f"  export ONAAI_MODEL_PATH={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
