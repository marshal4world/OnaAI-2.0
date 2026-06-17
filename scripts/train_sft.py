#!/usr/bin/env python3
"""Run the SFT (Spectrum) phase on a local replica.

    python scripts/train_sft.py --model models/tiny-replica \
        --sft-data data/sample_sft.jsonl --out checkpoints/sft
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onaai.training.data import load_sft_jsonl  # noqa: E402
from onaai.training.sft import SFTConfig, train_sft  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT a VibeThinker replica.")
    p.add_argument("--model", required=True, help="local model dir (tokenizer + weights)")
    p.add_argument("--sft-data", default="data/sample_sft.jsonl")
    p.add_argument("--out", default="checkpoints/sft")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--max-length", type=int, default=256)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    examples = load_sft_jsonl(args.sft_data)
    cfg = SFTConfig(
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
    )
    trainer = train_sft(model, tokenizer, examples, cfg)

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"SFT complete. Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
