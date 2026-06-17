#!/usr/bin/env python3
"""Build a tiny, randomly-initialized VibeThinker-style replica.

Creates a small byte-level BPE tokenizer (trained on the sample data) and a
tiny Qwen2-architecture model, and saves both to a local directory so the
SFT/RL scripts and the end-to-end test can load them like any HF model.

    python scripts/build_tiny_replica.py --out models/tiny-replica

This produces an *architecture* replica with fresh random weights -- it is
meant for pipeline testing, not for getting good answers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onaai.modeling import build_model, count_parameters  # noqa: E402
from onaai.training.data import load_sft_jsonl, load_rl_jsonl, sft_target_text  # noqa: E402
from onaai.training.tokenizer_utils import (  # noqa: E402
    train_tiny_tokenizer,
    default_corpus_from_records,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a tiny VibeThinker replica.")
    p.add_argument("--out", default="models/tiny-replica", help="output directory")
    p.add_argument("--sft-data", default="data/sample_sft.jsonl")
    p.add_argument("--rl-data", default="data/sample_rl.jsonl")
    p.add_argument("--vocab-size", type=int, default=2000)
    p.add_argument("--preset", default="tiny")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Corpus: include problems, reasoning, answers and the rendered SFT targets.
    sft = load_sft_jsonl(args.sft_data)
    rl = load_rl_jsonl(args.rl_data)
    records = [vars(e) for e in sft] + [vars(e) for e in rl]
    corpus = default_corpus_from_records(records)
    corpus += [sft_target_text(e) for e in sft]

    print(f"Training tiny tokenizer (vocab_size={args.vocab_size}) ...")
    tokenizer = train_tiny_tokenizer(corpus, vocab_size=args.vocab_size, save_dir=str(out))
    vocab_size = len(tokenizer)
    print(f"  tokenizer vocab size = {vocab_size}")

    print(f"Building tiny model (preset={args.preset}) ...")
    model = build_model(args.preset, vocab_size=vocab_size)
    model.save_pretrained(str(out))
    print(f"  parameters = {count_parameters(model):,}")

    print(f"\nReplica saved to: {out}")
    print("Next:")
    print(f"  python scripts/train_sft.py --model {out}")
    print(f"  python scripts/train_rl.py  --model {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
