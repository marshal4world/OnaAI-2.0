#!/usr/bin/env python3
"""Run the GRPO-style RL (Signal) phase on a local replica.

    python scripts/train_rl.py --model checkpoints/sft \
        --rl-data data/sample_rl.jsonl --out checkpoints/rl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onaai.training.data import load_rl_jsonl  # noqa: E402
from onaai.training.rl import GRPOConfig, grpo_train  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GRPO-style RL for an OnaAI-2.0 replica.")
    p.add_argument("--model", required=True, help="local model dir (tokenizer + weights)")
    p.add_argument("--rl-data", default="data/sample_rl.jsonl")
    p.add_argument("--out", default="checkpoints/rl")
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--eval-data", default=None, help="optional held-out set for periodic eval")
    p.add_argument("--eval-every", type=int, default=0, help="eval every N steps (0=off)")
    p.add_argument("--eval-k", type=int, default=1)
    p.add_argument("--eval-n-samples", type=int, default=4)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    examples = load_rl_jsonl(args.rl_data)
    eval_examples = load_rl_jsonl(args.eval_data) if args.eval_data else None
    cfg = GRPOConfig(
        group_size=args.group_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_new_tokens=args.max_new_tokens,
        eval_every=args.eval_every,
        eval_k=args.eval_k,
        eval_n_samples=args.eval_n_samples,
    )
    history = grpo_train(model, tokenizer, examples, config=cfg, eval_examples=eval_examples)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    msg = f"RL complete. mean_reward={history['mean_reward']:.3f}."
    if history.get("eval"):
        last = history["eval"][-1]
        msg += f" final pass@{last['k']}={last['pass_at_k']:.3f} acc={last['accuracy']:.3f}."
    print(msg + f" Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
