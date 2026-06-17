#!/usr/bin/env python3
"""Evaluate a replica on a held-out verifiable set (pass@k + accuracy).

    python scripts/evaluate.py --model checkpoints/rl \
        --eval-data data/sample_eval.jsonl --k 1 --n-samples 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from onaai.eval import EvalConfig, evaluate  # noqa: E402
from onaai.training.data import load_rl_jsonl  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pass@k / accuracy evaluation.")
    p.add_argument("--model", required=True, help="local model dir (tokenizer + weights)")
    p.add_argument("--eval-data", default="data/sample_eval.jsonl")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=1, help="completions per problem (>= k)")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--json", action="store_true", help="print full per-example JSON")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    examples = load_rl_jsonl(args.eval_data)
    cfg = EvalConfig(
        k=args.k,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    result = evaluate(model, tokenizer, examples, cfg)

    print(result.summary())
    if args.json:
        payload = {
            "pass_at_k": result.pass_at_k,
            "accuracy": result.accuracy,
            "k": result.k,
            "n_samples": result.n_samples,
            "per_example": [
                {
                    "problem": e.problem,
                    "ground_truth": e.ground_truth,
                    "num_correct": e.num_correct,
                    "num_samples": e.num_samples,
                    "pass_at_k": e.pass_at_k,
                }
                for e in result.per_example
            ],
        }
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
