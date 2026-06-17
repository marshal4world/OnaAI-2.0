"""Test the periodic held-out evaluation hook inside GRPO RL training."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from onaai.modeling import build_model  # noqa: E402
from onaai.training.data import load_rl_jsonl  # noqa: E402
from onaai.training.rl import GRPOConfig, grpo_train  # noqa: E402
from onaai.training.tokenizer_utils import train_tiny_tokenizer  # noqa: E402

RL_PATH = "data/sample_rl.jsonl"
EVAL_PATH = "data/sample_eval.jsonl"


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    rl = load_rl_jsonl(RL_PATH)
    ev = load_rl_jsonl(EVAL_PATH)
    corpus = [e.problem for e in rl + ev] + [e.answer for e in rl + ev]
    corpus += ["<think>", "</think>", "\\boxed{", "}"]
    tokenizer = train_tiny_tokenizer(corpus, vocab_size=600)
    model = build_model("tiny", vocab_size=len(tokenizer))
    return model, tokenizer, rl, ev


def test_rl_records_periodic_eval(tiny):
    model, tokenizer, rl, ev = tiny
    cfg = GRPOConfig(
        group_size=4,
        epochs=1,
        learning_rate=1e-4,
        max_new_tokens=12,
        max_prompt_length=64,
        eval_every=2,        # eval every 2 optimizer steps
        eval_k=1,
        eval_n_samples=2,
        seed=0,
    )
    history = grpo_train(model, tokenizer, rl[:4], config=cfg, eval_examples=ev[:3])

    assert "eval" in history
    assert len(history["eval"]) >= 1
    steps = [rec["step"] for rec in history["eval"]]
    assert steps == sorted(steps)  # recorded in increasing step order
    for rec in history["eval"]:
        assert set(rec) >= {"step", "pass_at_k", "accuracy", "k", "n_samples"}
        assert 0.0 <= rec["pass_at_k"] <= 1.0
        assert 0.0 <= rec["accuracy"] <= 1.0
        assert math.isfinite(rec["step"])


def test_rl_eval_disabled_by_default(tiny):
    model, tokenizer, rl, _ = tiny
    cfg = GRPOConfig(group_size=4, epochs=1, max_new_tokens=8, max_prompt_length=64)
    history = grpo_train(model, tokenizer, rl[:2], config=cfg)  # no eval_examples
    assert history["eval"] == []
