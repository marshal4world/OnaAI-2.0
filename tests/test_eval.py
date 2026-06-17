import math

import pytest

from onaai.eval import estimate_pass_at_k


# ----------------------------- estimator unit tests ----------------------- #

def test_pass_at_1_all_correct():
    assert estimate_pass_at_k(1, 1, 1) == 1.0


def test_pass_at_1_none_correct():
    assert estimate_pass_at_k(4, 0, 1) == 0.0


def test_pass_at_1_is_fraction_correct():
    # With k=1, pass@1 reduces to c/n.
    assert estimate_pass_at_k(5, 2, 1) == pytest.approx(0.4)


def test_pass_at_k_known_value():
    # n=4, c=1, k=2 -> 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
    assert estimate_pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_pass_at_k_fewer_incorrect_than_k():
    # n=3, c=2, k=2 -> only 1 incorrect, every pair contains a correct -> 1.0
    assert estimate_pass_at_k(3, 2, 2) == 1.0


def test_pass_at_k_monotonic_in_correct():
    vals = [estimate_pass_at_k(8, c, 3) for c in range(0, 9)]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


@pytest.mark.parametrize("n,c,k", [(1, 0, 2), (0, 0, 1), (3, 4, 1), (3, -1, 1)])
def test_invalid_args_raise(n, c, k):
    with pytest.raises(ValueError):
        estimate_pass_at_k(n, c, k)


# ----------------------------- end-to-end eval ----------------------------- #

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from onaai.eval import EvalConfig, evaluate  # noqa: E402
from onaai.modeling import build_model  # noqa: E402
from onaai.training.data import load_rl_jsonl  # noqa: E402
from onaai.training.tokenizer_utils import train_tiny_tokenizer  # noqa: E402

EVAL_PATH = "data/sample_eval.jsonl"


def test_evaluate_end_to_end():
    torch.manual_seed(0)
    examples = load_rl_jsonl(EVAL_PATH)
    corpus = [e.problem for e in examples] + [e.answer for e in examples]
    corpus += ["<think>", "</think>", "\\boxed{", "}"]
    tokenizer = train_tiny_tokenizer(corpus, vocab_size=500)
    model = build_model("tiny", vocab_size=len(tokenizer))

    cfg = EvalConfig(k=1, n_samples=3, max_new_tokens=12, max_prompt_length=64)
    result = evaluate(model, tokenizer, examples[:3], cfg)

    assert result.num_examples == 3
    assert result.n_samples == 3  # max(n_samples, k)
    assert 0.0 <= result.pass_at_k <= 1.0
    assert 0.0 <= result.accuracy <= 1.0
    assert len(result.per_example) == 3
    for e in result.per_example:
        assert len(e.completions) == 3
        assert 0 <= e.num_correct <= 3
    # summary string is well-formed
    assert "pass@1" in result.summary()


def test_evaluate_with_perfect_judge():
    # A judge that always accepts -> pass@k and accuracy must be 1.0.
    torch.manual_seed(0)
    examples = load_rl_jsonl(EVAL_PATH)
    tokenizer = train_tiny_tokenizer([e.problem for e in examples], vocab_size=300)
    model = build_model("tiny", vocab_size=len(tokenizer))

    cfg = EvalConfig(k=2, n_samples=2, max_new_tokens=8, max_prompt_length=64)
    result = evaluate(
        model, tokenizer, examples[:2], cfg, judge=lambda c, gt: True
    )
    assert result.pass_at_k == 1.0
    assert result.accuracy == 1.0
