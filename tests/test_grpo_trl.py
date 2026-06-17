"""Tests for the TRL-based GRPO drop-in.

The actual ``GRPOTrainer.train()`` needs a TRL-capable environment, so it is not
exercised here. Everything OnaAI owns -- the reward adapter, completion
normalization, dataset construction, and the import guard -- is tested.
"""

import pytest

from onaai.training.grpo_trl import (
    TRLUnavailableError,
    _completion_to_text,
    _import_trl,
    grpo_train_trl,
    make_verifiable_reward_func,
)
from onaai.training.data import RLExample


# ------------------------------ reward adapter ----------------------------- #

def test_completion_to_text_variants():
    assert _completion_to_text("\\boxed{9}") == "\\boxed{9}"
    assert _completion_to_text(
        [{"role": "assistant", "content": "\\boxed{9}"}]
    ) == "\\boxed{9}"


def test_reward_func_scores_against_answer_column():
    fn = make_verifiable_reward_func()
    rewards = fn(
        prompts=["p1", "p2"],
        completions=["<think>..</think>\\boxed{15}", "\\boxed{99}"],
        answer=["15", "16"],
    )
    assert rewards == [1.0, 0.0]


def test_reward_func_handles_conversational_completions():
    fn = make_verifiable_reward_func()
    rewards = fn(
        completions=[[{"role": "assistant", "content": "\\boxed{42}"}]],
        answer=["42"],
    )
    assert rewards == [1.0]


def test_reward_func_requires_answer_column():
    fn = make_verifiable_reward_func()
    with pytest.raises(KeyError):
        fn(completions=["\\boxed{1}"])  # no 'answer' kwarg


# ------------------------------ dataset build ------------------------------ #

def test_build_trl_dataset():
    pytest.importorskip("datasets")
    pytest.importorskip("tokenizers")
    from onaai.training.grpo_trl import build_trl_dataset
    from onaai.training.tokenizer_utils import train_tiny_tokenizer

    tok = train_tiny_tokenizer(["What is 2+2?", "4", "<think>", "</think>"], vocab_size=300)
    examples = [RLExample("What is 2+2?", "4"), RLExample("What is 3+3?", "6")]
    ds = build_trl_dataset(examples, tok)

    assert set(ds.column_names) == {"prompt", "answer"}
    assert len(ds) == 2
    assert ds[0]["answer"] == "4"
    assert "<|im_start|>" in ds[0]["prompt"]


# ------------------------------ import guard ------------------------------- #

def test_import_guard_or_run():
    """Either TRL's GRPO imports (capable env) or the guard raises cleanly."""
    try:
        GRPOTrainer, _ = _import_trl()
    except TRLUnavailableError as e:
        # Expected in minimal/CPU environments; message must be actionable.
        assert "TRL's GRPOTrainer is unavailable" in str(e)
        # And the public entrypoint surfaces the same typed error.
        with pytest.raises(TRLUnavailableError):
            grpo_train_trl(model=None, tokenizer=None, examples=[RLExample("p", "a")])
    else:
        assert GRPOTrainer is not None  # TRL is fully available here
