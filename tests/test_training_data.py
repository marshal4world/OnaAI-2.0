import pytest

from onaai.training.data import (
    RLExample,
    SFTExample,
    build_sft_messages,
    load_rl_jsonl,
    load_sft_jsonl,
    sft_target_text,
)

SFT_PATH = "data/sample_sft.jsonl"
RL_PATH = "data/sample_rl.jsonl"


def test_load_sample_sft():
    examples = load_sft_jsonl(SFT_PATH)
    assert len(examples) >= 8
    assert all(isinstance(e, SFTExample) for e in examples)
    assert examples[0].problem and examples[0].reasoning and examples[0].answer


def test_load_sample_rl():
    examples = load_rl_jsonl(RL_PATH)
    assert len(examples) >= 4
    assert all(isinstance(e, RLExample) for e in examples)


def test_sft_target_contains_think_and_boxed():
    ex = SFTExample(problem="What is 2+2?", reasoning="2 plus 2 is 4.", answer="4")
    target = sft_target_text(ex)
    assert "<think>2 plus 2 is 4.</think>" in target
    assert "\\boxed{4}" in target


def test_build_sft_messages_roles():
    ex = SFTExample(problem="p", reasoning="r", answer="a")
    msgs = build_sft_messages(ex)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "p"


def test_sft_missing_field_raises():
    with pytest.raises(ValueError):
        SFTExample.from_dict({"problem": "p", "answer": "a"})  # no reasoning


def test_rl_missing_field_raises():
    with pytest.raises(ValueError):
        RLExample.from_dict({"problem": "p"})  # no answer
