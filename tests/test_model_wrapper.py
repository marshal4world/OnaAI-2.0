"""Tests for the VibeThinkerModel inference wrapper (transformers backend).

Builds a tiny model + tokenizer, saves them, and exercises the wrapper's
generate path -- including the greedy (temperature=0) branch that must use
do_sample=False rather than passing temperature=0 to transformers.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from onaai.config import Config  # noqa: E402
from onaai.model import VibeThinkerModel  # noqa: E402
from onaai.modeling import build_model  # noqa: E402
from onaai.training.tokenizer_utils import train_tiny_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def saved_replica(tmp_path_factory):
    out = tmp_path_factory.mktemp("wrapper-model")
    tok = train_tiny_tokenizer(
        ["What is 2+2?", "4", "<think>", "</think>", "\\boxed{", "}"], vocab_size=300
    )
    tok.save_pretrained(str(out))
    model = build_model("tiny", vocab_size=len(tok))
    model.save_pretrained(str(out))
    return str(out)


def test_greedy_generation(saved_replica):
    # temperature=0 -> greedy. Must not raise (the bug passed temperature=0 to
    # transformers, which rejects it unless do_sample=False).
    cfg = Config(model_path=saved_replica, temperature=0.0, max_new_tokens=8)
    model = VibeThinkerModel(cfg)
    out = model.generate("What is 2+2?")
    assert isinstance(out, str)


def test_sampling_generation(saved_replica):
    cfg = Config(model_path=saved_replica, temperature=1.0, top_p=0.95, max_new_tokens=8)
    model = VibeThinkerModel(cfg)
    out = model.generate("What is 2+2?")
    assert isinstance(out, str)


def test_per_call_override(saved_replica):
    cfg = Config(model_path=saved_replica, temperature=1.0, max_new_tokens=8)
    model = VibeThinkerModel(cfg)
    # Override to greedy for a single call.
    out = model.generate("What is 2+2?", temperature=0.0)
    assert isinstance(out, str)
