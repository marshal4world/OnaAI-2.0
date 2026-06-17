"""End-to-end test for the OnaAI-2.0 model replica.

Exercises the entire pipeline on CPU with a tiny tokenizer + tiny model:

    sample data -> train BPE tokenizer -> build Qwen2-arch model
                -> save/reload -> SFT -> GRPO RL -> generate -> parse + reward

This validates that every component wires together. It does NOT assert the tiny
random model becomes *accurate* (it can't, at this scale) -- only that training
runs, losses are finite, rewards are well-formed, and generation/parse work.
"""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from onaai.engine import ReasoningEngine  # noqa: E402
from onaai.modeling import build_model, count_parameters  # noqa: E402
from onaai.training.data import (  # noqa: E402
    load_rl_jsonl,
    load_sft_jsonl,
    sft_target_text,
    tokenize_sft_example,
)
from onaai.training.rl import GRPOConfig, grpo_train  # noqa: E402
from onaai.training.sft import SFTConfig, train_sft  # noqa: E402
from onaai.training.tokenizer_utils import (  # noqa: E402
    default_corpus_from_records,
    train_tiny_tokenizer,
)

SFT_PATH = "data/sample_sft.jsonl"
RL_PATH = "data/sample_rl.jsonl"


@pytest.fixture(scope="module")
def replica(tmp_path_factory):
    """Build a tiny tokenizer + model, save and reload them (round-trip)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("replica")

    sft = load_sft_jsonl(SFT_PATH)
    rl = load_rl_jsonl(RL_PATH)
    records = [vars(e) for e in sft] + [vars(e) for e in rl]
    corpus = default_corpus_from_records(records) + [sft_target_text(e) for e in sft]

    tokenizer = train_tiny_tokenizer(corpus, vocab_size=1000, save_dir=str(out))
    model = build_model("tiny", vocab_size=len(tokenizer))
    model.save_pretrained(str(out))

    # Reload via the same Auto* path the scripts use (tests save/load round-trip).
    tokenizer = AutoTokenizer.from_pretrained(str(out))
    model = AutoModelForCausalLM.from_pretrained(str(out))
    return model, tokenizer, sft, rl


def test_tokenizer_and_model_built(replica):
    model, tokenizer, _, _ = replica
    assert len(tokenizer) > 0
    assert count_parameters(model) > 0
    # chat template round-trips
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<|im_start|>" in text and "assistant" in text


def test_sft_prompt_masking(replica):
    _, tokenizer, sft, _ = replica
    feat = tokenize_sft_example(sft[0], tokenizer, max_length=128)
    assert len(feat["input_ids"]) == len(feat["labels"]) == len(feat["attention_mask"])
    # Some prompt tokens are masked (-100) and some completion tokens are not.
    assert any(l == -100 for l in feat["labels"])
    assert any(l != -100 for l in feat["labels"])


def test_sft_then_rl_end_to_end(replica):
    model, tokenizer, sft, rl = replica

    # --- SFT phase ---
    sft_cfg = SFTConfig(
        output_dir="/tmp/onaai-e2e-sft",
        epochs=3.0,
        batch_size=4,
        learning_rate=1e-3,
        max_length=128,
        logging_steps=1,
    )
    trainer = train_sft(model, tokenizer, sft, sft_cfg)
    losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
    assert len(losses) > 0
    assert all(math.isfinite(l) for l in losses)

    # --- RL phase (GRPO-style) on a small subset to keep it fast ---
    rl_cfg = GRPOConfig(
        group_size=4,
        epochs=1,
        learning_rate=1e-4,
        max_new_tokens=16,
        max_prompt_length=64,
        seed=0,
    )
    history = grpo_train(model, tokenizer, rl[:4], config=rl_cfg)
    assert len(history["step_loss"]) > 0
    assert all(math.isfinite(x) for x in history["step_loss"])
    assert 0.0 <= history["mean_reward"] <= 1.0

    # --- generate + parse + reward (full inference path) ---
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 7 + 8?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=16, do_sample=False)
    completion = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    result = ReasoningEngine.parse("What is 7 + 8?", completion)
    assert isinstance(result.answer, str)  # parsing the model output works end-to-end
