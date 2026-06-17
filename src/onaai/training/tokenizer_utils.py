"""Tokenizer utilities for the OnaAI-2.0 model replica.

Two paths, one interface (both return a 🤗 ``PreTrainedTokenizerFast``-compatible
tokenizer with a Qwen-style chat template):

* :func:`load_tokenizer` — load the **real** vendored Qwen2-compatible
  tokenizer from a local directory (after ``scripts/download_model.py``).
* :func:`train_tiny_tokenizer` — train a small **byte-level BPE** tokenizer on
  given texts. Fully offline; used by the end-to-end test so it needs no
  network or multi-GB download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

# Qwen2.5-style ChatML template, used so apply_chat_template() works for the
# tiny replica exactly like it does for the real model.
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
)

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
END_OF_TEXT = "<|endoftext|>"
SPECIAL_TOKENS = [END_OF_TEXT, IM_START, IM_END]


def load_tokenizer(path: str):
    """Load the real vendored tokenizer from a local directory or HF repo id."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def train_tiny_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 2000,
    save_dir: Optional[str] = None,
):
    """Train a tiny byte-level BPE tokenizer and wrap it for 🤗 transformers.

    Returns a ``PreTrainedTokenizerFast`` with special tokens and a ChatML
    chat template, ready for ``apply_chat_template``.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    texts = list(texts)

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=IM_START,
        eos_token=IM_END,
        pad_token=END_OF_TEXT,
        unk_token=None,
        additional_special_tokens=[IM_START, IM_END],
    )
    fast.chat_template = CHATML_TEMPLATE

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fast.save_pretrained(save_dir)

    return fast


def default_corpus_from_records(records: List[dict]) -> List[str]:
    """Build a training corpus for the tiny tokenizer from dataset records.

    Accepts SFT records (``problem``/``reasoning``/``answer``) and/or RL records
    (``problem``/``answer``), plus the special markup the pipeline emits.
    """
    corpus: List[str] = []
    for r in records:
        for key in ("problem", "reasoning", "answer"):
            if r.get(key):
                corpus.append(str(r[key]))
    # Make sure the structural tokens are well represented.
    corpus += ["<think>", "</think>", "\\boxed{", "}", "assistant", "user"]
    return corpus
