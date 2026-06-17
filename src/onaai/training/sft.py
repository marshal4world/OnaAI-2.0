"""Supervised fine-tuning (the "Spectrum" phase) for the VibeThinker replica.

Thin wrapper over 🤗 ``Trainer`` with prompt-masked labels. Heavy imports are
local so the rest of the package stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .data import SFTDataset, SFTExample


@dataclass
class SFTConfig:
    output_dir: str = "checkpoints/sft"
    epochs: float = 1.0
    batch_size: int = 2
    learning_rate: float = 5e-4
    max_length: int = 1024
    logging_steps: int = 5
    seed: int = 0


class SFTCollator:
    """Pad a batch of ``{input_ids, attention_mask, labels}`` to equal length."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_sft(
    model,
    tokenizer,
    examples: List[SFTExample],
    config: Optional[SFTConfig] = None,
):
    """Run supervised fine-tuning. Returns the 🤗 ``Trainer`` (already trained)."""
    from transformers import Trainer, TrainingArguments

    config = config or SFTConfig()
    dataset = SFTDataset(examples, tokenizer, max_length=config.max_length)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    collator = SFTCollator(pad_id)

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        seed=config.seed,
        use_cpu=True,  # the e2e test runs on CPU; ignored when a GPU is present
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    return trainer
