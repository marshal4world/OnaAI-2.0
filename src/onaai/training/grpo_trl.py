"""TRL-based GRPO: a drop-in alternative to ``onaai.training.rl.grpo_train``.

`onaai.training.rl` ships a compact, dependency-light GRPO loop that runs
anywhere (including CPU). This module instead delegates to Hugging Face
**TRL**'s production ``GRPOTrainer`` while reusing OnaAI's *verifiable reward*,
so you get TRL's batching/logging/scaling with the same correctness signal.

Use this when you have a TRL-capable environment (TRL installed, plus a working
generation backend). Otherwise prefer the built-in loop.

Reward adapter
--------------
TRL reward functions have the signature ``fn(prompts, completions, **columns)``
and return a list of floats, where ``**columns`` carries the extra dataset
columns (here, ``answer``). :func:`make_verifiable_reward_func` adapts OnaAI's
:func:`onaai.training.reward.verifiable_reward` to that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .data import RLExample, build_prompt_messages
from .reward import verifiable_reward


class TRLUnavailableError(RuntimeError):
    """Raised when TRL's GRPOTrainer cannot be imported in this environment."""


def _import_trl():
    """Import TRL's GRPOTrainer/GRPOConfig with a clear, actionable error.

    TRL lazily imports its trainers and can fail not only with ``ImportError``
    (TRL not installed) but also ``RuntimeError`` from its lazy module loader
    (e.g. an optional backend like vLLM is mis-detected). We surface both as a
    single, explanatory :class:`TRLUnavailableError`.
    """
    try:
        from trl import GRPOConfig as TRLGRPOConfig  # noqa: N811
        from trl import GRPOTrainer
    except Exception as e:  # ImportError or TRL's lazy-loader RuntimeError
        raise TRLUnavailableError(
            "TRL's GRPOTrainer is unavailable in this environment "
            f"({type(e).__name__}: {e}).\n"
            'Install a TRL-capable setup with:  pip install -e ".[trl]"\n'
            "Note: some TRL versions eagerly require a generation backend "
            "(e.g. vLLM). If you do not have one, use the built-in loop "
            "`onaai.training.rl.grpo_train` instead."
        ) from e
    return GRPOTrainer, TRLGRPOConfig


@dataclass
class TRLGRPOSettings:
    output_dir: str = "checkpoints/grpo_trl"
    num_generations: int = 4         # group size G
    learning_rate: float = 1e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    max_prompt_length: int = 256
    max_completion_length: int = 64
    temperature: float = 1.0
    top_p: float = 0.95
    logging_steps: int = 5
    use_vllm: bool = False
    seed: int = 0


def make_verifiable_reward_func(
    base_reward: Callable[[str, str], float] = verifiable_reward,
):
    """Return a TRL-compatible reward function backed by the verifiable reward."""

    def reward_func(prompts=None, completions=None, **columns) -> List[float]:
        completions = completions or []
        answers = columns.get("answer")
        if answers is None:
            raise KeyError(
                "reward_func expected an 'answer' column from the dataset"
            )
        rewards: List[float] = []
        for completion, answer in zip(completions, answers):
            # TRL may pass conversational completions (list of message dicts).
            text = _completion_to_text(completion)
            rewards.append(float(base_reward(text, answer)))
        return rewards

    reward_func.__name__ = "verifiable_reward"  # TRL uses this for logging
    return reward_func


def _completion_to_text(completion) -> str:
    """Normalize a TRL completion (str or chat message list) to plain text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # [{"role": ..., "content": ...}, ...]
        return " ".join(
            m.get("content", "") for m in completion if isinstance(m, dict)
        )
    return str(completion)


def build_trl_dataset(examples: List[RLExample], tokenizer):
    """Build a 🤗 ``datasets.Dataset`` with ``prompt`` and ``answer`` columns."""
    from datasets import Dataset

    prompts, answers = [], []
    for ex in examples:
        prompts.append(
            tokenizer.apply_chat_template(
                build_prompt_messages(ex.problem),
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        answers.append(ex.answer)
    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def grpo_train_trl(
    model,
    tokenizer,
    examples: List[RLExample],
    settings: Optional[TRLGRPOSettings] = None,
    reward_fn: Optional[Callable[[str, str], float]] = None,
):
    """Run GRPO via TRL's ``GRPOTrainer`` using OnaAI's verifiable reward.

    Returns the trained ``GRPOTrainer``. Raises :class:`TRLUnavailableError` if
    TRL's GRPOTrainer cannot be imported here.
    """
    GRPOTrainer, TRLGRPOConfig = _import_trl()
    settings = settings or TRLGRPOSettings()

    dataset = build_trl_dataset(examples, tokenizer)
    reward_func = make_verifiable_reward_func(reward_fn or verifiable_reward)

    config = TRLGRPOConfig(
        output_dir=settings.output_dir,
        num_generations=settings.num_generations,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        max_prompt_length=settings.max_prompt_length,
        max_completion_length=settings.max_completion_length,
        temperature=settings.temperature,
        top_p=settings.top_p,
        logging_steps=settings.logging_steps,
        use_vllm=settings.use_vllm,
        report_to=[],
        seed=settings.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer
