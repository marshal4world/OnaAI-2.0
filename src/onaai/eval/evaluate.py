"""pass@k + accuracy evaluation on a held-out verifiable set.

``pass@k`` uses the **unbiased estimator** from Chen et al. 2021 (the Codex
"Evaluating Large Language Models Trained on Code" paper):

    pass@k = 1 - C(n - c, k) / C(n, k)

where ``n`` completions are sampled per problem and ``c`` of them are correct.
This is lower-variance and unbiased compared to naively sampling exactly ``k``.

A completion is "correct" iff its extracted answer matches the ground truth
(:func:`onaai.training.reward.answers_match`) -- the same verifiable signal used
by the RL reward, so training and evaluation agree on correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Callable, List, Optional

from ..training.data import RLExample, build_prompt_messages
from ..training.reward import answers_match

# A judge takes (completion_text, ground_truth) -> bool.
Judge = Callable[[str, str], bool]


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k estimate for one problem.

    Args:
        num_samples: total completions sampled for the problem (``n``).
        num_correct: how many were correct (``c``).
        k: the ``k`` in pass@k.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if k > num_samples:
        raise ValueError(f"k ({k}) cannot exceed num_samples ({num_samples})")
    if num_correct < 0 or num_correct > num_samples:
        raise ValueError("num_correct must be in [0, num_samples]")
    # If there are fewer than k incorrect samples, every size-k subset hits a
    # correct one -> pass@k = 1.
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - comb(num_samples - num_correct, k) / comb(num_samples, k)


@dataclass
class EvalConfig:
    k: int = 1
    n_samples: int = 1           # completions sampled per problem (n >= k)
    max_new_tokens: int = 64
    max_prompt_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    seed: int = 0


@dataclass
class ExampleEval:
    problem: str
    ground_truth: str
    num_correct: int
    num_samples: int
    pass_at_k: float
    sample_correct: bool         # was the first sampled completion correct?
    completions: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    num_examples: int
    k: int
    n_samples: int
    pass_at_k: float             # mean over examples
    accuracy: float              # fraction of examples whose 1st sample is correct
    per_example: List[ExampleEval] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"examples={self.num_examples}  "
            f"pass@{self.k}={self.pass_at_k:.3f}  "
            f"accuracy(sample@1)={self.accuracy:.3f}  "
            f"(n_samples={self.n_samples})"
        )


def generate_samples(
    model,
    tokenizer,
    problem: str,
    n: int,
    config: EvalConfig,
) -> List[str]:
    """Sample ``n`` completions for a single problem; returns decoded strings."""
    import torch

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_text = tokenizer.apply_chat_template(
        build_prompt_messages(problem), tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=config.max_prompt_length,
    )
    prompt_len = enc["input_ids"].shape[1]

    # Sampling is required for n > 1 to get distinct completions.
    do_sample = n > 1 or config.temperature > 0
    gen_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        num_return_sequences=n,
        pad_token_id=pad_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=config.temperature, top_p=config.top_p)

    model.eval()
    with torch.no_grad():
        seqs = model.generate(**enc, **gen_kwargs)

    return [
        tokenizer.decode(seqs[i, prompt_len:], skip_special_tokens=True)
        for i in range(seqs.shape[0])
    ]


def evaluate(
    model,
    tokenizer,
    examples: List[RLExample],
    config: Optional[EvalConfig] = None,
    judge: Optional[Judge] = None,
) -> EvalResult:
    """Evaluate a model on a held-out set, returning pass@k and accuracy."""
    import torch

    config = config or EvalConfig()
    judge = judge or answers_match
    n = max(config.n_samples, config.k)
    torch.manual_seed(config.seed)

    per_example: List[ExampleEval] = []
    for ex in examples:
        completions = generate_samples(model, tokenizer, ex.problem, n, config)
        correct_flags = [bool(judge(c, ex.answer)) for c in completions]
        c = sum(correct_flags)
        per_example.append(
            ExampleEval(
                problem=ex.problem,
                ground_truth=ex.answer,
                num_correct=c,
                num_samples=n,
                pass_at_k=estimate_pass_at_k(n, c, config.k),
                sample_correct=correct_flags[0] if correct_flags else False,
                completions=completions,
            )
        )

    num = len(per_example)
    pass_at_k = sum(e.pass_at_k for e in per_example) / num if num else 0.0
    accuracy = sum(1 for e in per_example if e.sample_correct) / num if num else 0.0

    return EvalResult(
        num_examples=num,
        k=config.k,
        n_samples=n,
        pass_at_k=pass_at_k,
        accuracy=accuracy,
        per_example=per_example,
    )
