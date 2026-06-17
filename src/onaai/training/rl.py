"""GRPO-style reinforcement learning (the "Signal" phase).

VibeThinker amplifies correct reasoning with a verifiable-reward RL stage
(their MGPO). MGPO is a custom max-entropy variant; here we implement the
closely-related, widely-used **GRPO** (Group Relative Policy Optimization)
idea in a compact, dependency-light form:

  1. For each prompt, sample a *group* of G completions from the current policy.
  2. Score each completion with the verifiable reward.
  3. Compute **group-relative advantages**: ``A_i = (r_i - mean) / (std + eps)``
     (the group mean acts as the baseline -- no separate value network).
  4. Take a policy-gradient step: ``loss = -mean(A_i * logprob(completion_i))``.

This is intentionally minimal (single gradient step per group, no PPO clipping
or KL term) so it is easy to read and runs on CPU. The hooks for a reference
model / KL penalty are noted inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .data import build_prompt_messages, RLExample
from .reward import verifiable_reward

RewardFn = Callable[[str, str], float]


@dataclass
class GRPOConfig:
    group_size: int = 4          # G: completions sampled per prompt
    learning_rate: float = 1e-4
    epochs: int = 1              # passes over the prompt set
    max_new_tokens: int = 40
    max_prompt_length: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    adv_eps: float = 1e-6
    seed: int = 0
    # Periodic evaluation (pass@k / accuracy) on a held-out set during training.
    eval_every: int = 0          # run eval every N optimizer steps (0 = disabled)
    eval_k: int = 1
    eval_n_samples: int = 4


def _completion_logprobs(model, input_ids, attention_mask, prompt_len, completion_mask):
    """Sum of per-token log-probs of the completion tokens, per sequence.

    Returns a 1-D tensor (one scalar mean-logprob per sequence) with gradient.
    """
    import torch

    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]            # predict token t+1 from t
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

    mask = completion_mask[:, 1:].to(token_logp.dtype)               # align with shift
    summed = (token_logp * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts                                           # mean logprob / token


def _run_eval(model, tokenizer, eval_examples, config: "GRPOConfig", step: int) -> dict:
    """Run held-out pass@k/accuracy eval mid-training; returns a metrics record."""
    # Local import avoids a circular dependency (onaai.eval imports onaai.training).
    from ..eval import EvalConfig, evaluate

    was_training = model.training
    eval_cfg = EvalConfig(
        k=config.eval_k,
        n_samples=config.eval_n_samples,
        max_new_tokens=config.max_new_tokens,
        max_prompt_length=config.max_prompt_length,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
    )
    result = evaluate(model, tokenizer, eval_examples, eval_cfg)
    if was_training:
        model.train()
    return {
        "step": step,
        "pass_at_k": result.pass_at_k,
        "accuracy": result.accuracy,
        "k": result.k,
        "n_samples": result.n_samples,
    }


def grpo_train(
    model,
    tokenizer,
    examples: List[RLExample],
    reward_fn: Optional[RewardFn] = None,
    config: Optional[GRPOConfig] = None,
    eval_examples: Optional[List[RLExample]] = None,
):
    """Run GRPO-style RL. Returns a metrics dict (per-step rewards/losses).

    If ``config.eval_every > 0`` and ``eval_examples`` is provided, pass@k and
    accuracy are measured on the held-out set every ``eval_every`` optimizer
    steps and recorded under ``history["eval"]``.
    """
    import torch
    from transformers import GenerationConfig

    config = config or GRPOConfig()
    reward_fn = reward_fn or verifiable_reward
    torch.manual_seed(config.seed)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    gen_config = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        num_return_sequences=config.group_size,
        pad_token_id=pad_id,
    )

    do_eval = bool(config.eval_every and config.eval_every > 0 and eval_examples)
    history = {"step_reward": [], "step_loss": [], "eval": []}
    step = 0
    model.train()

    for _ in range(config.epochs):
        for ex in examples:
            prompt_text = tokenizer.apply_chat_template(
                build_prompt_messages(ex.problem),
                tokenize=False,
                add_generation_prompt=True,
            )
            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=config.max_prompt_length,
            )
            prompt_len = enc["input_ids"].shape[1]

            # --- 1. rollout: sample a group of completions ---
            model.eval()
            with torch.no_grad():
                seqs = model.generate(
                    **enc, generation_config=gen_config
                )  # [G, prompt_len + new]
            model.train()

            # --- 2. reward each completion ---
            rewards = []
            for i in range(seqs.shape[0]):
                completion_ids = seqs[i, prompt_len:]
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                rewards.append(float(reward_fn(text, ex.answer)))
            rewards_t = torch.tensor(rewards, dtype=torch.float32)

            # --- 3. group-relative advantages ---
            adv = (rewards_t - rewards_t.mean()) / (rewards_t.std() + config.adv_eps)

            # Build attention + completion masks for the (padded) group.
            attention_mask = (seqs != pad_id).long()
            attention_mask[:, :prompt_len] = 1  # prompt is always attended
            completion_mask = torch.zeros_like(seqs)
            for i in range(seqs.shape[0]):
                # completion runs from prompt_len up to (and including) first eos
                end = seqs.shape[1]
                row = seqs[i, prompt_len:]
                eos_pos = (row == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    end = prompt_len + int(eos_pos[0]) + 1
                completion_mask[i, prompt_len:end] = 1

            # --- 4. policy-gradient step ---
            mean_logp = _completion_logprobs(
                model, seqs, attention_mask, prompt_len, completion_mask
            )
            # (A reference-model KL term would be subtracted here for full GRPO.)
            loss = -(adv * mean_logp).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            history["step_reward"].append(rewards_t.mean().item())
            history["step_loss"].append(loss.item())

            # --- periodic held-out evaluation ---
            if do_eval and step % config.eval_every == 0:
                history["eval"].append(
                    _run_eval(model, tokenizer, eval_examples, config, step)
                )
                model.train()

    model.eval()
    # Final evaluation snapshot (if enabled and not already taken at this step).
    if do_eval and (not history["eval"] or history["eval"][-1]["step"] != step):
        history["eval"].append(_run_eval(model, tokenizer, eval_examples, config, step))

    history["mean_reward"] = (
        sum(history["step_reward"]) / len(history["step_reward"])
        if history["step_reward"]
        else 0.0
    )
    return history
