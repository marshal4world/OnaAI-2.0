# Datasets for the OnaAI-2.0 model replica

This directory ships small **sample** datasets that demonstrate the *schema* and
make the training pipeline runnable end-to-end. Replace them with your own
verifiable-reasoning data to train at scale.

The whole project targets **verifiable** reasoning: every example has a
ground-truth answer that can be checked programmatically (the basis of the
verifiable reward used in the RL phase).

## Files

| File              | Phase | Schema |
| ----------------- | ----- | ------ |
| `sample_sft.jsonl`| SFT   | `{"problem", "reasoning", "answer"}` |
| `sample_rl.jsonl` | RL    | `{"problem", "answer"}` |

All files are **JSON Lines** (one JSON object per line, UTF-8).

## SFT schema (`sample_sft.jsonl`)

Supervised fine-tuning teaches the model to produce a chain-of-thought and a
final boxed answer.

```json
{"problem": "What is 12 * 12?", "reasoning": "12 times 12 is 144.", "answer": "144"}
```

* `problem`   — the user prompt (string, required).
* `reasoning` — the chain-of-thought (string, required). Rendered inside
  `<think> ... </think>` in the training target.
* `answer`    — the verifiable final answer (string, required). Rendered as
  `\boxed{answer}`.

The training target string built for each row is:

```
<think>{reasoning}</think>
\boxed{answer}
```

Only the assistant/completion tokens contribute to the loss; the prompt tokens
are masked (see `onaai.training.data.tokenize_sft_example`).

## RL schema (`sample_rl.jsonl`)

The RL phase only needs a prompt and a checkable ground-truth answer. The model
generates candidate completions, and the **verifiable reward**
(`onaai.training.reward`) compares each completion's extracted answer against
`answer`.

```json
{"problem": "What is 9 + 10?", "answer": "19"}
```

* `problem` — the user prompt (string, required).
* `answer`  — the verifiable ground-truth answer (string, required).

## Bringing your own data

1. Produce JSONL files matching the schemas above.
2. Point the training configs/scripts at them
   (`--sft-data path.jsonl`, `--rl-data path.jsonl`).
3. For the SFT diversity ("spectrum") phase, include multiple distinct
   reasoning traces per problem (different solution paths, same answer) — this
   is the "spectrum" half of the SFT → RL recipe.
