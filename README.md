# OnaAI-2.0

OnaAI-2.0 is a lightweight reasoning assistant built **on top of [VibeThinker-3B](https://github.com/WeiboAI/VibeThinker)** — a 3-billion-parameter dense reasoning model from WeiboAI that targets *verifiable* reasoning (competition math, competitive programming, and STEM).

OnaAI-2.0 wraps the raw model with:

- A clean **model wrapper** that supports both the `transformers` and `vLLM` backends.
- A higher-level **reasoning engine** that separates the model's chain-of-thought from the final answer.
- A **CLI** for interactive and one-shot use.
- An optional **FastAPI server** for local HTTP access.
- **Config-driven** defaults (sampling params, backend, model path).

> ⚠️ OnaAI-2.0 inherits VibeThinker-3B's scope. It is excellent at *verifiable* tasks (math, code, STEM) and is **not** a general-purpose open-domain chatbot. For broad knowledge tasks, use a larger general model.

---

## Architecture

```
            ┌──────────────────────────────────────────┐
            │                OnaAI-2.0                   │
            │                                            │
  user ──▶  │  CLI / API  ──▶  ReasoningEngine          │
            │                      │                     │
            │                      ▼                     │
            │              VibeThinkerModel              │
            │           (transformers | vLLM)            │
            │                      │                     │
            └──────────────────────┼─────────────────────┘
                                   ▼
                          WeiboAI/VibeThinker-3B
```

## Requirements

- Python >= 3.9
- `transformers>=4.54.0`
- (recommended) `vLLM==0.10.1` for fast inference
- A GPU is strongly recommended; the 3B model loads in bf16.

Install:

```bash
pip install -e .
# or, for the API server extras:
pip install -e ".[server]"
```

## Vendoring the model locally (download tokenizer + weights)

By default OnaAI-2.0 pulls `WeiboAI/VibeThinker-3B` from the Hugging Face Hub on
first use and caches it. To keep the tokenizer files and weights on local disk
(e.g. for offline/air-gapped use), vendor them into `models/`:

```bash
# Full model + tokenizer -> ./models/VibeThinker-3B   (several GB)
python scripts/download_model.py

# Tokenizer + config only (fast, no weights) -- handy for building the
# data/training pipeline before downloading the full weights:
python scripts/download_model.py --tokenizer-only

# Then point OnaAI-2.0 at the local copy:
export ONAAI_MODEL_PATH=models/VibeThinker-3B
```

> Weights are git-ignored (`*.safetensors`, `models/`). Never commit multi-GB
> weights into the repo — keep them on disk or use the HF Hub / Git LFS.

## Quick start

### CLI — one-shot

```bash
onaai solve "Find the number of ordered pairs (a, b) of integers with 1 <= a, b <= 100 such that a*b is a perfect square."
```

### CLI — interactive REPL

```bash
onaai chat
```

### Python

```python
from onaai import ReasoningEngine

engine = ReasoningEngine.from_default()
result = engine.solve("What is the remainder when 7^100 is divided by 13?")
print(result.answer)      # final answer (extracted)
print(result.reasoning)   # full chain-of-thought
```

### Local API server

```bash
onaai serve --host 127.0.0.1 --port 8000
# POST http://127.0.0.1:8000/solve  {"prompt": "..."}
```

> 🔒 **Security note:** the bundled server has **no authentication**. Bind it to `127.0.0.1` (the default) and do not expose it to a public network without adding an auth layer / reverse proxy.

## Training a replica (SFT → RL)

OnaAI-2.0 includes a runnable replica of VibeThinker's architecture and
**Spectrum-to-Signal** training recipe. Two scales share the same code:

| Scale | Model | Tokenizer | Runs on |
| ----- | ----- | --------- | ------- |
| `tiny` | ~100K-param Qwen2-arch (random init) | small BPE trained on the data | **CPU, seconds** |
| `vibethinker-3b` | real ~3B Qwen2 dims | real vendored tokenizer | GPU(s) |

> ⚠️ This reproduces the **architecture + tokenizer + training pipeline**, not
> the real trained weights. VibeThinker's SSP corpus is private, so bring your
> own verifiable-reasoning data (see [`data/README.md`](./data/README.md)).
> Building from a preset yields *fresh random weights*; to start from the real
> pretrained model, vendor it first (above) and point the scripts at that dir.

### 1. Build a tiny replica (tokenizer + model)

```bash
python scripts/build_tiny_replica.py --out models/tiny-replica
```

### 2. SFT — the "Spectrum" phase

Supervised fine-tuning on `{problem, reasoning, answer}` data. The target is
`<think>{reasoning}</think>\boxed{answer}`, and prompt tokens are masked so the
loss only covers the completion.

```bash
python scripts/train_sft.py --model models/tiny-replica --out checkpoints/sft
```

### 3. RL — the "Signal" phase (GRPO-style)

Group Relative Policy Optimization with a **verifiable reward** (the model is
rewarded only when its extracted `\boxed{}` answer matches the ground truth).
This is the open analogue of VibeThinker's MGPO.

```bash
python scripts/train_rl.py --model checkpoints/sft --out checkpoints/rl
```

You can monitor held-out **pass@k / accuracy during RL** by passing an eval set;
metrics are recorded under `history["eval"]` and a final snapshot is printed:

```bash
python scripts/train_rl.py --model checkpoints/sft --out checkpoints/rl \
    --eval-data data/sample_eval.jsonl --eval-every 10 --eval-k 1 --eval-n-samples 4
```

#### TRL-based GRPO (alternative backend)

For a production RL backend, `onaai.training.grpo_trl.grpo_train_trl` is a
drop-in that delegates to Hugging Face **TRL**'s `GRPOTrainer` while reusing
OnaAI's verifiable reward:

```bash
pip install -e ".[trl]"
```

```python
from onaai.training.grpo_trl import grpo_train_trl, TRLGRPOSettings
grpo_train_trl(model, tokenizer, examples, TRLGRPOSettings(num_generations=8))
```

> The built-in `grpo_train` runs anywhere (incl. CPU). The TRL path needs a
> TRL-capable environment; some TRL versions also require a generation backend
> (e.g. vLLM). If TRL's `GRPOTrainer` can't be imported, `grpo_train_trl` raises
> a clear `TRLUnavailableError` pointing you back to the built-in loop.

### End-to-end test

The full pipeline (build tokenizer → build model → save/reload → SFT → RL →
generate → parse + reward) runs as a CPU test in a few seconds:

```bash
PYTHONPATH=src pytest tests/test_end_to_end.py -q
```

### Evaluation (pass@k + accuracy)

Evaluate a model on a held-out verifiable set. A completion counts as correct
iff its extracted `\boxed{}` answer matches the ground truth — the same signal
used by the RL reward. `pass@k` uses the **unbiased estimator** from the Codex
paper (`1 - C(n-c, k) / C(n, k)`).

```bash
python scripts/evaluate.py --model checkpoints/rl \
    --eval-data data/sample_eval.jsonl --k 1 --n-samples 4
# -> examples=8  pass@1=0.000  accuracy(sample@1)=0.000  (n_samples=4)
```

Programmatic use:

```python
from onaai.eval import evaluate, EvalConfig
result = evaluate(model, tokenizer, examples, EvalConfig(k=4, n_samples=8))
print(result.summary())
```

### Scaling to the real 3B

```bash
# 1. vendor the real tokenizer (+ optionally weights)
python scripts/download_model.py
# 2. build at real dimensions, or fine-tune the real weights directly
python scripts/build_tiny_replica.py --preset vibethinker-3b \
    --vocab-size 151936 --out models/vibethinker-3b-fresh
# 3. SFT / RL with your own large verifiable dataset on GPU(s)
```

## Configuration

Defaults live in `config/default.yaml`. Override with a `--config path.yaml` flag,
or with environment variables prefixed `ONAAI_` (e.g. `ONAAI_MODEL_PATH`).

| Key                  | Default                  | Description                              |
| -------------------- | ------------------------ | ---------------------------------------- |
| `model_path`         | `WeiboAI/VibeThinker-3B` | HF repo id or local path                 |
| `backend`            | `transformers`           | `transformers` or `vllm`                 |
| `temperature`        | `1.0`                    | sampling temperature                     |
| `top_p`              | `0.95`                   | nucleus sampling                         |
| `top_k`              | `-1`                     | `-1` disables top-k (vLLM/SGLang style)  |
| `max_new_tokens`     | `40960`                  | long CoT needs a large budget            |

## Project layout

```
OnaAI-2.0/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── default.yaml     # inference defaults
│   └── train.yaml       # training (SFT + RL) config
├── data/
│   ├── README.md        # dataset schema docs
│   ├── sample_sft.jsonl # sample SFT data (problem/reasoning/answer)
│   └── sample_rl.jsonl  # sample RL data (problem/answer)
├── scripts/
│   ├── download_model.py     # vendor real tokenizer + weights locally
│   ├── build_tiny_replica.py # build tiny tokenizer + model
│   ├── train_sft.py          # run SFT
│   ├── train_rl.py           # run GRPO-style RL
│   └── evaluate.py           # pass@k + accuracy on a held-out set
├── src/onaai/
│   ├── __init__.py
│   ├── config.py        # config loading + env overrides
│   ├── model.py         # VibeThinker-3B inference wrapper (transformers/vLLM)
│   ├── modeling.py      # build Qwen2-arch replica (3B + tiny presets)
│   ├── engine.py        # ReasoningEngine: answer extraction
│   ├── cli.py           # `onaai` command
│   ├── server.py        # optional FastAPI app
│   ├── training/
│   │   ├── data.py            # JSONL loader + prompt-masked tokenization
│   │   ├── reward.py          # verifiable reward (boxed-answer match)
│   │   ├── sft.py             # SFT trainer
│   │   ├── rl.py              # GRPO-style RL loop
│   │   └── tokenizer_utils.py # load real / train tiny tokenizer
│   └── eval/
│       └── evaluate.py        # unbiased pass@k estimator + evaluate()
├── examples/
│   └── solve_math.py
└── tests/
    ├── test_config.py
    ├── test_engine.py
    ├── test_reward.py
    ├── test_training_data.py
    ├── test_eval.py
    └── test_end_to_end.py     # full pipeline on CPU
```

## License

MIT — see [LICENSE](./LICENSE). VibeThinker-3B is also MIT-licensed by WeiboAI.

## Acknowledgements

Built on [VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) by WeiboAI.
