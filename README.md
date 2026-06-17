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
│   └── default.yaml
├── src/onaai/
│   ├── __init__.py
│   ├── config.py        # config loading + env overrides
│   ├── model.py         # VibeThinker-3B wrapper (transformers/vLLM)
│   ├── engine.py        # ReasoningEngine: answer extraction
│   ├── cli.py           # `onaai` command
│   └── server.py        # optional FastAPI app
├── examples/
│   └── solve_math.py
└── tests/
    ├── test_config.py
    └── test_engine.py
```

## License

MIT — see [LICENSE](./LICENSE). VibeThinker-3B is also MIT-licensed by WeiboAI.

## Acknowledgements

Built on [VibeThinker-3B](https://huggingface.co/WeiboAI/VibeThinker-3B) by WeiboAI.
