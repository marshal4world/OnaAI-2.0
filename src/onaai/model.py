"""Model wrapper around VibeThinker-3B.

Supports two backends:
  * ``transformers`` — works everywhere, simplest to set up.
  * ``vllm``         — much faster, requires the optional ``vllm`` dependency.

Heavy ML libraries are imported lazily so that the rest of OnaAI-2.0
(config parsing, answer extraction, tests) can be used without a GPU or a
multi-gigabyte model download.
"""

from __future__ import annotations

from typing import List, Optional

from .config import Config


class VibeThinkerModel:
    """Thin, backend-agnostic wrapper around the VibeThinker-3B model."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._model = None
        self._tokenizer = None
        self._llm = None  # vLLM engine handle
        self._loaded = False

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(self) -> "VibeThinkerModel":
        """Load model weights into memory. Idempotent."""
        if self._loaded:
            return self
        if self.config.backend == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()
        self._loaded = True
        return self

    def _load_transformers(self) -> None:
        import torch  # noqa: F401  (ensures a clear error if torch is missing)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(__import__("torch"), self.config.dtype, None)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            low_cpu_mem_usage=True,
            torch_dtype=dtype or "auto",
            device_map="auto",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )

    def _load_vllm(self) -> None:
        from vllm import LLM  # type: ignore

        self._llm = LLM(model=self.config.model_path, dtype=self.config.dtype)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, **overrides) -> str:
        """Generate a completion for a single user ``prompt``.

        ``overrides`` may contain ``temperature``, ``top_p``, ``top_k`` or
        ``max_new_tokens`` to override the configured defaults for this call.
        """
        if not self._loaded:
            self.load()

        params = self._sampling_params(overrides)
        messages = [{"role": "user", "content": prompt}]

        if self.config.backend == "vllm":
            return self._generate_vllm(messages, params)
        return self._generate_transformers(messages, params)

    def _sampling_params(self, overrides: dict) -> dict:
        return {
            "temperature": overrides.get("temperature", self.config.temperature),
            "top_p": overrides.get("top_p", self.config.top_p),
            "top_k": overrides.get("top_k", self.config.top_k),
            "max_new_tokens": overrides.get("max_new_tokens", self.config.max_new_tokens),
        }

    def _generate_transformers(self, messages: List[dict], params: dict) -> str:
        from transformers import GenerationConfig

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # temperature <= 0 means greedy decoding (do_sample=False). Passing
        # temperature/top_p with sampling disabled is invalid in transformers.
        top_k = params["top_k"]
        if params["temperature"] and params["temperature"] > 0:
            gen_config = GenerationConfig(
                max_new_tokens=params["max_new_tokens"],
                do_sample=True,
                temperature=params["temperature"],
                top_p=params["top_p"],
                top_k=None if top_k is not None and top_k < 0 else top_k,
            )
        else:
            gen_config = GenerationConfig(
                max_new_tokens=params["max_new_tokens"],
                do_sample=False,
            )
        generated = self._model.generate(**model_inputs, generation_config=gen_config)
        trimmed = [
            out[len(inp):] for inp, out in zip(model_inputs.input_ids, generated)
        ]
        return self._tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _generate_vllm(self, messages: List[dict], params: dict) -> str:
        from vllm import SamplingParams  # type: ignore

        sp = SamplingParams(
            temperature=params["temperature"],
            top_p=params["top_p"],
            top_k=params["top_k"],
            max_tokens=params["max_new_tokens"],
        )
        outputs = self._llm.chat(messages, sp)
        return outputs[0].outputs[0].text

    @property
    def is_loaded(self) -> bool:
        return self._loaded
