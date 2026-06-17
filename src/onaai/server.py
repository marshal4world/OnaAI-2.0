"""Optional local HTTP API server for OnaAI-2.0.

Requires the ``server`` extra:  ``pip install -e ".[server]"``

Security:
    This server has NO authentication. It is intended for local use only and
    binds to 127.0.0.1 by default. Do not expose it directly to an untrusted
    network; put it behind an authenticating reverse proxy if you must.
"""

from __future__ import annotations

from typing import Optional

from .config import Config
from .engine import ReasoningEngine


def create_app(config: Optional[Config] = None):
    """Create and return a FastAPI application bound to a lazily-loaded engine."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="OnaAI-2.0", version="2.0.0")

    class SolveRequest(BaseModel):
        prompt: str
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        max_new_tokens: Optional[int] = None

    class SolveResponse(BaseModel):
        answer: str
        reasoning: str

    # The engine (and model weights) load on first use, not at import time.
    state: dict = {"engine": None}

    def get_engine() -> ReasoningEngine:
        if state["engine"] is None:
            cfg = config or Config()
            state["engine"] = ReasoningEngine(__import__(
                "onaai.model", fromlist=["ReasoningModel"]
            ).ReasoningModel(cfg), cfg)
        return state["engine"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "OnaAI-2.0"}

    @app.post("/solve", response_model=SolveResponse)
    def solve(req: SolveRequest) -> SolveResponse:
        overrides = {
            k: v
            for k, v in {
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_new_tokens": req.max_new_tokens,
            }.items()
            if v is not None
        }
        result = get_engine().solve(req.prompt, **overrides)
        return SolveResponse(answer=result.answer, reasoning=result.reasoning)

    return app


def run_server(config: Optional[Config] = None, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    if host not in ("127.0.0.1", "localhost"):
        print(
            f"[OnaAI-2.0] WARNING: binding to {host} exposes an UNAUTHENTICATED "
            "API. Use a reverse proxy with auth for non-local access."
        )
    uvicorn.run(create_app(config), host=host, port=port)
