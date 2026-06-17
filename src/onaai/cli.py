"""Command-line interface for OnaAI-2.0.

Usage:
    onaai solve "your problem here" [--config path] [--show-reasoning]
    onaai chat   [--config path]
    onaai serve  [--host 127.0.0.1] [--port 8000] [--config path]
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import __version__
from .config import load_config
from .engine import ReasoningEngine


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onaai",
        description="OnaAI-2.0 — verifiable reasoning on top of VibeThinker-3B",
    )
    parser.add_argument("--version", action="version", version=f"OnaAI-2.0 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="solve a single problem and print the answer")
    p_solve.add_argument("prompt", help="the problem/question to solve")
    p_solve.add_argument("--config", help="path to a YAML config file")
    p_solve.add_argument(
        "--show-reasoning", action="store_true", help="also print the chain-of-thought"
    )

    p_chat = sub.add_parser("chat", help="interactive REPL")
    p_chat.add_argument("--config", help="path to a YAML config file")

    p_serve = sub.add_parser("serve", help="run the local HTTP API server")
    p_serve.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    p_serve.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
    p_serve.add_argument("--config", help="path to a YAML config file")

    return parser


def _cmd_solve(args: argparse.Namespace) -> int:
    engine = ReasoningEngine.from_default(args.config)
    result = engine.solve(args.prompt)
    if args.show_reasoning and result.reasoning:
        print("=== reasoning ===")
        print(result.reasoning)
        print("=== answer ===")
    print(result.answer)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    engine = ReasoningEngine.from_default(args.config)
    print("OnaAI-2.0 interactive chat. Type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        result = engine.solve(prompt)
        print(result.answer)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    # Imported lazily so the optional server deps aren't required for solve/chat.
    from .server import run_server

    config = load_config(args.config)
    run_server(config, host=args.host, port=args.port)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    dispatch = {
        "solve": _cmd_solve,
        "chat": _cmd_chat,
        "serve": _cmd_serve,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
