#!/usr/bin/env python3
"""Salesforce docs agent entry point.

status     show which LLM / MCP options this machine can use
retrieve   Python retrieval pack (no LLM required) for Cursor or other host agents
ask        full pipeline when an LLM is configured; otherwise retrieval-only
ui         Gradio chat with a provider picker
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app import build_evidence_pack, format_retrieval_only_answer, run_agent
from providers import provider_status


def _question(parts: list[str]) -> str:
    return " ".join(parts).strip()


def cmd_status() -> int:
    status = provider_status()
    print(f"Selected provider: {status['selected']}")
    print(f"Active LLM:        {status['active'] or 'none (retrieval-only)'}")
    print(f"Prefer local:      {status['prefer_local']}")
    print(f"Env file:          {status['env_file']} ({'exists' if status['env_file_exists'] else 'missing — copy .env.example'})")
    print()
    print(f"{'Provider':<24} {'Kind':<8} {'Ready':<6} How to connect")
    for row in status["providers"]:
        ready = "yes" if row["available"] else "no"
        print(f"{row['label']:<24} {row['kind']:<8} {ready:<6} {row['detail']}")
    print()
    if not status["active"]:
        print("No LLM is configured. The Python agent still retrieves Salesforce docs.")
        print("Pick one:")
        print("  1. Ask in Cursor — sfdocx-capability-loop uses Salesforce Docs MCP (no .env)")
        print("  2. Start Ollama or LM Studio, then re-run this command")
        print("  3. python cli.py ui  and paste a Groq / OpenAI / Anthropic / Gemini key")
        print("  4. Copy .env.example to .env and set a key")
    return 0


def cmd_retrieve(question: str, as_json: bool) -> int:
    if not question:
        print("Pass a question: python cli.py retrieve \"What is a Platform Event?\"", file=sys.stderr)
        return 2
    pack = build_evidence_pack(question, llm=None)
    if as_json:
        print(json.dumps(pack, indent=2, ensure_ascii=False))
        return 0
    if pack.get("blocked"):
        print(pack.get("answer") or "Blocked.")
        return 0
    print(pack.get("host_instructions", ""))
    print()
    print(
        format_retrieval_only_answer(
            question,
            pack.get("intent") or {},
            pack.get("evidence") or [],
            pack.get("temporal") or {},
            pack.get("solution_plan") or {},
        )
    )
    return 0


def cmd_ask(question: str) -> int:
    if not question:
        print("Pass a question: python cli.py ask \"What is a Platform Event?\"", file=sys.stderr)
        return 2
    print(run_agent(question, []))
    return 0


def cmd_ui() -> int:
    from ui.gradio_app import demo

    demo.launch(theme=__import__("gradio").themes.Soft())
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"status", "retrieve", "ask", "ui", "-h", "--help"}
    if argv and argv[0] not in commands:
        argv = ["ask", *argv]

    parser = argparse.ArgumentParser(
        description="Salesforce docs agent — retrieval always works; synthesis uses whatever LLM this machine has.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show detected LLM providers and how to connect one")
    sub.add_parser("ui", help="Open the Gradio chat UI")

    retrieve = sub.add_parser("retrieve", help="Retrieve Salesforce docs without an LLM")
    retrieve.add_argument("question", nargs="+")
    retrieve.add_argument("--json", action="store_true", help="Print the evidence pack as JSON")

    ask = sub.add_parser("ask", help="Full agent if an LLM is configured; otherwise retrieval-only")
    ask.add_argument("question", nargs="+")

    args = parser.parse_args(argv)
    if args.command == "status":
        return cmd_status()
    if args.command == "ui":
        return cmd_ui()
    if args.command == "retrieve":
        return cmd_retrieve(_question(args.question), args.json)
    if args.command == "ask":
        return cmd_ask(_question(args.question))
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    raise SystemExit(main())
