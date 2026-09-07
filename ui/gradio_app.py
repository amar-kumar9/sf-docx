import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from app import run_agent
from providers import PROVIDER_CHOICES, apply_runtime_provider, provider_status


def _status_markdown() -> str:
    status = provider_status()
    active = status["active"] or "none — retrieval-only"
    lines = [
        f"**Active LLM:** `{active}`",
        f"**Selected:** `{status['selected']}`",
        "",
        "| Provider | Ready |",
        "|---|---|",
    ]
    for row in status["providers"]:
        mark = "yes" if row["available"] else "no"
        lines.append(f"| {row['label']} | {mark} |")
    if not status["active"]:
        lines.append("")
        lines.append(
            "No model on this machine. Chat still retrieves Salesforce docs. "
            "Paste a key below, start Ollama/LM Studio, or ask in Cursor."
        )
    return "\n".join(lines)


def save_provider(provider: str, api_key: str, base_url: str, model: str) -> str:
    apply_runtime_provider(provider or "auto", api_key or "", base_url or "", model or "")
    return _status_markdown()


with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("## SFDC Architect Agent")
        gr.Markdown(
            "Python retrieves and grounds Salesforce docs. "
            "Synthesis uses whichever LLM this machine has."
        )
        status_md = gr.Markdown(_status_markdown())
        provider = gr.Dropdown(
            choices=list(PROVIDER_CHOICES),
            value=os.getenv("LLM_PROVIDER", "auto"),
            label="LLM provider",
        )
        api_key = gr.Textbox(
            label="API key (optional for Ollama / LM Studio)",
            type="password",
            placeholder="sk-... or gsk_...",
        )
        base_url = gr.Textbox(
            label="Base URL (Ollama, LM Studio, or custom)",
            placeholder="http://127.0.0.1:1234/v1",
        )
        model = gr.Textbox(
            label="Model name (optional)",
            placeholder="gpt-4o-mini, llama3.1, ...",
        )
        save_btn = gr.Button("Save provider on this machine")
        save_btn.click(
            fn=save_provider,
            inputs=[provider, api_key, base_url, model],
            outputs=[status_md],
        )
        gr.Markdown(
            "Keys are written to a local `.env` (not committed). "
            "`auto` uses the first ready cloud key, then Ollama, then LM Studio."
        )

    gr.ChatInterface(
        fn=run_agent,
        fill_height=True,
        examples=[
            "Agentforce Coworker Benefits and Use Cases",
            "What is the latest Salesforce API version?",
            "What is a Platform Event?",
            "Can Flow replace an Apex trigger?",
            "We process 500,000 records every night. What Salesforce architecture should we use?",
            "Why am I getting Too many SOQL queries?",
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
