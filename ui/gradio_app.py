import os
import gradio as gr
from app import run_agent


def set_groq_key(key: str):
    if key and isinstance(key, str) and key.strip():
        os.environ["GROQ_API_KEY"] = key.strip()
        return "✅ Groq key saved."
    return "⚠️ No key provided."


with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("## ☁️ SFDC Architect Agent")
        gr.Markdown(
            "Salesforce Technical Architect AI — grounded in live "
            "Salesforce documentation via MCP."
        )
        gr.HTML("<hr>")
        gr.Markdown("### ⚙️ Providers")
        groq_input = gr.Textbox(
            label="Groq API Key",
            type="password",
            placeholder="gsk_...",
        )
        gr.Markdown("_Ollama is used as fallback._")
        status_label = gr.Label(value="Status: Ready.")
        groq_input.change(fn=set_groq_key, inputs=[groq_input], outputs=[status_label])
        gr.HTML("<hr>")
        gr.Markdown(
            "**Pipeline per turn:**\n"
            "1. Classify intent + entities\n"
            "2. Validate temporal scope\n"
            "3. Plan search queries\n"
            "4. Retrieve evidence (bounded)\n"
            "5. Normalize evidence\n"
            "6. Resolve evidence conflicts\n"
            "7. Generate answer (tier-routed)\n"
            "8. Grounding check\n\n"
            "**Model tiers:**\n"
            "- Fast — facts, definitions\n"
            "- Standard — comparisons, research\n"
            "- Reasoning — architecture, complex code"
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
