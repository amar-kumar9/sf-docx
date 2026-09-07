# Salesforce Docs Agent

A local AI agent that answers Salesforce questions by retrieving and grounding answers against live Salesforce documentation — not model training data.

The **Python agent always owns retrieval, Architecture Center routing, and grounding**. Synthesis uses whatever LLM is on *this* machine. If none is configured, Python still returns the retrieved docs (retrieval-only), and Cursor can synthesize from that pack.

## Two ways to run it

| Where you are | Who writes the answer | What you run |
|---|---|---|
| Cursor on a new laptop (no Groq/Ollama key) | Cursor (local agent) | Ask in chat. The project skill runs `python cli.py retrieve --json`. |
| Gradio / CLI with a key or Ollama | Python agent | `python cli.py ui` or `python cli.py ask "..."` |

`python cli.py status` shows what this machine can use.

## Connect an LLM (pick one)

Copy `.env.example` to `.env`, or use the Gradio sidebar. Keys stay in `.env` (gitignored).

1. **Cursor** — no key. The skill in `.cursor/skills/sfdc-docs-agent/` retrieves via Python, then the local agent writes the answer.
2. **Ollama** — install [Ollama](https://ollama.com), pull a model, leave `LLM_PROVIDER=auto`.
3. **LM Studio** — start the local server (`http://127.0.0.1:1234/v1`).
4. **Cloud key** — set any one of `GROQ_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.
5. **Custom OpenAI-compatible server** — `LLM_PROVIDER=openai_compatible` plus `LLM_BASE_URL` and `LLM_API_KEY` if required.

`LLM_PROVIDER=auto` uses the first ready cloud key, then Ollama, then LM Studio. Set `LLM_PREFER_LOCAL=true` to try local runtimes first.

If nothing is configured, `python cli.py ask` still retrieves Salesforce docs and prints them as `[retrieval-only]`.

## Commands

```bash
python cli.py status
python cli.py retrieve --json "What is a Platform Event?"
python cli.py ask "Can Flow replace an Apex trigger?"
python cli.py ui
```

## Evaluate

```bash
python eval/agent_eval.py --compare --baseline-google-mode none --candidate-google-mode hybrid
```

## Diagnose retrieval

```bash
python scripts/diagnose_retrieval.py
```

## Layout

- `app.py` — retrieval pipeline, grounding, temporal validation, answer synthesis
- `solution_architect.py` — business requirement → Salesforce capability translation
- `providers.py` — pluggable LLMs and retrieve-only fallback
- `cli.py` — status / retrieve / ask / ui
- `prompts.py` — prompt templates
- `data/` — golden dataset
- `eval/` — eval harness, MCP fixtures, architecture probes
- `scripts/` — retrieval diagnostics
- `tests/` — pipeline tests
- `.cursor/skills/sfdc-docs-agent/` — Cursor skill (host agent + Python retrieval)

## Notes

- Keep `data/golden_dataset.json` versioned.
- Keep `eval/eval_mcp_fixtures.json` versioned.
- Generated logs and eval outputs belong under `artifacts/` or `logs/`.
- Never commit `.env`.
