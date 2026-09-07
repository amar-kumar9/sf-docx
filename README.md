# Salesforce Docs Agent

Grounds Salesforce design answers in **live official docs**. In Cursor, the model is the agent. Search goes through the Salesforce Docs MCP. There is no `.env` on that path.

## Cursor (this is the product)

1. Open this folder in Cursor.
2. Enable the **salesforce-docs** MCP if Cursor prompts (project file: `.cursor/mcp.json`).
3. Ask a design question.

The skill `.cursor/skills/sfdocx-capability-loop/SKILL.md` tells the agent to search capability-first, keep going until Architecture Center hits, and refuse when docs do not support a product. For heap, API version, and other seasonal facts it **must open Help release notes in the browser** — Salesforce Docs MCP often still has last season’s `apex_gov_limits` page and will miss `release-notes.rn_*` articles.

## Optional: Python CLI / Gradio

`python cli.py` is a separate app for evals and a local UI. It is **not** what the Cursor skill runs. That path uses `.env` for LLM keys and `MCP_URL` (a different HTTP index). Copy `.env.example` only if you are running Gradio or `python cli.py ask`.

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

## Layout

- `.cursor/skills/sfdocx-capability-loop/` — Cursor skill (agent + Salesforce Docs MCP)
- `.cursor/mcp.json` — project Salesforce Docs MCP
- `app.py` / `cli.py` / `ui/` — optional Python retrieve + Gradio
- `solution_architect.py` — capability translation used by the Python path and evals
- `data/` / `eval/` / `tests/` — golden set and harness

## Notes

- Never commit `.env`.
- Generated logs belong under `artifacts/` or `logs/`.
