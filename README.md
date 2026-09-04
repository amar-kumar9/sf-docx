# Salesforce Docs Agent

A local AI agent that answers questions about Salesforce by retrieving and grounding answers against real Salesforce documentation — not model training data.

Every answer is verified against the source. If a claim isn't in the retrieved doc, the agent says so. For questions about things that change between releases — governor limits, heap sizes, API versions — the agent flags the answer as potentially stale and links directly to the relevant Salesforce release notes.

## What it does

- Retrieves relevant Salesforce documentation chunks via MCP
- Synthesizes answers grounded in the retrieved content
- Detects temporally volatile questions (limits, versioned features) and forces a release notes check
- Runs a regression eval harness against a golden dataset

## Layout

- `app.py` - retrieval pipeline, grounding, temporal validation, answer synthesis
- `prompts.py` - prompt templates
- `data/` - golden dataset and sample inputs
- `eval/` - eval harness and MCP fixtures
- `scripts/` - local retrieval diagnostics
- `tests/` - pipeline tests

## Run

```powershell
python app.py
```

## Evaluate

```powershell
python eval/agent_eval.py --compare --baseline-google-mode none --candidate-google-mode hybrid
```

## Diagnose retrieval

```powershell
python scripts/diagnose_retrieval.py
```

## Notes

- Keep `data/golden_dataset.json` versioned.
- Keep `eval/eval_mcp_fixtures.json` versioned.
- Generated logs and eval outputs belong under `artifacts/` or `logs/`.
