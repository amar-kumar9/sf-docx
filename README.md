# Salesforce QA Agent

Local Salesforce documentation Q&A agent with retrieval, synthesis, grounding, and eval support.

## Layout

- `app.py` - runtime app and retrieval pipeline
- `prompts.py` - prompt templates and routing instructions
- `data/` - labeled datasets and sample inputs
- `eval/` - benchmark harness and MCP fixtures
- `scripts/` - local diagnostic helpers
- `tests/` - focused pipeline tests
- `artifacts/` - generated eval outputs and reports

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
