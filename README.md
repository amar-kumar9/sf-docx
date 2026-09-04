# Salesforce QA Agent

This repo contains a local Salesforce documentation Q&A agent with:

- `app.py` - the main Gradio app and retrieval pipeline
- `prompts.py` - prompt templates and routing instructions
- `agent_eval.py` - session-level eval harness and regression gate
- `golden_dataset.json` - labeled evaluation cases
- `tests/` - focused pipeline tests

## Run

```powershell
python app.py
```

## Evaluate

```powershell
python agent_eval.py --compare --baseline-google-mode none --candidate-google-mode hybrid
```

## Notes

- Keep `golden_dataset.json` versioned.
- Generated logs, caches, and eval artifacts are ignored by git.
- The repo is intentionally flat so the main entry points are easy to find.
