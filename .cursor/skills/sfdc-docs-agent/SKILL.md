---
name: sfdc-docs-agent
description: >-
  Grounds Salesforce architecture and documentation answers in live Salesforce
  docs via this repo's Python agent. Use when the user asks about Salesforce,
  Apex, Flow, Agentforce, Well-Architected, Decision Guides, governor limits,
  API versions, or integration patterns. Runs `python cli.py retrieve --json`
  first, then synthesizes from that evidence. Use even when no Groq/Ollama key
  is configured on the machine.
---

# Salesforce docs + local agent

This repo's Python agent owns retrieval, Architecture Center routing, and grounding. You (the Cursor model) own synthesis when no cloud/local LLM is configured for Python.

## Every Salesforce question

1. From the workspace root, run:

```bash
python cli.py retrieve --json "<user question>"
```

If `python` is missing, try `python3`.

2. Parse the JSON. Use `evidence[].excerpt`, `evidence[].url`, and `intent`.
3. Follow `host_instructions` from the JSON.
4. Synthesize the answer yourself. Do **not** skip retrieval because this machine has no Groq or Ollama key.
5. If `evidence` is empty, say so and point the user at `python cli.py status`.
6. If the JSON includes `solution_plan`, treat that as the architecture intake:
   - `business_capability` is the thing you are designing (outcome, not a product).
   - `technical_requirements` are the searchable Salesforce layers.
   - `discovery_questions` must stay visible. If they are unanswered, write an
     ADR-style **decision deferred** instead of picking Einstein/Agentforce/Flow.
   - When `full_protocol` is true, follow capability → taxonomy/decisions →
     detection vs policy → data → options → domain → UX → scope. Do not open
     with a Salesforce feature.
   - When `separate_detection_from_policy` is true, do not let an LLM be the
     business decision.
   - Do not search or recommend org-security products when
     `avoid_platform_security` is true.

7. For current limits, API version, or “what’s new this release”: Salesforce ships three
   seasonal releases a year. Treat docs like case law:
   - **Release notes** are the court order (what just changed).
   - **Current developer guides** are the statute (the restated law, labeled with the release).
   - **Cheatsheets / older help articles** are a digest and often still show the previous ruling.
   If they conflict, prefer the notes. If the pack only has a digest, do not treat that
   number as current — say so and point at the notes.

## Do not

- Invent Salesforce URLs, limits, or API versions.
- Answer architecture questions from memory when the pack has Architecture Center pages.
- Call `python cli.py ask` unless `python cli.py status` shows an **Active LLM**. `ask` is the standalone Python synthesizer; you are the synthesizer when it is `none`.

## Optional

If status shows an active LLM and the user wants the Python model to write the answer, run `python cli.py ask "<question>"` and return that output.
