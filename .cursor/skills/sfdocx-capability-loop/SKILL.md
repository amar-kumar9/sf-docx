---
name: sfdocx-capability-loop
description: >-
  Ground Salesforce design and current-fact answers in live official docs.
  Use when designing a capability, comparing Flow vs Apex, asking
  Well-Architected or Decision Guides, or needing a current API version or
  governor limit.
user-invocable: true
---

# Salesforce capability loop

Name the business capability before any product. Retrieve live Salesforce docs. Do not answer architecture from memory.

## Triggers

Use this skill when the user:
- Designs or implements a Salesforce capability (classify, flag, show, notify, route)
- Compares platform options (Flow vs Apex, CDC vs Platform Events, MCP vs API)
- Asks Well-Architected, Architecture Center, or Decision Guides
- Needs a current limit, API version, or what shipped this release

Do not use for org-specific Apex debugging, stack traces, or Agentforce org snapshots.

## How to retrieve

From the workspace root (try `python3` if `python` is missing):

```bash
python cli.py retrieve --json "<user question>"
```

Parse `evidence[].excerpt`, `evidence[].url`, `solution_plan`, and `host_instructions`. Follow `host_instructions`. You write the answer — do not call `python cli.py ask` unless they asked the Python model to write.

Empty `evidence` means `MCP_URL` is down. Refuse. Do not invent URLs, limits, or API versions.

On design questions the pack must include `architect.salesforce.com` or you must say it is missing. If `avoid_platform_security` is true, do not recommend Shield, Event Monitoring, or Security Center.

## How to answer

- Open with the capability (outcome), not a Salesforce SKU
- Unanswered `discovery_questions` → ADR-style **decision deferred**, not Einstein/Agentforce/Flow
- `separate_detection_from_policy` → an LLM is not the business rule
- Seasonal facts: release notes over current guide over cheatsheets. If only a digest is in the pack, say it may lag
- Cite source URLs. Label gaps `[Unverified]`. Use Trusted/Easy/Adaptable only if those words are in the evidence

## Examples

**User:** implement Case risk detection from subject/description, visual indicator for the support rep.

Search Case Classification, record-triggered Flow vs Apex, Lightning record page — not Shield. If product choice is unknown, defer.

**User:** Can Flow replace an Apex trigger?

Cite Architecture Center. Do not stop on Help-only hits.

**User:** What is the latest Salesforce API version?

Release notes, then the current guide labeled with this release. Not training memory.

**User:** How do I get Live Agent chat queue position?

If the pack is Omni-Channel and not queue position, refuse. Ask for the exact API.

## Rules

- Do not skip retrieve because this machine has no Groq/Ollama key
- Do not search the user's product guess; search the capability layer
- Do not pick a SKU to sound complete
- CRM process language on Case/Account/Opportunity is not platform security
- Fact lookups ("how do I get X") are not capability designs — retrieve the named subject
