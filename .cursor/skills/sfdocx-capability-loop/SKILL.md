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

You are the agent. Search live Salesforce docs with the Salesforce Docs MCP, then write the answer. Do not run Python. Do not read `.env`. Do not answer architecture from memory.

## Triggers

Use this skill when the user:
- Designs or implements a Salesforce capability (classify, flag, show, notify, route)
- Compares platform options (Flow vs Apex, CDC vs Platform Events, MCP vs API)
- Asks Well-Architected, Architecture Center, or Decision Guides
- Needs a current limit, API version, or what shipped this release

Do not use for org-specific Apex debugging, stack traces, or Agentforce org snapshots.

## How to retrieve

Use the **salesforce-docs** MCP (search / fetch tools; names vary, e.g. `salesforce_docs_search`). If that server is disconnected, say so and stop.

On design questions, search capability layers — not the user's product guess. Budget 4. Do not stop until a kept URL is `architect.salesforce.com`:

1. Capability queries (classification, Flow vs Apex, record UI, integration, LDV, …)
2. `Salesforce Well-Architected Framework Trusted Easy Adaptable`
3. Architecture Center decision guides, or integration patterns if the ask is integrate/sync/CDC/ERP/API

Drop Shield / Event Monitoring / Security Center hits unless the user asked for org/session security. Seasonal facts: release notes, then the current guide labeled with this release.

## How to answer

- Open with the capability (outcome), not a Salesforce SKU
- If taxonomy, error-cost, or licensed-product choice is unknown → ADR-style **decision deferred**, not Einstein/Agentforce/Flow
- Keep detection separate from business policy; an LLM is not the business rule
- Cite source URLs. On design questions, Architecture Center must be cited or you must say it was missing
- Label gaps `[Unverified]`. Use Trusted/Easy/Adaptable only if those words are in the evidence
- If search returns nothing on-topic, refuse. Ask for the exact API or object. Do not invent URLs, limits, or versions

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

- Do not call `python cli.py` or `python cli.py retrieve`
- Do not search the user's product guess; search the capability layer
- Do not pick a SKU to sound complete
- CRM process language on Case/Account/Opportunity is not platform security
- Fact lookups ("how do I get X") are not capability designs — search the named subject
