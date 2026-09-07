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

You are the agent. Get the live official page. Do not wait for MCP. Do not answer architecture or current limits from memory or from a lagging atlas cheatsheet.

## Triggers

Use this skill when the user:
- Designs or implements a Salesforce capability (classify, flag, show, notify, route)
- Compares platform options (Flow vs Apex, CDC vs Platform Events, MCP vs API)
- Asks Well-Architected, Architecture Center, or Decision Guides
- Needs a current limit, API version, heap size, or what shipped this release

Do not use for org-specific Apex debugging, stack traces, or Agentforce org snapshots.

## Current facts (heap, governors, API version, what’s new)

These change every seasonal release. MCP search ranking `apex_gov_limits` / cheatsheet is **not** enough.

1. Open Help release notes in the **browser** (Help is a JS shell — curl/`WebFetch` returning “Loading” or “CSS Error” is a failed extract, not evidence):

`https://help.salesforce.com/s/articleView?id=release-notes.rn_<topic>.htm&release=<current>&type=5`

Heap example that MCP does not index: [rn_apex_heap_limit, Winter ’27](https://help.salesforce.com/s/articleView?id=release-notes.rn_apex_heap_limit.htm&release=264&type=5)

2. If you do not know the article id, search Help for `Salesforce <topic> release notes` and open the `release-notes.rn_*` hit. Keep going until the article body contains the number.
3. Salesforce Docs MCP is optional extra. If it only returns unversioned `atlas.en-us.apexcode` / cheatsheet, **discard it as the answer**. Quote the Help release-note body.
4. Do not stop because MCP is disconnected. Browser the official URL.

Winter ’27 heap (from that Help article): synchronous **6 MB → 10 MB**, asynchronous **12 MB → 25 MB**. Confirm on the page; do not reuse this paragraph if a later release note supersedes it.

## Design questions

Search capability layers, not the user’s product guess. Prefer Architecture Center. Budget 4. Do not stop until a kept URL is `architect.salesforce.com`. Drop Shield / Event Monitoring / Security Center unless the user asked for org/session security.

Use Salesforce Docs MCP if it has hits. If not, browser `architect.salesforce.com` the same way.

## How to answer

- Current facts: cite the Help release-notes URL. Say if the standing Apex guide still shows the old number.
- Design: open with the capability, not a SKU. Unanswered discovery → **decision deferred**.
- Cite URLs. Label gaps `[Unverified]`.
- Empty or off-topic evidence → refuse. Do not invent limits.

## Examples

**User:** Apex heap size as of 2026?

Browser the Winter ’27 (and later) Help article `rn_apex_heap_limit`. Do not answer 6 MB / 12 MB from Execution Governors if that RN exists.

**User:** Case risk from subject/description, visual indicator for the support rep.

Search classification / Flow vs Apex / record UI — not Shield.

**User:** Can Flow replace an Apex trigger?

Cite Architecture Center.

## Rules

- Do not call `python cli.py`
- Do not treat MCP-miss as “the limit did not change”
- Do not use third-party blogs as the number; they can corroborate after Help
- CRM process language on Case/Account/Opportunity is not platform security
