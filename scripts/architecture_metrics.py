#!/usr/bin/env python3
"""Live MCP retrieval A/B for Architecture Center grounding.

Does not import app.py (no LLM required). Measures what the current
single-result MCP search path would see for current planner queries vs
Architecture Center-boosted queries.

Usage:
  python3 scripts/architecture_metrics.py --label before --queries current
  python3 scripts/architecture_metrics.py --label after --queries boosted
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROBES = ROOT / "eval" / "architecture_probes.json"
DEFAULT_MCP_URL = "https://salesforce-docs-76258744c9d7.herokuapp.com/api/mcp"
ARCHITECT_HOST = "architect.salesforce.com"
TEA_TERMS = (
    "well-architected",
    "architecture center",
    "decision guide",
    "integration patterns",
    "trusted, easy",
    "easy, and adaptable",
)
PRODUCT_HOSTS = (
    "developer.salesforce.com",
    "help.salesforce.com",
    "docs.salesforce.com",
)


def _load_probes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Probe file must be a JSON array.")
    return payload


def _code_contracts() -> dict:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    prompts_text = (ROOT / "prompts.py").read_text(encoding="utf-8")
    official_fn = ""
    match = re.search(
        r"def _is_official_salesforce_url\(.*?\n(?:.*\n){0,25}?^def ",
        app_text,
        re.M,
    )
    if match:
        official_fn = match.group(0)
    else:
        official_fn = app_text
    architecture_prompt = ""
    prompt_match = re.search(
        r"ARCHITECTURE_REASONING_PROMPT = \"\"\"(.*?)\"\"\"",
        prompts_text,
        re.S,
    )
    if prompt_match:
        architecture_prompt = prompt_match.group(1)
    grounding_match = re.search(
        r"GROUNDING_CHECK_PROMPT = \"\"\"(.*?)\"\"\"",
        prompts_text,
        re.S,
    )
    grounding = grounding_match.group(1) if grounding_match else ""
    return {
        "architect_host_in_official_url_allowlist": ARCHITECT_HOST in app_text
        and ARCHITECT_HOST in official_fn,
        "tea_in_architecture_prompt": all(
            token in architecture_prompt for token in ("Trusted", "Easy", "Adaptable")
        ),
        "grounding_exempts_general_architecture": "Do not flag general architectural reasoning"
        in grounding,
        "grounding_flags_well_architected_claims": "Well-Architected" in grounding
        and "not in the evidence" in grounding.lower(),
        "architecture_query_boosters_present": "_architecture_search_boosters" in app_text,
        "solution_architect_translation": (ROOT / "solution_architect.py").is_file()
        and "translate_requirement" in app_text,
    }


def _mcp_search(mcp_url: str, query: str, timeout: int = 30) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "salesforce_docs_search",
            "arguments": {"query": query, "limit": 3},
        },
    }
    request = urllib.request.Request(
        mcp_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "chunks": []}

    try:
        envelope = json.loads(raw)
        content = envelope.get("result", envelope)
        if isinstance(content, dict):
            content = content.get("content", content)
        if isinstance(content, list):
            text = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            text = str(content)
        inner = json.loads(text) if text.strip().startswith("{") else {}
        chunks = inner.get("chunks") if isinstance(inner, dict) else []
        if not isinstance(chunks, list):
            chunks = []
        return {"ok": True, "error": None, "chunks": chunks}
    except Exception as exc:
        return {"ok": False, "error": f"parse:{exc}", "chunks": []}


def _chunk_url(chunk: dict) -> str:
    return str(chunk.get("url") or "").strip()


def _is_architect_url(url: str) -> bool:
    return ARCHITECT_HOST in url.lower()


def _is_product_url(url: str) -> bool:
    lower = url.lower()
    return any(host in lower for host in PRODUCT_HOSTS) and not _is_architect_url(url)


def _tea_hits(text: str) -> list[str]:
    lower = text.lower()
    return [term for term in TEA_TERMS if term in lower]


def _evaluate_probe(
    probe: dict,
    queries: list[str],
    mcp_url: str,
    search_budget: int,
    stop_after_hits: int | None,
    require_architect: bool,
) -> dict:
    evidence = []
    searches = 0
    stopped_early = False
    for query in queries[:search_budget]:
        searches += 1
        result = _mcp_search(mcp_url, query)
        chunks = result.get("chunks") or []
        primary = chunks[0] if chunks else {}
        excerpt = "\n".join(
            str(chunk.get("content") or "") for chunk in chunks if isinstance(chunk, dict)
        )
        urls = [_chunk_url(chunk) for chunk in chunks if isinstance(chunk, dict)]
        item = {
            "query": query,
            "ok": bool(result.get("ok")),
            "primary_url": _chunk_url(primary) if isinstance(primary, dict) else "",
            "chunk_urls": [url for url in urls if url],
            "excerpt_head": excerpt[:400],
            "tea_terms": _tea_hits(excerpt),
            "error": result.get("error"),
        }
        evidence.append(item)
        hits = [row for row in evidence if row.get("primary_url")]
        has_architect = any(_is_architect_url(row["primary_url"]) for row in hits)
        if require_architect and not has_architect:
            continue
        if stop_after_hits and len(hits) >= stop_after_hits:
            stopped_early = searches < min(len(queries), search_budget)
            break

    primary_urls = [row["primary_url"] for row in evidence if row.get("primary_url")]
    architect_primary = [url for url in primary_urls if _is_architect_url(url)]
    product_primary = [url for url in primary_urls if _is_product_url(url)]
    all_chunk_urls = [url for row in evidence for url in row.get("chunk_urls", [])]
    tea_terms = sorted({term for row in evidence for term in row.get("tea_terms", [])})
    return {
        "id": probe["id"],
        "group": probe.get("group"),
        "question": probe["question"],
        "searches": searches,
        "stopped_early": stopped_early,
        "queries": queries[:search_budget],
        "query_has_architecture_center_terms": any(
            _tea_hits(query) for query in queries[:search_budget]
        ),
        "evidence_count": len(primary_urls),
        "architect_primary_count": len(architect_primary),
        "product_primary_count": len(product_primary),
        "architect_chunk_count": sum(1 for url in all_chunk_urls if _is_architect_url(url)),
        "has_architect_primary": bool(architect_primary),
        "has_architect_chunk": any(_is_architect_url(url) for url in all_chunk_urls),
        "tea_term_count": len(tea_terms),
        "tea_terms": tea_terms,
        "primary_urls": primary_urls,
        "evidence": evidence,
    }


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rate(flags: list[bool]) -> float:
    return _mean([1.0 if flag else 0.0 for flag in flags])


def _summarize(rows: list[dict], contracts: dict, label: str, query_mode: str) -> dict:
    architecture = [row for row in rows if row.get("group") == "architecture"]
    control = [row for row in rows if row.get("group") == "control"]
    return {
        "label": label,
        "query_mode": query_mode,
        "architecture_questions": len(architecture),
        "architect_center_coverage": _rate(
            [row["has_architect_primary"] for row in architecture]
        ),
        "architect_chunk_coverage": _rate(
            [row["has_architect_chunk"] for row in architecture]
        ),
        "mean_architect_primary_urls": _mean(
            [row["architect_primary_count"] for row in architecture]
        ),
        "mean_product_primary_urls": _mean(
            [row["product_primary_count"] for row in architecture]
        ),
        "mean_searches": _mean([row["searches"] for row in architecture]),
        "query_boost_rate": _rate(
            [row["query_has_architecture_center_terms"] for row in architecture]
        ),
        "tea_excerpt_hit_rate": _rate(
            [row["tea_term_count"] > 0 for row in architecture]
        ),
        "mean_tea_terms": _mean([row["tea_term_count"] for row in architecture]),
        "control_architect_primary": sum(
            row["architect_primary_count"] for row in control
        ),
        "control_query_boost": _rate(
            [row["query_has_architecture_center_terms"] for row in control]
        ),
        "code_contracts": contracts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Architecture Center retrieval metrics")
    parser.add_argument("--label", required=True, help="Snapshot label: before|after")
    parser.add_argument(
        "--queries",
        choices=("current", "boosted"),
        required=True,
        help="Which query list to execute from the probe file.",
    )
    parser.add_argument("--probes", default=str(DEFAULT_PROBES))
    parser.add_argument("--mcp-url", default=DEFAULT_MCP_URL)
    parser.add_argument(
        "--output",
        default="",
        help="JSON output path. Default: artifacts/eval/architecture_metrics_<label>.json",
    )
    args = parser.parse_args()

    probes = _load_probes(Path(args.probes))
    query_key = "current_queries" if args.queries == "current" else "boosted_queries"
    if args.queries == "current":
        search_budget = 3
        stop_after_hits = 2
        require_architect = False
    else:
        search_budget = 4
        stop_after_hits = None
        require_architect = True

    rows = []
    for probe in probes:
        queries = probe.get(query_key) or []
        budget = 1 if probe.get("group") == "control" else search_budget
        stop = 1 if probe.get("group") == "control" else stop_after_hits
        require = False if probe.get("group") == "control" else require_architect
        rows.append(
            _evaluate_probe(
                probe,
                queries,
                args.mcp_url,
                search_budget=budget,
                stop_after_hits=stop,
                require_architect=require,
            )
        )

    contracts = _code_contracts()
    summary = _summarize(rows, contracts, args.label, args.queries)
    payload = {"summary": summary, "rows": rows}

    output = Path(args.output) if args.output else ROOT / "artifacts" / "eval" / f"architecture_metrics_{args.label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
