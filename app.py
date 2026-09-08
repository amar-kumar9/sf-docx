import os
import json
import re
import time
from contextvars import ContextVar
import requests
import tiktoken
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from providers import get_llm
import claim_risk
from prompts import (
    INTENT_CLASSIFIER_PROMPT,
    QUERY_PLANNER_PROMPT,
    TEMPORAL_FACT_VALIDATION_PROMPT,
    EVIDENCE_CONFLICT_RESOLUTION_PROMPT,
    CURRENT_FACT_DIRECT_ANSWER_PROMPT,
    DEFINITION_FIRST_SYNTHESIS_PROMPT,
    SOQL_101_HINT_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    ARCHITECTURE_REASONING_PROMPT,
    CODE_REVIEW_PROMPT,
    GROUNDING_CHECK_PROMPT,
    GROUNDING_REWRITE_PROMPT,
    EVIDENCE_CRITIC_PROMPT,
    SOLUTION_ARCHITECT_PROMPT,
    SOLUTION_ARCHITECTURE_PROTOCOL,
    SOLUTION_ARCHITECTURE_PROTOCOL_SHORT,
)
import solution_architect as sa

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

os.makedirs("logs", exist_ok=True)
logger.add("logs/app.log", rotation="10 MB", retention="7 days", level="DEBUG")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MCP_URL        = os.getenv("MCP_URL",        "https://salesforce-docs-76258744c9d7.herokuapp.com/api/mcp")
MCP_TIMEOUT    = int(os.getenv("MCP_TIMEOUT",    "30"))
MCP_MAX_TOKENS = int(os.getenv("MCP_MAX_TOKENS", "1500"))
# For intents where truncating an authoritative value (a limit number, a version,
# a definition) is worse than a wider context window, retrieval widens the excerpt
# cap. Scoped per-retrieval via a ContextVar so it never bloats unrelated packs.
MCP_MAX_TOKENS_GENEROUS = int(os.getenv("MCP_MAX_TOKENS_GENEROUS", str(MCP_MAX_TOKENS * 2)))
_MCP_TOKEN_BUDGET: ContextVar[int] = ContextVar("mcp_token_budget", default=MCP_MAX_TOKENS)


def _effective_mcp_max_tokens() -> int:
    return _MCP_TOKEN_BUDGET.get()
GROQ_MODEL     = os.getenv("GROQ_MODEL",     "openai/gpt-oss-20b")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "qwen2.5-coder:7b")
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", "gemma3:1b")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434/")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX", "")
GOOGLE_SEARCH_LIMIT = int(os.getenv("GOOGLE_SEARCH_LIMIT", "3"))
GOOGLE_FALLBACK_ENABLED = os.getenv("GOOGLE_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
GOOGLE_SEARCH_MODE = os.getenv("GOOGLE_SEARCH_MODE", "hybrid").lower()
DEBUG_FULL_EXCERPTS = os.getenv("DEBUG_FULL_EXCERPTS", "false").lower() in {"1", "true", "yes", "on"}

SEARCH_BUDGET  = {"quick": 1, "standard": 2, "deep": 3}
FETCH_BUDGET   = {"quick": 0, "standard": 1, "deep": 1}
ARCHITECTURE_SEARCH_BUDGET = 4
ARCHITECTURE_RESCUE_BUDGET = 2
ARCHITECTURE_SECOND_PASS_BUDGET = 2
ARCHITECTURE_FETCH_BUDGET = 2
# Definition questions always fetch the top doc regardless of depth tier.
FETCH_BUDGET_DEFINITION = 1
# Court-order (release notes) + statute (current guide); cheatsheet is a digest.
SEASONAL_SEARCH_BUDGET = 4
SEASONAL_FETCH_BUDGET = 2
_GENERIC_RELEASE_NOTES_QUERIES = {
    "salesforce release notes current release",
}
_ATLAS_DOC_URL_RE = re.compile(
    r"https://developer\.salesforce\.com/docs/atlas\.en-us\.[^/\s\]]+\.meta/([^/\s\]]+)/([^\s\)\]]+)",
    re.IGNORECASE,
)
_MARKDOWN_DOC_LINK_RE = re.compile(
    r"\[([^\]]+)\]\((https://developer\.salesforce\.com/docs/[^)]+)\)",
    re.IGNORECASE,
)
MCP_FAILURE_CACHE_SECONDS = int(os.getenv("MCP_FAILURE_CACHE_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Per-session cost tracking (per AI Engineer Guide Chapter 4B)
# Cost is the FIRST constraint; quality is maximised subject to it.
# Groq free tier = $0; Ollama local = $0. Values here are placeholders
# so the telemetry shape is correct when paid tiers are used.
# ---------------------------------------------------------------------------
_SESSION_COST: dict[str, float] = {}   # thread_id -> cumulative cents

GROQ_COST_PER_1K_TOKENS  = float(os.getenv("GROQ_COST_PER_1K_TOKENS",  "0.0"))
OLLAMA_COST_PER_1K_TOKENS = float(os.getenv("OLLAMA_COST_PER_1K_TOKENS", "0.0"))


def _charge_cost(thread_id: str, tokens: int, provider: str) -> float:
    """Record token cost for a session. Returns cost in cents charged."""
    rate = GROQ_COST_PER_1K_TOKENS if "Groq" in provider else OLLAMA_COST_PER_1K_TOKENS
    cost = (tokens / 1000) * rate
    _SESSION_COST[thread_id] = _SESSION_COST.get(thread_id, 0.0) + cost
    return cost


def _session_cost(thread_id: str) -> float:
    return _SESSION_COST.get(thread_id, 0.0)

# Legacy topic patterns — kept as a thin alias so older search-boost paths and
# tests still recognize classic limit wording. Prefer claim_risk for gating.
_VOLATILE_LIMIT_PATTERNS = re.compile(
    r"\b(heap\s*size|governor\s*limit|platform\s*limit|apex\s*limit|soql\s*limit|"
    r"dml\s*limit|cpu\s*limit|callout\s*limit|api\s*version|api\s*limit|"
    r"bulk\s*api\s*limit|max\s*query|query\s*limit|heap\s*limit)\b",
    re.IGNORECASE,
)


def _is_volatile_limit_question(message: str) -> bool:
    """True when the question asserts a mutable Salesforce platform claim shape.

    Uses claim-risk heuristics (limits, quotas, versions, availability) rather
    than only a hard-coded topic list. Legacy pattern match remains as fallback.
    """
    if claim_risk.question_suggests_mutable_platform_fact(message):
        return True
    return bool(_VOLATILE_LIMIT_PATTERNS.search(message))


TEMPORAL_KEYWORDS = (
    "current", "latest", "today", "now", "this release", "current release",
    "release notes", "deprec", "retir", "seasonal release", "api version",
    "governor limit", "heap size", "feature availability",
    "security behavior", "integration behavior", "data cloud",
    "platform limit",
)
HISTORICAL_RELEASE_RE = re.compile(
    r"\b(?:spring|summer|winter|fall)\s*['’]?\d{2}\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# MCP Layer — deterministic, no @tool decorator
# ---------------------------------------------------------------------------

_MCP_RETRY_AFTER = 0.0
_REPORTED_LLM_FAILURES: set[str] = set()
_REPORTED_INVOKE_FAILURES: set[str] = set()


def _smart_truncate(text: str, max_tokens: int) -> str:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        logger.warning(f"MCP truncated {len(tokens)} → {max_tokens} tokens")
        truncated = enc.decode(tokens[:max_tokens])
        last_nl = truncated.rfind("\n")
        if last_nl > max_tokens * 3 // 4:
            truncated = truncated[:last_nl]
        return truncated + "\n\n[Content truncated — fetch source document for full details.]"
    except Exception as e:
        logger.error(f"Tokenizer error: {e}")
        return text[:4000]


def _call_mcp(tool_name: str, arguments: dict) -> tuple[str, bool]:
    global _MCP_RETRY_AFTER
    now = time.time()
    if now < _MCP_RETRY_AFTER:
        return "", False

    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    try:
        resp = requests.post(MCP_URL, json=payload, headers=headers, timeout=MCP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("result", data)
        if isinstance(content, dict):
            content = content.get("content", content)
        if isinstance(content, list):
            raw = "\n".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            raw = str(content)
        if not raw.strip():
            return "", False
        return raw, True
    except requests.exceptions.Timeout:
        logger.error(f"MCP timeout | {tool_name}")
        _MCP_RETRY_AFTER = time.time() + MCP_FAILURE_CACHE_SECONDS
        return "", False
    except requests.exceptions.HTTPError as e:
        logger.error(f"MCP HTTP error | {tool_name} | {e}")
        _MCP_RETRY_AFTER = time.time() + MCP_FAILURE_CACHE_SECONDS
        return "", False
    except Exception as e:
        logger.warning(f"MCP unavailable | {tool_name} | {e}")
        _MCP_RETRY_AFTER = time.time() + MCP_FAILURE_CACHE_SECONDS
        return "", False


def _chunk_document_identity(chunk: dict) -> tuple[str, str, str]:
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    path = str(
        chunk.get("documentPath")
        or chunk.get("document_path")
        or meta.get("documentPath")
        or meta.get("document_path")
        or ""
    ).strip()
    url = str(chunk.get("url") or chunk.get("link") or meta.get("url") or "").strip()
    title = str(
        chunk.get("title") or chunk.get("name") or meta.get("title") or path or ""
    ).strip()
    return path, url, title


def _parse_mcp_search_documents(raw: str) -> list[dict]:
    """One evidence item per document path. Search chunks often mix docs."""
    fallback = {
        "title": "",
        "url": "",
        "document_path": "",
        "excerpt": raw,
    }
    try:
        data = json.loads(raw)
    except Exception:
        return [fallback] if str(raw or "").strip() else []

    if not isinstance(data, dict):
        return [fallback] if str(raw or "").strip() else []

    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return [fallback] if str(raw or "").strip() else []

    grouped: dict[str, dict] = {}
    order: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        path, url, title = _chunk_document_identity(chunk)
        key = path or url or title or f"chunk-{len(grouped)}"
        if key not in grouped:
            grouped[key] = {
                "title": title,
                "url": url,
                "document_path": path,
                "contents": [],
            }
            order.append(key)
        rec = grouped[key]
        if title and not rec["title"]:
            rec["title"] = title
        if url and not rec["url"]:
            rec["url"] = url
        if path and not rec["document_path"]:
            rec["document_path"] = path
        content = chunk.get("content")
        if content:
            rec["contents"].append(str(content))

    docs = []
    for key in order:
        rec = grouped[key]
        excerpt = _smart_truncate("\n\n".join(rec["contents"]) if rec["contents"] else raw, _effective_mcp_max_tokens())
        if not excerpt.strip():
            continue
        docs.append(
            {
                "title": rec["title"] or rec["document_path"],
                "url": rec["url"],
                "document_path": rec["document_path"],
                "excerpt": excerpt,
            }
        )
    return docs or [fallback]


def _parse_mcp_document_payload(raw: str) -> dict:
    docs = _parse_mcp_search_documents(raw)
    return docs[0] if docs else {
        "title": "",
        "url": "",
        "document_path": "",
        "excerpt": raw,
    }


def _parse_mcp_fetch_payload(raw: str) -> dict:
    """Parse the flat fetch response: {id, documentPath, url, content, ...}."""
    parsed = {"title": "", "url": "", "document_path": "", "excerpt": raw}
    try:
        data = json.loads(raw)
    except Exception:
        return _parse_mcp_document_payload(raw)
    if not isinstance(data, dict):
        return _parse_mcp_document_payload(raw)
    content = data.get("content")
    if content:
        parsed["excerpt"] = _smart_truncate(str(content), _effective_mcp_max_tokens())
        parsed["document_path"] = str(data.get("documentPath") or data.get("id") or "").strip()
        parsed["url"] = str(data.get("url") or "").strip()
        parsed["title"] = parsed["document_path"]
        return parsed
    return _parse_mcp_document_payload(raw)


def _extract_json_payload(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return json.loads(raw)


class _LLMResponse:
    def __init__(self, content: str):
        self.content = content


def _prompt_kind(system_prompt: str) -> str:
    if INTENT_CLASSIFIER_PROMPT in system_prompt:
        return "intent"
    if TEMPORAL_FACT_VALIDATION_PROMPT in system_prompt:
        return "temporal"
    if QUERY_PLANNER_PROMPT in system_prompt:
        return "query"
    if SOLUTION_ARCHITECT_PROMPT in system_prompt:
        return "solution"
    if EVIDENCE_CONFLICT_RESOLUTION_PROMPT in system_prompt:
        return "conflict"
    if GROUNDING_CHECK_PROMPT in system_prompt:
        return "grounding"
    if GROUNDING_REWRITE_PROMPT in system_prompt:
        return "grounding_rewrite"
    return "synthesis"


def _safe_fallback_content(kind: str, human_prompt: str) -> str:
    if kind == "intent":
        return json.dumps(_SAFE_FALLBACK_INTENT)
    if kind == "temporal":
        return json.dumps(_determine_temporal_context(human_prompt, {}))
    if kind == "query":
        return "[]"
    if kind == "solution":
        return json.dumps(
            {
                "business_capability": "",
                "objects": [],
                "capabilities": [],
                "technical_requirements": [],
                "search_queries": [],
                "avoid_products": [],
                "separate_detection_from_policy": False,
                "discovery_questions": [],
            }
        )
    if kind == "conflict":
        return json.dumps({
            "status": "no_conflict",
            "applicable_fact": None,
            "conflicts": [],
        })
    if kind == "critic":
        return json.dumps({"aligned": True, "reason": None, "rescue_queries": []})
    if kind == "grounding":
        # Fail closed: if the grounding checker is unavailable we must NOT pass
        # the answer as verified. This sentinel maps to an unverified status.
        return "GROUNDING: unverified"
    if kind == "grounding_rewrite":
        return (
            "## Answer\n"
            "[Unverified] I could not fully verify the answer against the retrieved evidence."
        )
    if "Current docs required: True" in human_prompt or "Question type: current_fact" in human_prompt:
        return (
            "## Answer\n"
            "[Unverified] I could not confirm the current Salesforce value from the retrieved evidence."
        )
    return (
        "## Answer\n"
        "[Unverified] I could not generate a verified answer because the local model provider was unavailable."
    )


def _is_fallback_content(content: str) -> bool:
    return (
        "[Unverified]" in content
        and "local model provider was unavailable" in content
    )


class ResilientLLM:
    def __init__(self, llm, label: str):
        self._llm = llm
        self._label = label

    def invoke(self, messages):
        try:
            return self._llm.invoke(messages)
        except Exception as e:
            system_prompt = messages[0].content if messages else ""
            human_prompt = messages[1].content if len(messages) > 1 else ""
            kind = _prompt_kind(system_prompt)
            failure_key = f"{self._label}:{kind}"
            if failure_key not in _REPORTED_LLM_FAILURES:
                logger.warning(f"LLM unavailable | provider={self._label} | kind={kind} | {e}")
                _REPORTED_LLM_FAILURES.add(failure_key)
            fallback_content = _safe_fallback_content(kind, human_prompt)
            logger.warning(f"FallbackContentUsed | provider={self._label} | kind={kind} | reason=llm_unavailable")
            return _LLMResponse(fallback_content)

    def __getattr__(self, item):
        return getattr(self._llm, item)


def _invoke_resilient(llm, messages, kind: str, human_prompt: str = ""):
    try:
        return llm.invoke(messages)
    except Exception as e:
        if kind not in _REPORTED_INVOKE_FAILURES:
            logger.warning(f"LLM unavailable | kind={kind} | {e}")
            _REPORTED_INVOKE_FAILURES.add(kind)
        fallback_content = _safe_fallback_content(kind, human_prompt)
        logger.warning(f"FallbackContentUsed | kind={kind} | reason=invoke_exception")
        return _LLMResponse(fallback_content)


def _find_historical_release(message: str) -> str | None:
    match = re.search(r'\b(?:spring|summer|winter|fall)\s*[\'"]?\d{2}\b', message, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _determine_temporal_context(message: str, intent_data: dict) -> dict:
    lower = message.lower()
    requested_release = None
    historical_release = None
    definition_question = _is_definition_question(message, intent_data)

    historical_release = _find_historical_release(message)
    if historical_release:
        requested_release = historical_release

    explicit_current_scope = "current" in lower or "latest" in lower
    if explicit_current_scope:
        temporal_validation_required = True
    else:
        temporal_validation_required = bool(
            intent_data.get("requires_current_docs")
            or historical_release
            or any(keyword in lower for keyword in TEMPORAL_KEYWORDS)
        )

    current_docs_required = bool(
        intent_data.get("requires_current_docs")
        or (temporal_validation_required and not historical_release)
    )

    if definition_question and not explicit_current_scope and not historical_release:
        temporal_validation_required = False
        current_docs_required = False

    return {
        "temporal_validation_required": temporal_validation_required,
        "current_docs_required": current_docs_required,
        "historical_release_requested": bool(historical_release),
        "requested_release": requested_release,
    }


def _temporal_search_boosters(message: str, temporal: dict) -> list[str]:
    lower = message.lower()
    boosters = []

    if temporal.get("historical_release_requested") and temporal.get("requested_release"):
        boosters.extend([
            f'Salesforce release notes {temporal["requested_release"]}',
            f'Salesforce developer documentation {temporal["requested_release"]}',
        ])
        return boosters

    if not temporal.get("temporal_validation_required"):
        return boosters

    boosters.append("Salesforce Release Notes current release")
    boosters.append("Salesforce Developer Documentation current release")

    if "api version" in lower:
        boosters.extend([
            "Salesforce API version current release notes",
            "Salesforce Developer Documentation API version",
        ])
    if "heap" in lower or "limit" in lower or "governor" in lower:
        boosters.extend([
            "Salesforce Apex heap size limit release notes Winter Spring Summer",
            "Salesforce governor limits release notes current",
            "Salesforce Apex limits developer documentation",
        ])

    return boosters


def _needs_seasonal_authority(
    message: str,
    intent_data: dict | None = None,
    temporal: dict | None = None,
) -> bool:
    """True when standing docs can lag a Salesforce seasonal release."""
    temporal = temporal or {}
    if temporal.get("historical_release_requested") or temporal.get("current_docs_required"):
        return True
    if _is_volatile_limit_question(message):
        return True
    if claim_risk.question_suggests_mutable_platform_fact(message):
        return True
    intent = str((intent_data or {}).get("intent") or "")
    return intent in {"limits", "release"}


def _is_release_notes_evidence(item: dict) -> bool:
    """Identity of the document, not a passing mention in a digest body."""
    if _is_non_authoritative_salesforce_evidence(item):
        return False
    metadata = _authority_metadata_blob(item)
    return bool(
        re.search(
            r"release[-_\s]?notes|\brn_[a-z]|salesforce_release_notes",
            metadata,
            re.IGNORECASE,
        )
    )


def _is_limits_cheatsheet(item: dict) -> bool:
    blob = _evidence_blob(item)
    return "cheatsheet" in blob or "salesforce_app_limits" in blob


def _is_generic_release_notes_query(query: str) -> bool:
    return str(query or "").strip().lower() in _GENERIC_RELEASE_NOTES_QUERIES


def _merge_temporal_queries(
    queries: list[str],
    message: str,
    intent_data: dict,
    temporal: dict,
    max_q: int,
) -> list[str]:
    """Court order first, then the question, then the standing statute.

    Do not spend the whole search budget on a generic release-notes hub query.
    The hub is the index; the ruling is the topic-specific note. Cheatsheets
    are a digest and are not a search target of their own.
    """
    if not _needs_seasonal_authority(message, intent_data, temporal):
        return queries[:max_q]
    boosters = _temporal_search_boosters(message, temporal)
    merged: list[str] = []

    def _add(query: str) -> None:
        text = str(query or "").strip()
        if text and text not in merged:
            merged.append(text)

    topic_rn = [
        booster
        for booster in boosters
        if "release note" in booster.lower() and not _is_generic_release_notes_query(booster)
    ]
    generic_rn = [booster for booster in boosters if _is_generic_release_notes_query(booster)]
    statute = [
        booster
        for booster in boosters
        if "release note" not in booster.lower()
    ]
    if not topic_rn:
        compact = " ".join(
            word
            for word in re.findall(r"[A-Za-z0-9']+", message)
            if word.lower() not in {"what", "whats", "the", "is", "a", "an"}
        ).strip()
        topic_rn = [f"Salesforce {compact or message} release notes"]

    for booster in topic_rn[:1]:
        _add(booster)
    for query in queries:
        if _is_generic_release_notes_query(query):
            continue
        if "release note" in query.lower():
            continue
        _add(query)
    for booster in statute:
        _add(booster)
    for booster in topic_rn[1:]:
        _add(booster)
    for booster in generic_rn:
        _add(booster)
    return merged[:max_q]


def _requires_court_order(message: str, intent_data: dict | None = None) -> bool:
    if _is_volatile_limit_question(message):
        return True
    return str((intent_data or {}).get("intent") or "") in {"limits", "release"}


def _court_order_covers_question(item: dict, message: str) -> bool:
    """True when release notes actually rule on this question, not just the hub."""
    if not _is_release_notes_evidence(item):
        return False
    blob = _evidence_blob(item)
    matches = list(_VOLATILE_LIMIT_PATTERNS.finditer(message))
    if matches:
        for match in matches:
            token = match.group(0).lower()
            if token in blob:
                return True
            head = token.split()[0]
            if len(head) >= 4 and head in blob:
                return True
        return False
    terms = _topic_terms(message, {"intent": "release"})
    return any(term in blob for term in terms if len(term) >= 4)


def _statute_on_topic(item: dict, message: str) -> bool:
    """True when a standing guide is about this limit/feature, not a TOC or changelog."""
    meta = _authority_metadata_blob(item)
    matches = list(_VOLATILE_LIMIT_PATTERNS.finditer(message))
    if matches:
        for match in matches:
            token = match.group(0).lower()
            head = token.split()[0]
            if token in meta or (len(head) >= 4 and head in meta):
                return True
        lower = message.lower()
        if any(key in lower for key in ("heap", "governor", "soql", "dml", "cpu", "callout")):
            if "governor" in meta and "limit" in meta:
                return True
            if "gov_limits" in meta:
                return True
        return False
    terms = _topic_terms(message, {"intent": "release"})
    return any(term in meta for term in terms if len(term) >= 4)


def _is_standing_statute(item: dict) -> bool:
    """Current developer/help guide — the law, not the seasonal ruling or a digest."""
    if _is_limits_cheatsheet(item) or _is_release_notes_evidence(item):
        return False
    return _is_authoritative_salesforce_evidence(item)


def _seasonal_authority_rank(item: dict, message: str) -> int:
    if _court_order_covers_question(item, message):
        return 4
    if _is_standing_statute(item) and _statute_on_topic(item, message):
        return 3
    if _is_standing_statute(item):
        return 2
    if _is_release_notes_evidence(item):
        return 1
    return 0


def _mcp_path_from_atlas_url(url: str) -> str | None:
    match = _ATLAS_DOC_URL_RE.search(url or "")
    if not match:
        return None
    bundle, page = match.group(1), match.group(2)
    page = page.split("?")[0].split("#")[0].rstrip(".,;")
    if "cheatsheet" in bundle.lower() or "cheatsheet" in page.lower():
        return None
    if page.endswith(".htm"):
        page = page[:-4] + ".html"
    elif not page.endswith(".html"):
        page = f"{page}.html"
    return f"{bundle}/{page}"


def _cited_standing_guides(excerpt: str) -> list[dict]:
    """Follow digest citations to the standing guide (citator, not a product list)."""
    found: list[dict] = []
    seen: set[str] = set()
    for title, url in _MARKDOWN_DOC_LINK_RE.findall(excerpt or ""):
        if "cheatsheet" in url.lower() or "release-notes" in url.lower():
            continue
        key = url.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(
            {
                "title": title.strip(),
                "url": url.strip(),
                "document_path": _mcp_path_from_atlas_url(url),
            }
        )
    if not found:
        for match in _ATLAS_DOC_URL_RE.finditer(excerpt or ""):
            url = match.group(0)
            path = _mcp_path_from_atlas_url(url)
            if not path or url.lower() in seen:
                continue
            seen.add(url.lower())
            found.append({"title": "", "url": url, "document_path": path})
    return found


def validate_temporal_context(message: str, intent_data: dict, llm) -> dict:
    fallback = _determine_temporal_context(message, intent_data)
    definition_question = _is_definition_question(message, intent_data)
    lower = message.lower()
    explicit_current_scope = "current" in lower or "latest" in lower
    historical_release = _find_historical_release(message)
    if llm is None:
        merged = dict(fallback)
        merged["reason"] = "heuristic"
        if definition_question and not explicit_current_scope and not historical_release:
            merged["temporal_validation_required"] = False
            merged["current_docs_required"] = False
        if _is_volatile_limit_question(message) and not historical_release:
            merged["temporal_validation_required"] = True
            merged["current_docs_required"] = True
        return merged
    try:
        resp = _invoke_resilient(llm, [
            SystemMessage(content=TEMPORAL_FACT_VALIDATION_PROMPT),
            HumanMessage(content=message),
        ], "temporal", message)
        parsed = _extract_json_payload(resp.content)
        merged = {
            "temporal_validation_required": bool(
                parsed.get("temporal_validation_required", fallback["temporal_validation_required"])
            ),
            "current_docs_required": bool(
                parsed.get("current_docs_required", fallback["current_docs_required"])
            ),
            "historical_release_requested": bool(
                parsed.get("historical_release_requested", fallback["historical_release_requested"])
            ),
            "requested_release": parsed.get("requested_release") or fallback["requested_release"],
            "reason": parsed.get("reason", ""),
        }
        if merged["historical_release_requested"]:
            merged["current_docs_required"] = False
        if definition_question and not explicit_current_scope and not historical_release:
            merged["temporal_validation_required"] = False
            merged["current_docs_required"] = False
        # Deterministic override: volatile limits always require release notes.
        # LLM classifiers routinely miss this because static docs exist for these topics.
        if _is_volatile_limit_question(message) and not historical_release:
            merged["temporal_validation_required"] = True
            merged["current_docs_required"] = True
            logger.info("TemporalOverride=volatile_limit | forcing current_docs_required=True")
        return merged
    except Exception as e:
        logger.warning(f"Temporal validation failed: {e} - using fallback")
        if definition_question and not explicit_current_scope and not historical_release:
            fallback["temporal_validation_required"] = False
            fallback["current_docs_required"] = False
        if _is_volatile_limit_question(message) and not historical_release:
            fallback["temporal_validation_required"] = True
            fallback["current_docs_required"] = True
        return fallback


def _format_search_hit(query: str, parsed: dict) -> dict:
    return {
        "query": query,
        "excerpt": parsed.get("excerpt") or "",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": parsed.get("title") or "",
        "url": parsed.get("url") or "",
        "document_path": parsed.get("document_path") or "",
        "relevance": None,
    }


def mcp_search(query: str) -> dict | list[dict]:
    logger.info(f"Search | query='{query}'")
    raw, ok = _call_mcp("salesforce_docs_search", {"query": query, "limit": 3})
    if not ok or not raw:
        logger.warning(f"Search empty | query='{query}'")
        return {}
    hits = [
        _format_search_hit(query, parsed)
        for parsed in _parse_mcp_search_documents(raw)
        if parsed.get("excerpt")
    ]
    if not hits:
        return {}
    return hits if len(hits) > 1 else hits[0]


def _google_available() -> bool:
    return bool(GOOGLE_FALLBACK_ENABLED and GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX)


def _is_official_salesforce_url(url: str | None) -> bool:
    if not url:
        return False
    lower = url.lower()
    return any(
        marker in lower
        for marker in (
            "developer.salesforce.com",
            "help.salesforce.com",
            "salesforce.com/docs",
            "docs.salesforce.com",
            "architect.salesforce.com",
        )
    )


def google_search(query: str) -> list[dict]:
    if not _google_available():
        return []

    logger.info(f"GoogleSearch | query='{query}'")
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_CX,
        "q": query,
        "num": GOOGLE_SEARCH_LIMIT,
    }
    try:
        resp = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=MCP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("items", [])[:GOOGLE_SEARCH_LIMIT]:
            url = item.get("link") or ""
            title = item.get("title") or None
            snippet = item.get("snippet") or ""
            results.append({
                "query": query,
                "title": title,
                "url": url or None,
                "document_path": None,
                "excerpt": snippet,
                "source_type": "Google Search Result",
                "authority": "Salesforce Documentation" if _is_official_salesforce_url(url) else "Google Search",
                "is_authoritative": _is_official_salesforce_url(url),
                "release": None,
                "api_version": None,
                "published_date": None,
                "last_updated": None,
                "relevance": item.get("cacheId") or item.get("formattedUrl"),
            })
        return results
    except Exception as e:
        logger.warning(f"Google search unavailable | query='{query}' | {e}")
        return []


def mcp_fetch(document_path: str) -> dict:
    logger.info(f"Fetch | path='{document_path}'")
    raw, ok = _call_mcp("salesforce_docs_fetch", {"documentPath": document_path})
    if not ok or not raw:
        logger.warning(f"Fetch empty | path='{document_path}'")
        return {}
    parsed = _parse_mcp_fetch_payload(raw)
    if not parsed["document_path"]:
        parsed["document_path"] = document_path
    return {
        "query": "",
        "excerpt": parsed["excerpt"],
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": parsed["title"] or document_path,
        "url": parsed["url"],
        "document_path": document_path,
        "relevance": None,
    }

# ---------------------------------------------------------------------------
# Provider / Model Selection — see providers.py
# ---------------------------------------------------------------------------


_PROTECTED_INTENTS = {"troubleshooting", "code_review", "limits", "release"}


def _looks_like_design_question(message: str) -> bool:
    """True for how-to, implement, and architecture questions."""
    if _is_volatile_limit_question(message):
        return False
    return sa.is_solution_question(message)


def _apply_troubleshooting_intent(message: str, parsed: dict) -> dict:
    """Bug reports and component repros are diagnosis, not architecture."""
    if parsed.get("intent") in {"limits", "release"}:
        return parsed
    if not sa.is_troubleshooting_question(message):
        return parsed
    parsed["intent"] = "troubleshooting"
    parsed["question_type"] = "diagnosis"
    parsed["requires_architecture_analysis"] = False
    parsed["requires_code_analysis"] = True
    parsed["requires_documentation"] = True
    if parsed.get("research_depth") in (None, "quick"):
        parsed["research_depth"] = "standard"
    if parsed.get("model_tier") in (None, "fast"):
        parsed["model_tier"] = "standard"
    return parsed


def _apply_design_intent(message: str, parsed: dict) -> dict:
    """Upgrade research/fact classifications when the question is a design problem."""
    if parsed.get("intent") in _PROTECTED_INTENTS:
        return parsed
    if sa.is_troubleshooting_question(message):
        return parsed
    if not _looks_like_design_question(message):
        return parsed
    parsed["intent"] = "architecture"
    parsed["question_type"] = "design"
    parsed["requires_architecture_analysis"] = True
    parsed["requires_multiple_sources"] = True
    parsed["research_depth"] = "deep"
    parsed["model_tier"] = "reasoning"
    parsed["requires_documentation"] = True
    return parsed


def classify_intent_heuristic(message: str) -> dict:
    """Rules-based intent when no LLM is configured. Prefers recall over precision."""
    parsed = dict(_SAFE_FALLBACK_INTENT)
    lower = message.lower().strip()
    parsed["is_salesforce_specific"] = True
    parsed["requires_documentation"] = True
    if _is_volatile_limit_question(message) or (
        "api version" in lower and any(token in lower for token in ("latest", "current", "what"))
    ):
        parsed["intent"] = "limits" if _is_volatile_limit_question(message) else "release"
        parsed["question_type"] = "current_fact"
        parsed["requires_current_docs"] = True
        parsed["temporal_validation_required"] = True
        parsed["research_depth"] = "quick"
        parsed["model_tier"] = "fast"
    elif sa.is_definition_shaped(message) and not _looks_like_design_question(message):
        parsed["intent"] = "quick_fact"
        parsed["question_type"] = "explanation"
        parsed["research_depth"] = "quick"
        parsed["model_tier"] = "fast"
    parsed = _apply_troubleshooting_intent(message, parsed)
    parsed = _apply_design_intent(message, parsed)
    temporal = _determine_temporal_context(message, parsed)
    parsed["temporal_validation_required"] = temporal["temporal_validation_required"]
    parsed["requested_release"] = temporal["requested_release"]
    parsed["requires_current_docs"] = bool(
        parsed.get("requires_current_docs") or temporal["current_docs_required"]
    )
    return parsed

# ---------------------------------------------------------------------------
# Input Guardrail
# ---------------------------------------------------------------------------
# Per OpenAI agent guide: guardrails are a layered defense.
# This is the first layer — a fast rules-based relevance check that runs
# BEFORE the full pipeline. It short-circuits non-Salesforce inputs cheaply
# without spending LLM tokens on classification + retrieval.

_GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening)|what can you (do|help)|who are you)\b",
    re.IGNORECASE,
)
_CLEARLY_OFF_TOPIC = re.compile(
    r"\b(weather|recipe|sports|movie|music|stock price|bitcoin|crypto price|news today)\b",
    re.IGNORECASE,
)
_SALESFORCE_SIGNALS = re.compile(
    r"\b(salesforce|apex|soql|flow|lwc|lightning|visualforce|mcp|agentforce|platform event|"  
    r"change data capture|bulk api|governor limit|org|sandbox|scratch org|metadata|"  
    r"deployment|permission set|profile|object|field|trigger|batch|queueable|"  
    r"future method|named credential|external credential|data cloud|tableau|mulesoft|"  
    r"heroku|slack|einstein|omni.?channel|service cloud|sales cloud|experience cloud|"
    r"well-?architected|architecture center|decision guide)\b",
    re.IGNORECASE,
)


def input_guardrail(message: str) -> tuple[bool, str]:
    """
    Fast rules-based input guardrail (Layer 0).
    Returns (should_continue, early_response).
    If should_continue=False, return early_response directly.
    """
    stripped = message.strip()

    # Empty input
    if not stripped:
        return False, "Please enter a question."

    # Greeting — respond helpfully without running the full pipeline
    if _GREETING_PATTERNS.match(stripped) and len(stripped) < 80:
        logger.info("Guardrail=greeting | short-circuiting pipeline")
        return False, (
            "Hi! I'm your Salesforce Technical Architect Agent. I can help with:\n\n"
            "- Salesforce architecture and design questions\n"
            "- Apex, Flow, LWC, SOQL guidance\n"
            "- Governor limits and platform limits\n"
            "- Integration patterns (Platform Events, CDC, Bulk API, REST)\n"
            "- Code review for Apex and SOQL\n"
            "- Current Salesforce release and API version information\n\n"
            "What would you like to know?"
        )

    # Clearly off-topic
    if _CLEARLY_OFF_TOPIC.search(stripped) and not _SALESFORCE_SIGNALS.search(stripped):
        logger.info("Guardrail=off_topic | short-circuiting pipeline")
        return False, (
            "I'm specialized in Salesforce architecture and development. "
            "I can't help with that topic, but feel free to ask me anything about "
            "Salesforce, Apex, Flow, integrations, or platform architecture."
        )

    return True, ""


# ---------------------------------------------------------------------------
# Intent Classification
# ---------------------------------------------------------------------------

_SAFE_FALLBACK_INTENT = {
    "intent": "research",
    "question_type": "explanation",
    "topics": [],
    "salesforce_features": [],
    "is_salesforce_specific": True,
    "requires_documentation": True,
    "requires_current_docs": False,
    "temporal_validation_required": False,
    "requested_release": None,
    "requires_multiple_sources": False,
    "requires_code_analysis": False,
    "requires_architecture_analysis": False,
    "research_depth": "standard",
    "model_tier": "standard",
}


def classify_intent(message: str, llm) -> dict:
    if llm is None:
        parsed = classify_intent_heuristic(message)
        logger.info(
            f"Intent=heuristic | Intent={parsed.get('intent')} | "
            f"ResearchDepth={parsed.get('research_depth')}"
        )
        return parsed
    try:
        resp = _invoke_resilient(llm, [
            SystemMessage(content=INTENT_CLASSIFIER_PROMPT),
            HumanMessage(content=message),
        ], "intent", message)
        raw = resp.content

        # Structured refusal path (per AI Engineer Guide Ch.3 — give the model
        # a permitted way to say "I can't determine this" rather than forcing a guess).
        # If the classifier returns {"unable_to_classify": true}, use safe fallback.
        try:
            maybe = _extract_json_payload(raw)
            if isinstance(maybe, dict) and maybe.get("unable_to_classify"):
                logger.warning("IntentClassifier=unable_to_classify | using safe fallback")
                return _apply_design_intent(
                    message,
                    _apply_troubleshooting_intent(message, dict(_SAFE_FALLBACK_INTENT)),
                )
        except Exception:
            pass

        parsed = _extract_json_payload(raw)

        temporal = _determine_temporal_context(message, parsed)
        parsed.setdefault("temporal_validation_required", temporal["temporal_validation_required"])
        parsed.setdefault("requested_release", temporal["requested_release"])
        parsed.setdefault("requires_current_docs", temporal["current_docs_required"])
        if temporal["current_docs_required"] and not temporal["historical_release_requested"]:
            parsed["requires_current_docs"] = True

        # Enforce safety: if Salesforce-specific, documentation should be required
        if parsed.get("is_salesforce_specific") and not parsed.get("requires_documentation"):
            parsed["requires_documentation"] = True
        if temporal["historical_release_requested"]:
            parsed["requested_release"] = temporal["requested_release"]
            parsed["temporal_validation_required"] = True
            parsed["requires_current_docs"] = False

        parsed = _apply_troubleshooting_intent(message, parsed)
        parsed = _apply_design_intent(message, parsed)

        logger.info(
            f"Intent={parsed.get('intent')} | "
            f"QuestionType={parsed.get('question_type')} | "
            f"SalesforceSpecific={parsed.get('is_salesforce_specific')} | "
            f"DocumentationRequired={parsed.get('requires_documentation')} | "
            f"CurrentDocsRequired={parsed.get('requires_current_docs')} | "
            f"TemporalValidation={parsed.get('temporal_validation_required')} | "
            f"RequestedRelease={parsed.get('requested_release')} | "
            f"ResearchDepth={parsed.get('research_depth')} | "
            f"ModelTier={parsed.get('model_tier')}"
        )
        return parsed
    except json.JSONDecodeError as e:
        logger.warning(f"Intent classifier invalid JSON: {e} — using safe fallback")
        return _apply_design_intent(
            message,
            _apply_troubleshooting_intent(message, dict(_SAFE_FALLBACK_INTENT)),
        )
    except Exception as e:
        logger.warning(f"Intent classification failed: {e} — using safe fallback")
        return _apply_design_intent(
            message,
            _apply_troubleshooting_intent(message, dict(_SAFE_FALLBACK_INTENT)),
        )

# ---------------------------------------------------------------------------
# Search Query Planning
# ---------------------------------------------------------------------------

def _is_architecture_question(intent_data: dict) -> bool:
    intent = str(intent_data.get("intent") or "").lower()
    return intent == "architecture" or bool(intent_data.get("requires_architecture_analysis"))


_SUBJECT_RE = re.compile(
    r"`([^`]+)`"
    r"|\"([A-Za-z][\w.-]+)\""
    r"|(\b[A-Za-z]\w*\.[A-Za-z]\w*)"
    r"|(\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+)"
    r"|(\b[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]+)+)"
    r"|(\b[A-Za-z]\w+__[a-zA-Z]\b)"
)
_CONTRACT_EVIDENCE_RE = re.compile(
    r"\b(methods?|properties|property|attributes?|events?|annotation|"
    r"parameter|signature|interface|governor)\b",
    re.IGNORECASE,
)
_QUERY_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "when", "after",
    "same", "back", "into", "does", "not", "there", "way", "fix", "example",
    "steps", "reproduce", "enter", "however", "please", "would", "like",
    "what", "your", "could", "should", "then", "than", "have", "been",
    "actual", "expected", "behavior", "behaviour",
}
_MAX_QUERY_CHARS = 140


def _is_diagnosis(message: str, intent_data: dict | None = None) -> bool:
    intent = str((intent_data or {}).get("intent") or "")
    return intent == "troubleshooting" or sa.is_troubleshooting_question(message)


def _named_subjects(message: str) -> list[str]:
    """Identifiers the user named — syntax, not a product catalog."""
    found: list[str] = []
    for match in _SUBJECT_RE.finditer(message):
        name = next((part for part in match.groups() if part), "").strip()
        if name and name not in found:
            found.append(name)
    return found


def _symptom_tokens(message: str, subjects: list[str] | None = None) -> list[str]:
    skip = set(_QUERY_STOPWORDS)
    for subject in subjects or _named_subjects(message):
        skip.add(subject.lower())
        skip.update(re.findall(r"[a-z0-9]+", subject.lower()))
    words: list[str] = []
    for word in re.findall(r"[a-z0-9-]{3,}", message.lower()):
        if word in skip or word in words:
            continue
        words.append(word)
        if len(words) >= 6:
            break
    return words


def _token_in_blob(token: str, blob: str) -> bool:
    return bool(re.search(rf"\b{re.escape(token)}\b", blob))


def _has_diagnosis_coverage(message: str, evidence: list[dict]) -> bool:
    blob = " ".join(_evidence_blob(item) for item in evidence).lower()
    if _CONTRACT_EVIDENCE_RE.search(blob):
        return True
    tokens = _symptom_tokens(message)
    if not tokens:
        return True
    hits = sum(1 for token in tokens if _token_in_blob(token, blob))
    return hits >= min(2, len(tokens))


def _ensure_named_subject_queries(
    result: list[str], message: str, intent_data: dict
) -> list[str]:
    """Guarantee the user's named subjects survive query planning.

    A factual named-subject question ("how do I read clientChatQueuePosition?")
    or a diagnosis must retrieve against the literal identifier the user named.
    The LLM planner can paraphrase it away (topic drift); re-insert any named
    subject not already present so retrieval stays anchored to what was named.
    Scoped to fact-lookup/diagnosis so it never crowds out design queries on
    genuine architecture questions.
    """
    if not (sa.is_fact_lookup(message) or _is_diagnosis(message, intent_data)):
        return result
    subjects = _named_subjects(message)
    if not subjects:
        return result
    lowered = " ".join(result).lower()
    missing = [s for s in subjects if s.lower() not in lowered]
    if not missing:
        return result
    # Front-load so the named subject survives the later max_q trim.
    return missing + result


def _compress_query(message: str) -> str:
    first = re.split(r"[.\n]", message.strip())[0].strip()
    if len(first) <= _MAX_QUERY_CHARS:
        return first or message[:_MAX_QUERY_CHARS]
    return " ".join(first.split()[:16])


def _compact_search_queries(message: str, intent_data: dict) -> list[str]:
    """Official-docs queries. Never send a multi-paragraph repro as query #1."""
    subjects = _named_subjects(message)
    if not subjects and _is_diagnosis(message, intent_data):
        subjects = [_compress_query(message)]
    if subjects and _is_diagnosis(message, intent_data):
        queries: list[str] = []
        for name in subjects[:2]:
            if name not in queries:
                queries.append(name)
            reference = f"{name} Salesforce reference"
            if reference not in queries:
                queries.append(reference)
            tail = " ".join(_symptom_tokens(message, [name])[:4])
            if tail:
                combined = f"{name} {tail}"
                if combined not in queries:
                    queries.append(combined)
        return queries
    if subjects:
        return subjects[:2]
    if len(message) > _MAX_QUERY_CHARS:
        return [_compress_query(message)]
    return [message]


def _is_architecture_center_evidence(item: dict) -> bool:
    blob = _evidence_blob(item)
    if "architect.salesforce.com" in blob:
        return True
    path = str(item.get("document_path") or "").lower()
    return (
        "well-architected" in path
        or "/docs/architect/" in path
        or path.startswith("architect/")
        or "decision-guides" in path
        or "decision_guides" in path
    )


def _architecture_search_boosters(message: str, intent_data: dict) -> list[str]:
    if not _is_architecture_question(intent_data):
        return []
    boosters = ["Salesforce Well-Architected Framework Trusted Easy Adaptable"]
    haystack = " ".join(
        [
            message,
            " ".join(str(part) for part in (intent_data.get("topics") or [])),
            " ".join(str(part) for part in (intent_data.get("salesforce_features") or [])),
        ]
    ).lower()
    integrationish = any(
        token in haystack
        for token in (
            "integrat",
            "mcp",
            " api",
            "apis",
            "platform event",
            "cdc",
            "change data",
            "erp",
            "mulesoft",
            "pubsub",
            "pub/sub",
            "callout",
            "sync",
            "bulk",
            "ldv",
        )
    )
    if integrationish:
        boosters.append("Salesforce Architecture Center integration patterns")
    else:
        boosters.append("Salesforce Architecture Center decision guides")
    return boosters


def translate_requirement(message: str, intent_data: dict, llm=None) -> dict:
    """Business requirement → Salesforce technical search plan."""
    if not _is_architecture_question(intent_data) and not sa.is_solution_question(message):
        return dict(sa.EMPTY_PLAN)
    plan = sa.heuristic_plan(message)
    if not plan.get("active"):
        if _is_architecture_question(intent_data):
            plan = dict(sa.EMPTY_PLAN)
            plan["active"] = True
            plan["skip_raw"] = sa.is_business_requirement(message)
        return plan
    if (
        llm is None
        or not plan.get("skip_raw")
        or not callable(getattr(llm, "invoke", None))
    ):
        return plan
    try:
        resp = _invoke_resilient(
            llm,
            [
                SystemMessage(content=SOLUTION_ARCHITECT_PROMPT),
                HumanMessage(content=message),
            ],
            "solution",
            message,
        )
        payload = _extract_json_payload(resp.content)
        if isinstance(payload, dict):
            return sa.merge_llm_plan(plan, payload)
    except Exception as exc:
        logger.warning(f"SolutionArchitect=llm_failed | {exc}")
    return plan


def _merge_architecture_queries(
    queries: list[str],
    intent_data: dict,
    message: str,
    max_q: int,
    solution_plan: dict | None = None,
) -> list[str]:
    boosters = _architecture_search_boosters(message, intent_data)
    plan = solution_plan if solution_plan is not None else {}
    capability_queries = list(plan.get("search_queries") or [])
    skip_raw = bool(plan.get("skip_raw") and capability_queries)
    if not boosters and not capability_queries:
        return queries[:max_q]
    merged: list[str] = []

    def _add(query: str) -> None:
        text = str(query or "").strip()
        if not text or text in merged:
            return
        if skip_raw and text == message.strip():
            return
        merged.append(text)

    if skip_raw:
        reserved = min(len(boosters), 1)
        cap_budget = max(1, max_q - reserved)
        for query in capability_queries[:cap_budget]:
            _add(query)
        if queries:
            _add(queries[0])
        for booster in boosters:
            _add(booster)
        for query in queries[1:]:
            _add(query)
        return merged[:max_q]

    if queries:
        _add(queries[0])
    for booster in boosters:
        _add(booster)
    for query in queries[1:]:
        _add(query)
    for query in capability_queries:
        _add(query)
    return merged[:max_q]


def _is_wrong_family_item(message: str, item: dict, solution_plan: dict | None = None) -> bool:
    plan = solution_plan if solution_plan is not None else sa.heuristic_plan(message)
    return sa.is_avoided_evidence(plan, item)


def _architecture_family_gap(
    message: str,
    intent_data: dict,
    evidence: list[dict],
    solution_plan: dict | None = None,
) -> dict:
    plan = solution_plan if solution_plan is not None else translate_requirement(message, intent_data)
    return sa.evidence_gap(plan, evidence)


def _drop_wrong_family_evidence(
    message: str,
    evidence: list[dict],
    solution_plan: dict | None = None,
) -> list[dict]:
    plan = solution_plan if solution_plan is not None else sa.heuristic_plan(message)
    return sa.drop_avoided_evidence(plan, evidence)


def _llm_architecture_rescue_queries(
    message: str,
    evidence: list[dict],
    llm,
    solution_plan: dict | None = None,
) -> list[str]:
    if llm is None or not callable(getattr(llm, "invoke", None)) or not evidence:
        return []
    titles = []
    for item in evidence[:6]:
        title = item.get("title") or item.get("authority") or "Salesforce source"
        url = item.get("url") or ""
        titles.append(f"- {title} {url}".strip())
    plan = solution_plan or {}
    prompt = (
        f"question={json.dumps(message)}\n"
        f"technical_requirements={json.dumps(plan.get('technical_requirements') or [])}\n"
        f"objects={json.dumps(plan.get('objects') or [])}\n"
        f"evidence:\n" + "\n".join(titles)
    )
    try:
        resp = _invoke_resilient(
            llm,
            [
                SystemMessage(content=EVIDENCE_CRITIC_PROMPT),
                HumanMessage(content=prompt),
            ],
            "critic",
            message,
        )
        payload = _extract_json_payload(resp.content)
        if not isinstance(payload, dict) or payload.get("aligned", True):
            return []
        queries = payload.get("rescue_queries") or []
        cleaned: list[str] = []
        for query in queries:
            text = str(query or "").strip()
            if len(text) < 8 or text == message.strip():
                continue
            cleaned.append(text)
        return cleaned[:ARCHITECTURE_RESCUE_BUDGET]
    except Exception as exc:
        logger.warning(f"ArchitectureCritic=llm_failed | {exc}")
        return []


def build_search_queries(
    message: str,
    intent_data: dict,
    llm,
    temporal_context: dict | None = None,
    solution_plan: dict | None = None,
) -> list[str]:
    depth = intent_data.get("research_depth", "standard")
    max_q = SEARCH_BUDGET.get(depth, 2)
    if _is_architecture_question(intent_data):
        max_q = max(max_q, ARCHITECTURE_SEARCH_BUDGET)
    if _is_diagnosis(message, intent_data):
        max_q = max(max_q, 3)
    features = intent_data.get("salesforce_features", [])
    temporal = temporal_context or _determine_temporal_context(message, intent_data)
    if _needs_seasonal_authority(message, intent_data, temporal):
        max_q = max(max_q, SEASONAL_SEARCH_BUDGET)
    boosters = _temporal_search_boosters(message, temporal)
    definition_intent = _is_definition_question(message, intent_data)
    definition_boosters = _definition_search_boosters(message, intent_data) if definition_intent else []
    plan = solution_plan if solution_plan is not None else translate_requirement(message, intent_data, llm)

    logger.debug(
        f"QueryPlan | definition_intent={definition_intent} | "
        f"core_concept={_extract_core_concept(message) if definition_intent else None} | "
        f"boosters={boosters} | temporal={temporal} | "
        f"sa_skip_raw={plan.get('skip_raw')} | sa_caps={plan.get('capabilities')}"
    )

    if definition_intent:
        max_q = max(max_q, 4)
        core = _extract_core_concept(message)
        result = _definition_query_variants(core, intent_data)
        for booster in reversed(definition_boosters):
            if booster not in result:
                result.insert(0, booster)
        for booster in reversed(boosters):
            if booster not in result:
                result.insert(0, booster)
        result = result[:max_q]
        result = _merge_architecture_queries(result, intent_data, message, max_q, plan)
        result = _merge_temporal_queries(result, message, intent_data, temporal, max_q)
        logger.info(f"Planned queries: {result}")
        return result

    if llm is None:
        if plan.get("skip_raw") and plan.get("search_queries"):
            result = list(plan["search_queries"])
        else:
            result = _compact_search_queries(message, intent_data)
        if not result:
            result = [message]
        if len(result) < max_q:
            for booster in boosters:
                if booster not in result:
                    result.append(booster)
        result = _merge_architecture_queries(result, intent_data, message, max_q, plan)
        result = _merge_temporal_queries(result, message, intent_data, temporal, max_q)
        logger.info(f"Planned queries: {result}")
        return result

    try:
        prompt_input = (
            f"question={json.dumps(message)}, "
            f"intent={json.dumps(intent_data.get('intent'))}, "
            f"features={json.dumps(features)}, "
            f"technical_requirements={json.dumps(plan.get('technical_requirements') or [])}, "
            f"capability_queries={json.dumps(plan.get('search_queries') or [])}, "
            f"temporal={json.dumps(temporal)}, "
            f"boosters={json.dumps(boosters)}, "
            f"max_queries={max_q}"
        )
        resp = _invoke_resilient(llm, [
            SystemMessage(content=QUERY_PLANNER_PROMPT),
            HumanMessage(content=prompt_input),
        ], "query", message)
        queries = _extract_json_payload(resp.content)
        if isinstance(queries, list) and queries:
            result = [str(q) for q in queries[:max_q]]
            for booster in reversed(definition_boosters):
                if booster not in result:
                    result.insert(0, booster)
            if len(result) < max_q:
                for booster in boosters:
                    if booster not in result:
                        result.append(booster)
            if definition_intent:
                core = _extract_core_concept(message)
                for query in _definition_query_variants(core, intent_data):
                    if query not in result:
                        result.insert(0, query)
            result = _ensure_named_subject_queries(result, message, intent_data)
            result = result[:max_q]
            result = _merge_architecture_queries(result, intent_data, message, max_q, plan)
            result = _merge_temporal_queries(result, message, intent_data, temporal, max_q)
            logger.info(f"Planned queries: {result}")
            return result
    except Exception as e:
        logger.warning(f"Query planning failed: {e} — using message as query")

    return _merge_temporal_queries(
        _merge_architecture_queries([message], intent_data, message, max_q, plan),
        message,
        intent_data,
        temporal,
        max_q,
    )


def _extract_core_concept(message: str) -> str:
    cleaned = re.sub(r"^[\"']+|[\"']+$", "", message.strip()).strip()
    lower = cleaned.lower()
    prefixes = (
        "what is",
        "what are",
        "explain",
        "define",
        "give an overview of",
        "give me an overview of",
        "tell me about",
        "what's",
        "whats",
    )
    for prefix in prefixes:
        if lower.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip(" ?:-")
            break
    # Strip leading articles so "a Platform Event" → "Platform Event"
    cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip(" ?")


def _is_definition_question(message: str, intent_data: dict) -> bool:
    question_type = str(intent_data.get("question_type", "")).lower()
    intent = str(intent_data.get("intent", "")).lower()
    if question_type == "explanation" or intent in {"quick_fact", "research"}:
        lower = message.strip().lower()
        return lower.startswith((
            "what is",
            "what are",
            "explain",
            "define",
            "give an overview of",
            "give me an overview of",
            "tell me about",
        ))
    return False


def _definition_query_variants(core_concept: str, intent_data: dict) -> list[str]:
    if not core_concept:
        return []
    variants = [
        core_concept,
        f"{core_concept} overview",
        f"{core_concept} introduction",
        f"{core_concept} home",
    ]
    features = intent_data.get("salesforce_features", []) or []
    for feature in features[:1]:
        variants.append(f"{feature} overview")
    return variants


def _definition_search_boosters(message: str, intent_data: dict) -> list[str]:
    core = _extract_core_concept(message)
    boosters = []
    if core:
        boosters.append(f"{core} Salesforce documentation")
        boosters.append(f"{core} overview Salesforce")
    features = intent_data.get("salesforce_features", []) or []
    if features:
        boosters.append(f"{features[0]} overview")
    return boosters


def _definition_rescue_queries(message: str, intent_data: dict) -> list[str]:
    core = _extract_core_concept(message)
    if not core:
        return []
    queries = [
        f"{core} overview",
        f"{core} introduction",
        f"{core} guide",
        f"{core} home",
        f"{core} Salesforce overview",
        f"Salesforce {core} overview",
    ]
    features = intent_data.get("salesforce_features", []) or []
    if features:
        queries.append(f"{features[0]} overview")
    # Preserve order while deduplicating.
    return list(dict.fromkeys(q for q in queries if q))

# ---------------------------------------------------------------------------
# Evidence Retrieval
# ---------------------------------------------------------------------------

def _is_sufficient(evidence: list[dict], intent_data: dict, require_authoritative: bool = False) -> bool:
    non_empty = [e for e in evidence if e.get("excerpt")]
    if require_authoritative and not _has_authoritative_salesforce_source(non_empty):
        return False
    depth = intent_data.get("research_depth", "standard")
    if depth == "quick":
        return len(non_empty) >= 1
    if depth == "standard":
        return len(non_empty) >= 1
    return len(non_empty) >= 2  # deep needs at least 2


def _topic_terms(message: str, intent_data: dict, solution_plan: dict | None = None) -> set[str]:
    stopwords = {
        "salesforce",
        "documentation",
        "current",
        "release",
        "notes",
        "docs",
        "latest",
        "overview",
        "please",
        "would",
        "like",
        "what",
        "whats",
        "your",
        "could",
        "should",
        "question",
        "query",
        "asper",
        # Generic function words: no topical signal, must not become match terms.
        "the", "and", "for", "with", "from", "that", "this", "about", "how",
        "why", "when", "where", "which", "who", "does", "did", "can", "will",
        "you", "our", "get", "want", "need", "there", "into", "using", "use",
        "have", "has", "had", "are", "was", "were", "any", "all", "its",
        "their", "them", "then", "than", "give", "tell", "show", "find",
    }
    terms: set[str] = set()
    sources = []
    sources.extend(intent_data.get("topics", []) or [])
    sources.extend(intent_data.get("salesforce_features", []) or [])
    if solution_plan and solution_plan.get("skip_raw"):
        sources.extend(solution_plan.get("objects") or [])
        sources.extend(solution_plan.get("search_queries") or [])
        sources.extend(solution_plan.get("technical_requirements") or [])
    elif _is_diagnosis(message, intent_data):
        sources.extend(_named_subjects(message) or [_compress_query(message)])
        sources.extend(_compact_search_queries(message, intent_data))
    else:
        sources.append(message)
    for source in sources:
        for word in re.findall(r"[a-zA-Z]{3,}", str(source).lower()):
            if word not in stopwords:
                terms.add(word)
    return terms


def _is_relevant(
    item: dict,
    terms: set[str],
    min_matches: int = 2,
    intent_data: dict | None = None,
    solution_plan: dict | None = None,
) -> bool:
    if intent_data and _is_architecture_question(intent_data) and _is_architecture_center_evidence(item):
        return True
    if solution_plan and sa.has_capability_evidence(solution_plan, [item]):
        return True
    if solution_plan and sa.is_avoided_evidence(solution_plan, item):
        return False
    if not terms:
        return True
    blob = _evidence_blob(item)
    matches = sum(1 for term in terms if term in blob)
    min_matches = max(2, len(terms) // 3)
    return matches >= min_matches


def _filter_relevant_evidence(
    evidence: list[dict],
    terms: set[str],
    intent_data: dict | None = None,
    solution_plan: dict | None = None,
    message: str = "",
) -> list[dict]:
    if not terms:
        return evidence
    filtered = [
        item
        for item in evidence
        if _is_relevant(item, terms, intent_data=intent_data, solution_plan=solution_plan)
    ]
    if intent_data and _is_architecture_question(intent_data):
        for item in evidence:
            if _is_architecture_center_evidence(item) and item not in filtered:
                filtered.append(item)
    if message and intent_data and _requires_court_order(message, intent_data):
        for item in evidence:
            if item in filtered:
                continue
            if (
                _court_order_covers_question(item, message)
                or (_is_standing_statute(item) and _statute_on_topic(item, message))
                or _is_limits_cheatsheet(item)
            ):
                filtered.append(item)
    if solution_plan:
        sa.drop_avoided_evidence(solution_plan, filtered)
    return filtered


def _build_unverified_relevance_response(message: str, evidence: list[dict]) -> str:
    source_lines = []
    for item in _sort_evidence_by_trust(evidence):
        title = item.get("title") or item.get("authority") or "Salesforce source"
        source_type = item.get("source_type") or item.get("authority") or "Salesforce source"
        source_lines.append(f"- {title} ({source_type})")

    source_block = "\n".join(source_lines) if source_lines else "- No relevant Salesforce source was found."
    return (
        "## Answer\n"
        "[Unverified] I could not find retrieved evidence that matched the question topic.\n\n"
        "## Note\n"
        "The retrieved evidence was not relevant enough to support a grounded Salesforce answer.\n\n"
        "## Sources\n"
        f"{source_block}"
    )


# Function words + generic doc noise that carry no topical signal. Deliberately
# NOT plan-derived: this gate must be an INDEPENDENT check of evidence vs. the
# literal question, so it cannot inherit a drifted plan's vocabulary.
_QUESTION_STOPWORDS = {
    "salesforce", "documentation", "docs", "doc", "the", "and", "for", "with",
    "from", "that", "this", "about", "how", "what", "whats", "why", "when",
    "where", "which", "who", "does", "did", "can", "could", "should", "would",
    "will", "your", "you", "our", "get", "getting", "got", "please", "want",
    "need", "there", "into", "using", "use", "used", "have", "has", "had",
    "are", "was", "were", "any", "all", "its", "their", "them", "then", "than",
    "latest", "current", "please", "give", "tell", "show", "find", "make",
    "set", "let", "via", "onto", "off",
}


def _question_content_terms(message: str) -> set[str]:
    """Topical terms from the LITERAL question, independent of any plan."""
    terms: set[str] = set()
    for word in re.findall(r"[a-z0-9]{3,}", str(message or "").lower()):
        if word in _QUESTION_STOPWORDS:
            continue
        terms.add(word)
    return terms


def _term_prefix_in_blob(term: str, blob: str) -> bool:
    # Prefix match so "event" covers "events", "queue" covers "queues", etc.
    return bool(re.search(rf"\b{re.escape(term)}", blob))


def _evidence_covers_question(message: str, evidence: list[dict]) -> bool:
    """Does any retrieved item actually cover the question's topical terms?

    Independent of the relevance filter (which derives terms from the plan and so
    cannot catch planner drift). Requires the best-matching item to cover a
    scaled fraction of the question's content terms.
    """
    terms = _question_content_terms(message)
    if len(terms) < 2:
        return True  # too few terms to judge — do not block
    threshold = max(2, (len(terms) + 1) // 2)
    best = 0
    for item in evidence:
        blob = _evidence_blob(item)
        matched = sum(1 for term in terms if _term_prefix_in_blob(term, blob))
        if matched > best:
            best = matched
    return best >= threshold


def _pack_is_off_topic(message: str, intent_data: dict, evidence: list[dict]) -> bool:
    """True when retrieved evidence does not answer the literal question.

    Skips genuine architecture/design questions (conceptual docs legitimately
    have low lexical overlap) UNLESS the question is a factual named-subject
    lookup — those must always be answered on-topic, even if a classifier
    misrouted them to architecture.
    """
    if not evidence:
        return False  # empty-pack is handled by the no-evidence paths
    architecture = _is_architecture_question(intent_data)
    if architecture and not sa.is_fact_lookup(message):
        return False
    return not _evidence_covers_question(message, evidence)


def _build_off_topic_pack_response(message: str, evidence: list[dict]) -> str:
    """Retrieval-only response when the pack does not match the question."""
    source_lines = []
    for item in _sort_evidence_by_trust(evidence):
        title = item.get("title") or item.get("authority") or "Salesforce source"
        url = item.get("url") or ""
        source_lines.append(f"- {title}{(' — ' + url) if url else ''}")
    source_block = "\n".join(source_lines) if source_lines else "- Nothing retrieved."
    return (
        "## No on-topic documentation found\n"
        "[Unverified] The retrieved Salesforce documentation does not appear to "
        f"answer this question: \"{message.strip()}\". Do not treat the sources "
        "below as an answer — they were retrieved but did not match the topic.\n\n"
        "Try naming the exact API, object, or feature (for example the specific "
        "class, method, field, or product name) so retrieval can target it.\n\n"
        "## Retrieved (unmatched) sources\n"
        f"{source_block}"
    )


def _evidence_identity(item: dict) -> tuple[str, str, str]:
    return (
        str(item.get("url") or "").strip().lower(),
        str(item.get("document_path") or "").strip().lower(),
        str(item.get("title") or "").strip().lower(),
    )


def _metadata_hints(item: dict) -> str:
    parts = [
        item.get("title"),
        item.get("url"),
        item.get("document_path"),
        item.get("source_type"),
        item.get("authority"),
    ]
    return " ".join(str(part or "").lower() for part in parts)


def _definition_page_bonus(item: dict) -> int:
    blob = _metadata_hints(item)
    score = 0
    if any(token in blob for token in ("overview", "home", "introduction", "intro", "getting-started", "getting started", "guide")):
        score += 20
    if any(token in blob for token in ("mapping", "config", "setup", "troubleshoot", "troubleshooting", "admin")):
        score -= 15
    if any(token in blob for token in ("review-user-mapping", "other-data-governance", "field_reference", "reference")):
        score -= 10
    logger.debug(
        f"DefinitionRerank | title={item.get('title')} | "
        f"path={item.get('document_path')} | bonus={score}"
    )
    return score


def _definition_evidence_is_sufficient(evidence: list[dict], topic_terms: set[str]) -> bool:
    if not evidence:
        return False
    relevant = [item for item in evidence if _is_relevant(item, topic_terms)]
    if not relevant:
        return False
    return any(_definition_page_bonus(item) > 0 for item in relevant)


def _evidence_sufficiency_stage(
    evidence: list[dict],
    intent_data: dict,
    message: str,
    topic_terms: set[str],
    require_authoritative: bool = False,
    solution_plan: dict | None = None,
) -> dict:
    """Canonical sufficiency gate for retrieval.

    Returns a small diagnostic payload so retrieval can explain why it stopped
    or continued searching.
    """
    non_empty = [item for item in evidence if item.get("excerpt")]
    definition_intent = _is_definition_question(message, intent_data)
    topic_match = not topic_terms or any(
        _is_relevant(
            item,
            topic_terms,
            intent_data=intent_data,
            solution_plan=solution_plan,
        )
        for item in non_empty
    )
    definition_satisfied = True

    if require_authoritative and not _has_authoritative_salesforce_source(non_empty):
        return {
            "sufficient": False,
            "reason": "missing_authoritative_source",
            "topic_match": topic_match,
            "definition_satisfied": definition_satisfied,
        }

    if not non_empty:
        return {
            "sufficient": False,
            "reason": "no_evidence",
            "topic_match": False,
            "definition_satisfied": False,
        }

    if definition_intent:
        definition_satisfied = _definition_evidence_is_sufficient(non_empty, topic_terms)

    sufficient = _is_sufficient(evidence, intent_data, require_authoritative=require_authoritative)
    sufficient = sufficient and topic_match and definition_satisfied
    reason = "sufficient" if sufficient else "insufficient"
    if not topic_match:
        reason = "topic_mismatch"
    elif definition_intent and not definition_satisfied:
        reason = "definition_not_satisfied"
    elif sufficient and _is_diagnosis(message, intent_data) and not _has_diagnosis_coverage(message, non_empty):
        sufficient = False
        reason = "diagnosis_uncovered"
    elif sufficient and _requires_court_order(message, intent_data):
        if not any(_court_order_covers_question(item, message) for item in non_empty):
            sufficient = False
            reason = (
                "release_notes_missing"
                if not any(_is_release_notes_evidence(item) for item in non_empty)
                else "release_notes_off_topic"
            )

    return {
        "sufficient": sufficient,
        "reason": reason,
        "topic_match": topic_match,
        "definition_satisfied": definition_satisfied,
    }


def _merge_unique_evidence(existing: list[dict], new_items: list[dict]) -> None:
    seen = {_evidence_identity(item) for item in existing}
    for item in new_items:
        ident = _evidence_identity(item)
        if ident in seen:
            continue
        existing.append(item)
        seen.add(ident)


def _sort_evidence_for_intent(evidence: list[dict], intent_data: dict, message: str) -> list[dict]:
    base = _sort_evidence_by_trust(evidence)
    if _is_architecture_question(intent_data):
        return sorted(
            base,
            key=lambda item: (
                1 if _is_architecture_center_evidence(item) else 0,
                _source_trust_score(item),
                len(_metadata_hints(item)),
            ),
            reverse=True,
        )
    if _requires_court_order(message, intent_data):
        return sorted(
            base,
            key=lambda item: (
                _seasonal_authority_rank(item, message),
                _source_trust_score(item),
                len(_metadata_hints(item)),
            ),
            reverse=True,
        )
    if not _is_definition_question(message, intent_data):
        return base
    return sorted(
        base,
        key=lambda item: (
            _definition_page_bonus(item),
            _source_trust_score(item),
            len(_metadata_hints(item)),
        ),
        reverse=True,
    )


def _search_hits(result) -> list[dict]:
    if not result:
        return []
    if isinstance(result, list):
        return [item for item in result if item]
    return [result]


def _run_retrieval_query(query: str, evidence: list[dict], *, log_label: str = "Search") -> None:
    logger.info(f"{log_label} | query='{query}'")
    hits = _search_hits(mcp_search(query))
    if hits:
        _merge_unique_evidence(evidence, hits)
    if GOOGLE_SEARCH_MODE in {"hybrid", "always"} and _google_available():
        google_hits = google_search(query)
        if google_hits:
            logger.info(
                f"GoogleSearchSupplement | query='{query}' | hits={len(google_hits)}"
            )
            _merge_unique_evidence(evidence, google_hits)


def _use_generous_truncation(intent_data: dict, message: str, temporal: dict) -> bool:
    """Widen the excerpt cap when a truncated value would corrupt the answer.

    Limits/release/current-fact answers hinge on a specific number or version, and
    a definition needs the whole conceptual statement — truncating those mid-value
    is a fluent-but-wrong risk the north star forbids. Everything else keeps the
    default cap so ordinary packs stay lean.
    """
    return bool(
        _is_current_fact_request(intent_data, temporal)
        or temporal.get("historical_release_requested")
        or intent_data.get("intent") in {"release", "limits"}
        or _is_definition_question(message, intent_data)
    )


def retrieve_evidence(
    message: str,
    intent_data: dict,
    llm,
    temporal_context: dict = None,
    solution_plan: dict | None = None,
) -> list[dict]:
    """Deterministic bounded retrieval with an intent-scoped excerpt-truncation cap.

    Thin wrapper: resolves temporal context once, widens the MCP excerpt budget for
    value-bearing intents (see `_use_generous_truncation`), and always resets it so
    the wider cap never leaks into a later retrieval on the same context.
    """
    if not intent_data.get("requires_documentation"):
        logger.info("DocumentationRequired=false — skipping retrieval")
        return []
    temporal = temporal_context or validate_temporal_context(message, intent_data, llm)
    budget = MCP_MAX_TOKENS_GENEROUS if _use_generous_truncation(intent_data, message, temporal) else MCP_MAX_TOKENS
    token = _MCP_TOKEN_BUDGET.set(budget)
    try:
        return _retrieve_evidence_impl(message, intent_data, llm, temporal, solution_plan)
    finally:
        _MCP_TOKEN_BUDGET.reset(token)


def _retrieve_evidence_impl(
    message: str,
    intent_data: dict,
    llm,
    temporal_context: dict = None,
    solution_plan: dict | None = None,
) -> list[dict]:
    """
    Deterministic bounded retrieval.
    Gate: requires_documentation (not requires_current_docs).
    """
    if not intent_data.get("requires_documentation"):
        logger.info("DocumentationRequired=false — skipping retrieval")
        return []

    depth = intent_data.get("research_depth", "standard")
    search_budget = SEARCH_BUDGET.get(depth, 2)
    fetch_budget  = FETCH_BUDGET.get(depth, 1)
    temporal = temporal_context or validate_temporal_context(message, intent_data, llm)

    require_temporal_depth = (
        _is_current_fact_request(intent_data, temporal)
        or temporal.get("historical_release_requested")
        or intent_data.get("intent") in {"release", "limits"}
    )
    if _is_definition_question(message, intent_data):
        search_budget = max(search_budget, 4)
    if _is_architecture_question(intent_data):
        search_budget = max(search_budget, ARCHITECTURE_SEARCH_BUDGET)
        fetch_budget = max(fetch_budget, ARCHITECTURE_FETCH_BUDGET)
    if _is_diagnosis(message, intent_data):
        search_budget = max(search_budget, 3)
    if _needs_seasonal_authority(message, intent_data, temporal):
        search_budget = max(search_budget, SEASONAL_SEARCH_BUDGET)
        fetch_budget = max(fetch_budget, SEASONAL_FETCH_BUDGET)
    if require_temporal_depth:
        depth = "standard" if depth == "quick" else depth
        search_budget = max(search_budget, SEARCH_BUDGET["standard"])
        fetch_budget = max(fetch_budget, FETCH_BUDGET["standard"])
        logger.info(
            f"RetrievalDepthOverride=temporal | depth={depth} | "
            f"search_budget={search_budget} | fetch_budget={fetch_budget}"
        )
    require_authoritative = _is_current_fact_request(intent_data, temporal)

    evidence: list[dict] = []
    plan = solution_plan if solution_plan is not None else translate_requirement(
        message, intent_data, llm
    )

    # Build focused queries
    try:
        queries = build_search_queries(
            message, intent_data, llm, temporal, solution_plan=plan
        )
    except TypeError:
        queries = build_search_queries(message, intent_data, llm, temporal)
    topic_terms = _topic_terms(message, intent_data, plan)

    logger.info(f"SearchBudget={search_budget} | Queries={len(queries)}")
    searched_queries: list[str] = []

    def _already_searched(query: str) -> bool:
        return query.strip().lower() in {item.strip().lower() for item in searched_queries}

    def _search(query: str, log_label: str = "Search") -> None:
        searched_queries.append(query)
        _run_retrieval_query(query, evidence, log_label=log_label)

    # Execute searches within budget
    for i, query in enumerate(queries[:search_budget], 1):
        _search(query, log_label=f"Search #{i}")

        sufficiency = _evidence_sufficiency_stage(
            evidence,
            intent_data,
            message,
            topic_terms,
            require_authoritative=require_authoritative,
            solution_plan=plan,
        )
        family_gap = _architecture_family_gap(message, intent_data, evidence, plan)
        if sufficiency["sufficient"]:
            if (
                _is_architecture_question(intent_data)
                and not any(_is_architecture_center_evidence(item) for item in evidence)
            ):
                logger.info("ArchitectureCenterMissing=continue_search")
            elif family_gap["needs_rescue"]:
                logger.info("ArchitectureCritic=continue_search | missing_product_family")
            else:
                logger.info(f"EvidenceSufficient=true after {i} search(es)")
                break

    if GOOGLE_SEARCH_MODE == "fallback" and _google_available():
        needs_google = (
            not evidence
            or not any(_is_relevant(item, topic_terms, intent_data=intent_data) for item in evidence)
            or (require_authoritative and not _has_authoritative_salesforce_source(evidence))
        )
        if needs_google:
            logger.info("GoogleFallback=enabled | supplementing MCP results")
            google_hits: list[dict] = []
            for query in queries[:search_budget]:
                google_hits.extend(google_search(query))
            _merge_unique_evidence(evidence, google_hits)
            if evidence:
                logger.info(f"GoogleFallbackEvidenceCount={len(google_hits)}")

    family_gap = _architecture_family_gap(message, intent_data, evidence, plan)
    if family_gap["needs_rescue"]:
        rescue_queries = [
            query
            for query in family_gap["rescue_queries"]
            if not _already_searched(query)
        ][:ARCHITECTURE_RESCUE_BUDGET]
        if rescue_queries:
            logger.info(f"ArchitectureCritic=rescue | queries={rescue_queries}")
            for query in rescue_queries:
                _search(query, log_label="RescueSearch")
        family_gap = _architecture_family_gap(message, intent_data, evidence, plan)
        if family_gap["needs_rescue"]:
            llm_rescue = [
                query
                for query in _llm_architecture_rescue_queries(message, evidence, llm, plan)
                if not _already_searched(query)
            ]
            for query in llm_rescue[:ARCHITECTURE_RESCUE_BUDGET]:
                _search(query, log_label="CriticSearch")

    if (
        _is_architecture_question(intent_data)
        and not any(_is_architecture_center_evidence(item) for item in evidence)
    ):
        second_pass = [
            query
            for query in _architecture_search_boosters(message, intent_data)
            if not _already_searched(query)
        ][:ARCHITECTURE_SECOND_PASS_BUDGET]
        if second_pass:
            logger.info(f"ArchitectureSecondPass={second_pass}")
            for query in second_pass:
                _search(query, log_label="SecondPass")

    evidence = _drop_wrong_family_evidence(message, evidence, plan)
    logger.info(f"EvidenceCount={len(evidence)}")

    # Prefer release/current-doc fetches when temporal validation is required.
    if temporal.get("temporal_validation_required") and evidence:
        first_excerpt = evidence[0].get("excerpt", "")
        if temporal.get("historical_release_requested"):
            logger.info(f"TemporalScope=historical | Release={temporal.get('requested_release')}")
        else:
            logger.info("TemporalScope=current")

    if _is_definition_question(message, intent_data) and not _definition_evidence_is_sufficient(evidence, topic_terms):
        rescue_queries = _definition_rescue_queries(message, intent_data)
        if rescue_queries:
            rescue_budget = min(2, max(1, search_budget // 2))
            logger.info(
                f"DefinitionRetrievalRetry=1 | rescue_budget={rescue_budget} | "
                f"queries={rescue_queries[:rescue_budget]}"
            )
            for query in rescue_queries[:rescue_budget]:
                _search(query, log_label="RescueSearch")

            sufficiency = _evidence_sufficiency_stage(
                evidence,
                intent_data,
                message,
                topic_terms,
                require_authoritative=require_authoritative,
                solution_plan=plan,
            )
            if sufficiency["sufficient"]:
                logger.info("DefinitionRetrievalRetry=resolved")
            else:
                logger.warning(
                    "DefinitionRetrievalRetry=insufficient | "
                    f"reason={sufficiency['reason']} | continuing with best available evidence"
                )

    # Always fetch the top doc when we have a document path — search excerpts are
    # capped at MCP_MAX_TOKENS and too thin for synthesis + grounding on their own.
    fetch_budget = max(fetch_budget, FETCH_BUDGET_DEFINITION)

    # Optional fetch: only spend fetch budget on a relevant result that has a real path.
    if fetch_budget > 0 and evidence:
        ranked_for_fetch = _sort_evidence_for_intent(evidence, intent_data, message)
        to_fetch: list[dict] = []
        if _is_architecture_question(intent_data):
            architect_fetch = next(
                (
                    item
                    for item in ranked_for_fetch
                    if item.get("document_path") and _is_architecture_center_evidence(item)
                ),
                None,
            )
            if architect_fetch:
                to_fetch.append(architect_fetch)
            for item in ranked_for_fetch:
                if len(to_fetch) >= fetch_budget:
                    break
                if item in to_fetch or not item.get("document_path"):
                    continue
                if _is_wrong_family_item(message, item, plan):
                    continue
                to_fetch.append(item)
        else:
            if _requires_court_order(message, intent_data):
                ranked_for_fetch = sorted(
                    ranked_for_fetch,
                    key=lambda item: (
                        _seasonal_authority_rank(item, message),
                        _source_trust_score(item),
                    ),
                    reverse=True,
                )
            for item in ranked_for_fetch:
                if len(to_fetch) >= fetch_budget:
                    break
                if item in to_fetch or not item.get("document_path"):
                    continue
                if not _is_relevant(item, topic_terms, intent_data=intent_data):
                    if not (
                        _requires_court_order(message, intent_data)
                        and _is_standing_statute(item)
                    ):
                        continue
                to_fetch.append(item)
        seen_paths: set[str] = set()
        for fetch_source in to_fetch:
            doc_path = fetch_source.get("document_path")
            if not doc_path or doc_path in seen_paths:
                continue
            seen_paths.add(doc_path)
            logger.debug(
                f"FetchSelection | title={fetch_source.get('title')} | "
                f"path={doc_path} | url={fetch_source.get('url')} | "
                f"definition={_is_definition_question(message, intent_data)}"
            )
            logger.info(f"Fetch | hinted path='{doc_path}'")
            fetched = mcp_fetch(doc_path)
            if fetched:
                evidence.append(fetched)
                if _requires_court_order(message, intent_data) and _is_limits_cheatsheet(fetched):
                    for citation in _cited_standing_guides(fetched.get("excerpt") or ""):
                        cite_path = citation.get("document_path")
                        if not cite_path or cite_path in seen_paths:
                            continue
                        logger.info(f"StatuteCitation=follow | path='{cite_path}' | from='{doc_path}'")
                        cited = mcp_fetch(cite_path)
                        if cited:
                            seen_paths.add(cite_path)
                            evidence.append(cited)
                            break
                        title = citation.get("title") or ""
                        if title and not _already_searched(title):
                            _search(title, log_label="StatuteCitationSearch")
                            break

    return evidence

# ---------------------------------------------------------------------------
# Evidence Normalization
# ---------------------------------------------------------------------------

SALESFORCE_AUTHORITATIVE_MARKERS = (
    "salesforce release notes",
    "salesforce developer documentation",
    "salesforce help",
    "salesforce architects",
)
NON_AUTHORITATIVE_CONTENT_MARKERS = (
    "mulesoft",
    "stack exchange",
    "third-party",
)
NON_AUTHORITATIVE_METADATA_MARKERS = (
    "trailhead",
    "salesforce developers blog",
    "blog",
    "community",
)

SALESFORCE_OFFICIAL_URL_MARKERS = (
    "developer.salesforce.com",
    "help.salesforce.com",
    "salesforce.com/docs",
    "docs.salesforce.com",
    "salesforce.com/documentation",
    "architect.salesforce.com",
)


def _evidence_blob(item: dict) -> str:
    return " ".join(
        str(part or "").lower()
        for part in (
            item.get("title"),
            item.get("source_type"),
            item.get("authority"),
            item.get("url"),
            item.get("document_path"),
            item.get("excerpt"),
        )
    )


def _authority_metadata_blob(item: dict) -> str:
    return " ".join(
        str(part or "").lower()
        for part in (
            item.get("title"),
            item.get("source_type"),
            item.get("authority"),
            item.get("url"),
            item.get("document_path"),
        )
    )


def _is_non_authoritative_salesforce_evidence(item: dict) -> bool:
    full_blob = _evidence_blob(item)
    if any(marker in full_blob for marker in NON_AUTHORITATIVE_CONTENT_MARKERS):
        return True
    return any(marker in _authority_metadata_blob(item) for marker in NON_AUTHORITATIVE_METADATA_MARKERS)


def _is_authoritative_salesforce_evidence(item: dict) -> bool:
    blob = _evidence_blob(item)
    if _is_non_authoritative_salesforce_evidence(item):
        return False
    if any(marker in blob for marker in SALESFORCE_AUTHORITATIVE_MARKERS):
        return True
    if any(marker in blob for marker in SALESFORCE_OFFICIAL_URL_MARKERS):
        return True
    if item.get("source_type") == "Salesforce Documentation":
        return True
    if item.get("authority") in SALESFORCE_AUTHORITATIVE_MARKERS:
        return True
    return bool(item.get("is_authoritative") and "salesforce" in blob)


def _filter_current_fact_evidence(evidence: list[dict]) -> list[dict]:
    filtered = [item for item in evidence if _is_authoritative_salesforce_evidence(item)]
    if len(filtered) != len(evidence):
        logger.warning(
            f"Filtered non-authoritative current-fact evidence | kept={len(filtered)} | "
            f"dropped={len(evidence) - len(filtered)}"
        )
    return filtered

def normalize_evidence(evidence: list[dict]) -> list[dict]:
    """Ensure all evidence items have consistent keys."""
    normalized = []
    for item in evidence:
        if not item.get("excerpt"):
            continue
        normalized.append({
            "title": item.get("title") or None,
            "url": item.get("url") or None,
            "document_path": item.get("document_path") or None,
            "excerpt": item.get("excerpt", ""),
            "source_type": item.get("source_type") or None,
            "authority": item.get("authority") or None,
            "is_authoritative": bool(item.get("is_authoritative", False)),
            "release": item.get("release") or None,
            "api_version": item.get("api_version") or None,
            "published_date": item.get("published_date") or None,
            "last_updated": item.get("last_updated") or None,
            "query": item.get("query") or None,
            "relevance": item.get("relevance"),
        })
    return normalized


def _source_trust_score(item: dict) -> int:
    blob = _evidence_blob(item)
    metadata_blob = _authority_metadata_blob(item)
    if any(marker in blob for marker in NON_AUTHORITATIVE_CONTENT_MARKERS):
        return 10
    if any(marker in metadata_blob for marker in NON_AUTHORITATIVE_METADATA_MARKERS):
        return 10
    if _is_release_notes_evidence(item):
        return 140 if item.get("is_authoritative") else 125
    if _is_limits_cheatsheet(item):
        return 80 if item.get("is_authoritative") else 70
    if any(marker in blob for marker in SALESFORCE_AUTHORITATIVE_MARKERS):
        return 110 if item.get("is_authoritative") else 100
    if any(marker in blob for marker in SALESFORCE_OFFICIAL_URL_MARKERS):
        return 105 if item.get("is_authoritative") else 95
    if item.get("source_type") == "Salesforce Documentation":
        return 100 if item.get("is_authoritative") else 90
    if "salesforce" in blob:
        return 65 if item.get("is_authoritative") else 60
    return 40


def _has_authoritative_salesforce_source(evidence: list[dict]) -> bool:
    return any(_is_authoritative_salesforce_evidence(item) or _source_trust_score(item) >= 90 for item in evidence)


def _sort_evidence_by_trust(evidence: list[dict]) -> list[dict]:
    return sorted(evidence, key=_source_trust_score, reverse=True)


def _is_current_fact_request(intent_data: dict, temporal_context: dict) -> bool:
    if temporal_context.get("historical_release_requested"):
        return False
    if temporal_context.get("current_docs_required"):
        return True
    return intent_data.get("question_type") == "current_fact" or intent_data.get("intent") in {"release", "limits"}


def _build_unverified_current_fact_response(message: str, evidence: list[dict], temporal_context: dict) -> str:
    source_lines = []
    for item in _sort_evidence_by_trust(evidence):
        if not _is_authoritative_salesforce_evidence(item):
            continue
        title = item.get("title") or item.get("authority") or "Salesforce source"
        source_type = item.get("source_type") or item.get("authority") or "Salesforce source"
        source_lines.append(f"- {title} ({source_type})")

    source_block = "\n".join(source_lines) if source_lines else "- No authoritative Salesforce source was found."

    return (
        "## Answer\n"
        "[Unverified] I could not confirm the current Salesforce value from authoritative Salesforce documentation.\n\n"
        "## Note\n"
        "The retrieved evidence did not include a Salesforce Release Notes or Salesforce Developer Documentation source that I could treat as authoritative for this current fact.\n\n"
        "## Sources\n"
        f"{source_block}"
    )


def _format_evidence_summary(evidence: list[dict]) -> str:
    lines = []
    for i, item in enumerate(evidence, 1):
        lines.append(
            f"Source {i}: title={item.get('title')}, path={item.get('document_path')}, "
            f"authority={item.get('authority')}, release={item.get('release')}, "
            f"api_version={item.get('api_version')}, published_date={item.get('published_date')}, "
            f"last_updated={item.get('last_updated')}"
        )
        lines.append(f"Excerpt: {item.get('excerpt', '')[:700]}")
    return "\n".join(lines)


def resolve_evidence_conflicts(evidence: list[dict], intent_data: dict, llm, message: str) -> dict:
    if llm is None or len(evidence) < 2:
        return {
            "status": "no_conflict",
            "applicable_fact": None,
            "conflicts": [],
        }

    try:
        prompt_input = (
            f"question={json.dumps(message)}, "
            f"intent={json.dumps(intent_data.get('intent'))}, "
            f"requested_release={json.dumps(intent_data.get('requested_release'))}, "
            f"evidence=\n{_format_evidence_summary(evidence)}"
        )
        resp = _invoke_resilient(llm, [
            SystemMessage(content=EVIDENCE_CONFLICT_RESOLUTION_PROMPT),
            HumanMessage(content=prompt_input),
        ], "conflict", message)
        parsed = _extract_json_payload(resp.content)
        status = parsed.get("status", "unresolved")
        if status not in {"resolved", "unresolved", "no_conflict"}:
            status = "unresolved"
        return {
            "status": status,
            "applicable_fact": parsed.get("applicable_fact"),
            "conflicts": parsed.get("conflicts", []),
        }
    except Exception as e:
        logger.warning(f"Conflict resolution failed: {e}")
        return {
            "status": "unresolved",
            "applicable_fact": None,
            "conflicts": [],
        }

# ---------------------------------------------------------------------------
# System Prompt Composition
# ---------------------------------------------------------------------------

def build_synthesis_prompt(
    message: str,
    intent_data: dict,
    current_fact_verified: bool = False,
    solution_plan: dict | None = None,
) -> str:
    parts = [RESEARCH_SYNTHESIS_PROMPT]
    if current_fact_verified:
        parts.append(CURRENT_FACT_DIRECT_ANSWER_PROMPT)
    if _is_definition_question(message, intent_data):
        parts.append(DEFINITION_FIRST_SYNTHESIS_PROMPT)
    intent = intent_data.get("intent", "general")

    if intent == "architecture" or intent_data.get("requires_architecture_analysis"):
        parts.append(ARCHITECTURE_REASONING_PROMPT)
        plan = solution_plan if solution_plan is not None else sa.heuristic_plan(message)
        if plan.get("full_protocol"):
            parts.append(SOLUTION_ARCHITECTURE_PROTOCOL)
        elif sa.is_solution_question(message):
            parts.append(SOLUTION_ARCHITECTURE_PROTOCOL_SHORT)

    if intent in ("code_review", "troubleshooting") or intent_data.get("requires_code_analysis"):
        parts.append(CODE_REVIEW_PROMPT)

    lower_message = message.lower()
    if any(token in lower_message for token in ("soql 101", "too many soql", "bulkify", "governor limit", "too many soql queries")):
        parts.append(SOQL_101_HINT_PROMPT)

    return "\n\n---\n\n".join(parts)

# ---------------------------------------------------------------------------
# Final Answer Generation (LangGraph — no tools, synthesis only)
# ---------------------------------------------------------------------------

memory_checkpoint = MemorySaver()

# Per-session thread IDs so multiple Gradio users don't share the same memory.
# Key: Gradio session hash → thread_id string.
_SESSION_THREADS: dict[int, str] = {}
_SESSION_COUNTER = 0


def _get_thread_id(history: list) -> str:
    """Return a stable thread_id for this conversation history object."""
    global _SESSION_COUNTER
    key = id(history)
    if key not in _SESSION_THREADS:
        _SESSION_COUNTER += 1
        _SESSION_THREADS[key] = f"session_{_SESSION_COUNTER}"
    return _SESSION_THREADS[key]


def generate_answer(
    message: str,
    evidence: list[dict],
    intent_data: dict,
    conflict_summary: dict,
    temporal_context: dict,
    llm,
    thread_id: str = "default",
    solution_plan: dict | None = None,
) -> str:
    synthesis_evidence = evidence
    if _is_current_fact_request(intent_data, temporal_context):
        synthesis_evidence = _filter_current_fact_evidence(evidence)
    ordered_evidence = _sort_evidence_for_intent(synthesis_evidence, intent_data, message)
    current_fact_verified = _is_current_fact_request(intent_data, temporal_context) and _has_authoritative_salesforce_source(ordered_evidence)
    system_prompt = build_synthesis_prompt(
        message,
        intent_data,
        current_fact_verified=current_fact_verified,
        solution_plan=solution_plan,
    )

    if ordered_evidence:
        ev_lines = ["## Retrieved Evidence\n"]
        for i, item in enumerate(ordered_evidence, 1):
            ev_lines.append(f"### Source {i}")
            if item.get("query"):
                ev_lines.append(f"Search query: {item['query']}")
            if item.get("document_path"):
                ev_lines.append(f"Document path: {item['document_path']}")
            if item.get("url"):
                ev_lines.append(f"URL: {item['url']}")
            ev_lines.append(item["excerpt"])
            ev_lines.append("")
        if temporal_context.get("current_docs_required") and not _has_authoritative_salesforce_source(ordered_evidence):
            ev_lines.insert(
                1,
                "No authoritative Salesforce Release Notes or Salesforce Developer Documentation source was identified for this current fact.",
            )
        evidence_block = "\n".join(ev_lines)
    else:
        evidence_block = (
            "## Retrieved Evidence\n"
            "[Retrieval failed or not required] No Salesforce documentation was retrieved.\n"
            "If this is a Salesforce question, label all claims as [Unverified]."
        )

    temporal_block = (
        "## Temporal Resolution\n"
        f"Temporal validation required: {temporal_context.get('temporal_validation_required')}\n"
        f"Current docs required: {temporal_context.get('current_docs_required')}\n"
        f"Historical release requested: {temporal_context.get('historical_release_requested')}\n"
        f"Requested release: {temporal_context.get('requested_release')}\n"
        f"Authoritative Salesforce source found: {_has_authoritative_salesforce_source(ordered_evidence)}\n"
        f"Conflict status: {conflict_summary.get('status')}\n"
        f"Applicable fact: {conflict_summary.get('applicable_fact')}\n"
        f"Conflicts: {json.dumps(conflict_summary.get('conflicts', []))}\n"
        + (
            "IMPORTANT: This question asks about a governor limit or platform limit that changes "
            "release-to-release. Salesforce Release Notes are the court order. The current "
            "developer guide is the statute. Cheatsheets are a digest and can lag both. "
            "Prefer release notes when they conflict. Prefer the current guide over a cheatsheet. "
            "If the evidence does not include release notes, explicitly note that "
            "the user should verify the current value in the latest Salesforce Release Notes at "
            "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm"
            if temporal_context.get("current_docs_required") and _is_volatile_limit_question(message)
            else "If this is a current fact question and no authoritative Salesforce source is available, answer [Unverified] instead of selecting a value from non-authoritative evidence."
        )
    )

    intent_block = (
        f"## Intent\n"
        f"Intent: {intent_data.get('intent')}\n"
        f"Question type: {intent_data.get('question_type')}\n"
        f"Topics: {', '.join(intent_data.get('topics', []))}\n"
        f"Features: {', '.join(intent_data.get('salesforce_features', []))}"
    )
    plan = solution_plan or {}
    if plan.get("business_capability") or plan.get("technical_requirements"):
        extra = ["\n\n## Solution architect intake"]
        if plan.get("business_capability"):
            extra.append(f"Business capability: {plan['business_capability']}")
        if plan.get("full_protocol"):
            extra.append(
                "Use the capability-first protocol. Do not pick a Salesforce product "
                "before discovery. If discovery questions are unanswered, document "
                "an ADR-style decision deferred."
            )
        if plan.get("separate_detection_from_policy"):
            extra.append(
                "Separate probabilistic detection (signals) from deterministic "
                "business policy (risk/severity decision)."
            )
        if plan.get("in_scope"):
            extra.append("In scope: " + "; ".join(plan["in_scope"]))
        if plan.get("out_of_scope"):
            extra.append("Out of scope: " + "; ".join(plan["out_of_scope"]))
        if plan.get("discovery_questions"):
            extra.append("Discovery questions still open:")
            extra.extend(f"- {question}" for question in plan["discovery_questions"])
        if plan.get("technical_requirements"):
            extra.append("Translated technical requirements:")
            extra.extend(f"- {req}" for req in plan["technical_requirements"])
        extra.append(f"Objects: {', '.join(plan.get('objects') or [])}")
        extra.append(f"Capabilities: {', '.join(plan.get('capabilities') or [])}")
        extra.append(
            "Do not retrieve or recommend platform-security products unless the "
            "requirement is org/session security."
        )
        intent_block += "\n".join(extra)

    augmented = f"{message}\n\n{intent_block}\n\n{temporal_block}\n\n{evidence_block}"

    agent = create_react_agent(
        llm,
        tools=[],
        checkpointer=memory_checkpoint,
        prompt=system_prompt,
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 5,
    }
    try:
        result = agent.invoke({"messages": [HumanMessage(content=augmented)]}, config=config)
        return result["messages"][-1].content
    except Exception as e:
        logger.warning(f"Synthesis unavailable | {e}")
        logger.warning("FallbackContentUsed | kind=synthesis | reason=agent_invoke_exception")
        if temporal_context.get("current_docs_required") or intent_data.get("question_type") == "current_fact":
            return _build_unverified_current_fact_response(message, evidence, temporal_context)
        return (
            "## Answer\n"
            "[Unverified] I could not generate a verified answer because the local model provider was unavailable.\n\n"
            "## Sources\n"
            "- No verified synthesis could be produced."
        )

# ---------------------------------------------------------------------------
# Grounding Check
# ---------------------------------------------------------------------------

_GROUNDING_UNVERIFIED_NOTE = (
    "\n\n---\n**⚠️ Grounding unverified:** The automated grounding check could not "
    "confirm this answer against the retrieved documentation. Treat the specifics "
    "as [Unverified] and confirm against the linked official Salesforce sources."
)


def _append_grounding_unverified_note(answer: str) -> str:
    if "Grounding unverified" in answer:
        return answer
    return answer + _GROUNDING_UNVERIFIED_NOTE


def validate_grounding(answer: str, evidence: list[dict], llm, message: str = "", intent_data: dict | None = None) -> tuple[str, str]:
    """
    Lightweight grounding check.
    Returns (validated_answer, grounding_status).
    Skipped when there is no evidence (nothing to ground against).
    """
    intent_data = intent_data or {}
    definition_intent = _is_definition_question(message, intent_data)
    if not evidence:
        return answer, "skipped (no evidence)"

    evidence_summary = "\n".join(
        f"Source {i}: {item.get('excerpt', '')}"
        for i, item in enumerate(evidence, 1)
    )
    check_input = (
        f"## Answer Draft\n{answer}\n\n"
        f"## Evidence\n{evidence_summary}"
    )
    try:
        resp = _invoke_resilient(llm, [
            SystemMessage(content=GROUNDING_CHECK_PROMPT),
            HumanMessage(content=check_input),
        ], "grounding", answer)
        result = resp.content.strip()
        if "GROUNDING: passed" in result:
            logger.info("GroundingCheck=passed")
            return answer, "passed"
        # Fail closed: the checker was unavailable (fallback sentinel) — do not
        # emit the draft as verified.
        if "GROUNDING: unverified" in result:
            logger.warning("GroundingCheck=unverified | checker unavailable — failing closed")
            return _append_grounding_unverified_note(answer), "unverified"
        # Parse unsupported claims and append warnings
        warnings = []
        for line in result.split("\n"):
            if line.startswith("UNSUPPORTED:"):
                claim = line.replace("UNSUPPORTED:", "").strip()
                warnings.append(f"[Unverified] {claim}")
        if warnings:
            logger.info(f"UnsupportedClaims: {warnings}")
            logger.warning(f"GroundingCheck=flagged | {len(warnings)} unsupported claim(s)")
            if definition_intent:
                warning_block = (
                    "\n\n---\n**Grounding note:** The following claims could not be verified against the retrieved documentation:\n"
                    + "\n".join(f"- {w}" for w in warnings)
                )
                logger.info("GroundingCheck=preserved_definition_spine")
                return answer + warning_block, "flagged"
            rewrite_input = (
                f"## Draft Answer\n{answer}\n\n"
                f"## Evidence\n{evidence_summary}\n\n"
                f"## Unsupported Claims\n" + "\n".join(f"- {w}" for w in warnings)
            )
            try:
                rewrite_resp = _invoke_resilient(llm, [
                    SystemMessage(content=GROUNDING_REWRITE_PROMPT),
                    HumanMessage(content=rewrite_input),
                ], "grounding_rewrite", answer)
                rewritten = rewrite_resp.content.strip()
                if rewritten:
                    logger.info("GroundingCheck=rewritten")
                    return rewritten, "rewritten"
            except Exception as rewrite_error:
                logger.warning(f"GroundingRewrite failed: {rewrite_error}")
            warning_block = "\n\n---\n**Grounding note:** The following claims could not be verified against the retrieved documentation:\n" + "\n".join(f"- {w}" for w in warnings)
            return answer + warning_block, "flagged"
        # Unrecognized grounding output (no "passed", no UNSUPPORTED lines). A
        # gate whose unknown case passes is not a gate — fail closed. Definitions
        # keep their spine; everything else gets an explicit unverified note.
        logger.warning("GroundingCheck=unrecognized_output | failing closed")
        if definition_intent:
            return answer, "flagged"
        return _append_grounding_unverified_note(answer), "unverified"
    except Exception as e:
        logger.warning(f"GroundingCheck failed: {e} — failing closed")
        return _append_grounding_unverified_note(answer), "unverified"

def _public_solution_plan(plan: dict) -> dict:
    return {
        "objects": plan.get("objects") or [],
        "actors": plan.get("actors") or [],
        "capabilities": plan.get("capabilities") or [],
        "technical_requirements": plan.get("technical_requirements") or [],
        "avoid_platform_security": bool(plan.get("avoid_platform_security")),
        "skip_raw": bool(plan.get("skip_raw")),
        "full_protocol": bool(plan.get("full_protocol")),
        "business_capability": plan.get("business_capability") or "",
        "in_scope": plan.get("in_scope") or [],
        "out_of_scope": plan.get("out_of_scope") or [],
        "discovery_questions": plan.get("discovery_questions") or [],
        "separate_detection_from_policy": bool(plan.get("separate_detection_from_policy")),
    }


def _host_instructions_for_plan(plan: dict) -> str:
    host = (
        "Synthesize an answer using only this evidence. "
        "Label undocumented Salesforce facts [Unverified]. "
        "Preserve source URLs. Do not invent documentation. "
        "If intent is architecture, use Trusted / Easy / Adaptable only when those "
        "terms appear in the evidence."
    )
    if not plan.get("skip_raw"):
        return host
    parts = [
        "Think like a Salesforce Solution Architect. Do not start from a product.",
        "Recommend against the translated capability layers, not the original jargon.",
    ]
    if plan.get("business_capability"):
        parts.append(f"Business capability: {plan['business_capability']}")
    if plan.get("full_protocol"):
        parts.append(
            "Follow the capability-first sequence: capability, taxonomy/decisions "
            "and cost of being wrong, detection vs deterministic policy, input "
            "boundary, then Salesforce options. Do not pick a product before "
            "discovery. If discovery_questions are unanswered, write an ADR-style "
            "decision deferred."
        )
    if plan.get("separate_detection_from_policy"):
        parts.append(
            "Keep probabilistic detection (signals) separate from deterministic "
            "business policy."
        )
    if plan.get("discovery_questions"):
        parts.append("Discovery questions: " + " | ".join(plan["discovery_questions"]))
    if plan.get("technical_requirements"):
        parts.append("Technical requirements: " + "; ".join(plan["technical_requirements"]))
    parts.append(
        "Do not recommend org-security products (Security Center, Shield Threat "
        "Detection, Event Monitoring) unless the requirement is platform security. "
        "Label undocumented implementation details [Inference] or [Unverified]. "
        "Preserve source URLs."
    )
    return " ".join(parts)


def build_evidence_pack(message: str, llm=None) -> dict:
    """Retrieve grounded Salesforce evidence without requiring synthesis."""
    should_continue, early_response = input_guardrail(message)
    if not should_continue:
        return {
            "question": message,
            "blocked": True,
            "answer": early_response,
            "intent": {},
            "temporal": {},
            "evidence": [],
            "mode": "blocked",
        }
    intent_data = classify_intent(message, llm)
    temporal_context = validate_temporal_context(message, intent_data, llm)
    plan = translate_requirement(message, intent_data, llm)
    raw_evidence = retrieve_evidence(
        message, intent_data, llm, temporal_context, solution_plan=plan
    )
    evidence = normalize_evidence(raw_evidence)
    topic_terms = _topic_terms(message, intent_data, plan)
    relevant = _filter_relevant_evidence(
        evidence, topic_terms, intent_data, plan, message=message
    )
    if relevant:
        evidence = relevant
    # Independent backstop: if the pack does not actually cover the question,
    # flag it and withhold the "synthesize" directive so the host does not
    # produce a fluent-but-wrong answer from off-topic authoritative docs.
    off_topic = _pack_is_off_topic(message, intent_data, evidence)
    if off_topic:
        logger.warning(
            "PackOffTopic=true | evidence did not cover the question topic | "
            f"question='{message[:120]}'"
        )
        host = (
            "The retrieved evidence below did NOT match the question topic. Do not "
            "synthesize a substantive answer from it and do not fill gaps from "
            "memory. Tell the user no on-topic official documentation was found and "
            "suggest they name the exact API, object, or feature. Preserve source URLs."
        )
    else:
        host = _host_instructions_for_plan(plan)
    claim_analysis = claim_risk.analyze_evidence_claims(
        message,
        evidence,
        is_release_notes=_is_release_notes_evidence,
    )
    verification = claim_risk.verification_from_claim_risk(claim_analysis)
    if not off_topic and (
        _needs_seasonal_authority(message, intent_data, temporal_context)
        or claim_analysis.get("requires_current_release")
    ):
        host += (
            " Seasonal facts: Salesforce ships three releases a year. Release notes "
            "are the court order. The current developer guide is the statute. "
            "Limits cheatsheets and older help articles are a digest and often lag. "
            "If they conflict on a limit or newly GA feature, prefer the release notes. "
            "If the pack has a digest without release notes, do not treat the digest "
            "value as current — state the lag and point at the notes."
        )
    if not off_topic:
        host += claim_risk.host_instructions_for_claim_risk(claim_analysis, verification)
    return {
        "question": message,
        "blocked": False,
        "intent": intent_data,
        "temporal": temporal_context,
        "solution_plan": _public_solution_plan(plan),
        "evidence": evidence,
        "claims": claim_analysis.get("claims") or [],
        "claim_risk": {
            "requires_current_release": bool(claim_analysis.get("requires_current_release")),
            "has_release_notes_evidence": bool(
                claim_analysis.get("has_release_notes_evidence")
            ),
            "volatile_claim_count": len(claim_analysis.get("volatile_claims") or []),
            "volatile_claims": claim_analysis.get("volatile_claims") or [],
        },
        "verification": verification,
        "relevance": "off_topic" if off_topic else "on_topic",
        "mode": "full" if llm is not None else "retrieval_only",
        "host_instructions": host,
    }


def format_retrieval_only_answer(
    message: str,
    intent_data: dict,
    evidence: list[dict],
    temporal_context: dict,
    solution_plan: dict | None = None,
) -> str:
    lines = [
        "[retrieval-only] No LLM is configured on this machine. The Python agent still retrieved Salesforce documentation.",
        "",
        "Connect a provider with `python cli.py status`, paste a key in the UI, or ask this in Cursor so the local agent can synthesize from the pack below.",
        "",
        f"Intent: {intent_data.get('intent')} | depth: {intent_data.get('research_depth')} | architecture: {bool(intent_data.get('requires_architecture_analysis'))}",
        f"Current docs required: {temporal_context.get('current_docs_required')}",
        "",
    ]
    plan = solution_plan or {}
    if plan.get("business_capability"):
        lines.append(f"Business capability: {plan['business_capability']}")
        lines.append("")
    if plan.get("full_protocol") and plan.get("discovery_questions"):
        lines.append("Discovery questions (answer these before committing to a product):")
        for question in plan["discovery_questions"]:
            lines.append(f"- {question}")
        lines.append("")
    if plan.get("in_scope"):
        lines.append("In scope: " + "; ".join(plan["in_scope"]))
    if plan.get("out_of_scope"):
        lines.append("Out of scope: " + "; ".join(plan["out_of_scope"]))
        lines.append("")
    if plan.get("technical_requirements"):
        lines.append("Translated technical requirements:")
        for req in plan["technical_requirements"]:
            lines.append(f"- {req}")
        if plan.get("objects"):
            lines.append(f"Objects: {', '.join(plan['objects'])}")
        if plan.get("capabilities"):
            lines.append(f"Capabilities: {', '.join(plan['capabilities'])}")
        lines.append("")
    if not evidence:
        lines.append("No documentation was retrieved. Check MCP_URL and network access.")
        return "\n".join(lines)
    for index, item in enumerate(evidence, 1):
        title = item.get("title") or item.get("authority") or "Salesforce source"
        url = item.get("url") or ""
        excerpt = (item.get("excerpt") or "")[:900]
        lines.append(f"### {index}. {title}")
        if url:
            lines.append(url)
        lines.append(excerpt)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_agent(message: str, history: list) -> str:
    message = message.strip().strip("'\"")
    logger.info(f"Query='{message[:120]}'")

    # Trajectory: structured record of each pipeline step for evaluation.
    # Inspired by Google agent guide — evaluate decisions, not just final output.
    trajectory = {
        "query": message,
        "steps": [],
    }

    def _record(step: str, outcome: str, **kw):
        entry = {"step": step, "outcome": outcome}
        entry.update(kw)
        trajectory["steps"].append(entry)
        logger.debug(f"Trajectory | step={step} | outcome={outcome} | {kw}")

    # Step 0 — input guardrail (rules-based, no LLM cost)
    should_continue, early_response = input_guardrail(message)
    if not should_continue:
        _record("guardrail", "blocked")
        logger.info(f"Trajectory | {trajectory}")
        return early_response
    _record("guardrail", "passed")

    # Step 1 — get a fast LLM for classification and query planning
    provider_label, classifier_llm = get_llm("fast")
    retrieval_only = classifier_llm is None
    if retrieval_only:
        logger.warning("NoLLMConfigured | running retrieval-only Python agent")
        intent_data = classify_intent_heuristic(message)
        _record("classify", "heuristic", intent=intent_data.get("intent"))
        temporal_context = validate_temporal_context(message, intent_data, None)
        _record("temporal", "heuristic")
        plan = translate_requirement(message, intent_data, None)
        raw_evidence = retrieve_evidence(
            message, intent_data, None, temporal_context, solution_plan=plan
        )
        evidence = normalize_evidence(raw_evidence)
        _record("retrieve", "ok", evidence_count=len(evidence))
        topic_terms = _topic_terms(message, intent_data, plan)
        relevant_evidence = _filter_relevant_evidence(
            evidence, topic_terms, intent_data, plan, message=message
        )
        if relevant_evidence:
            evidence = relevant_evidence
        if _pack_is_off_topic(message, intent_data, evidence):
            logger.warning(
                "PackOffTopic=true | retrieval-only pack did not match question "
                f"topic | question={message.strip()[:120]!r}"
            )
            _record("relevance_filter", "off_topic")
            logger.info(f"Trajectory | {trajectory}")
            return _build_off_topic_pack_response(message, evidence)
        logger.info(f"Trajectory | {trajectory}")
        return format_retrieval_only_answer(
            message, intent_data, evidence, temporal_context, plan
        )

    # Step 2 — classify intent
    intent_data = classify_intent(message, classifier_llm)
    _record("classify", "ok",
            intent=intent_data.get("intent"),
            tier=intent_data.get("model_tier"),
            depth=intent_data.get("research_depth"),
            requires_docs=intent_data.get("requires_documentation"))

    # Step 3 — temporal validation + deterministic bounded retrieval
    temporal_context = validate_temporal_context(message, intent_data, classifier_llm)
    _record("temporal", "ok",
            temporal_required=temporal_context.get("temporal_validation_required"),
            current_docs=temporal_context.get("current_docs_required"))

    plan = translate_requirement(message, intent_data, classifier_llm)
    raw_evidence = retrieve_evidence(
        message, intent_data, classifier_llm, temporal_context, solution_plan=plan
    )
    evidence = normalize_evidence(raw_evidence)
    for i, item in enumerate(evidence, 1):
        if DEBUG_FULL_EXCERPTS:
            logger.debug(
                f"RawEvidence #{i} | title={item.get('title')!r} | url={item.get('url')!r} | "
                f"source_type={item.get('source_type')!r} | is_authoritative={item.get('is_authoritative')} | "
                f"document_path={item.get('document_path')!r} | excerpt_full={item.get('excerpt', '')!r}"
            )
        else:
            logger.debug(
                f"RawEvidence #{i} | title={item.get('title')!r} | url={item.get('url')!r} | "
                f"source_type={item.get('source_type')!r} | is_authoritative={item.get('is_authoritative')} | "
                f"document_path={item.get('document_path')!r} | excerpt_head={item.get('excerpt', '')[:300]!r}"
            )
    _record("retrieve", "ok", evidence_count=len(evidence))

    topic_terms = _topic_terms(message, intent_data, plan)
    relevant_evidence = _filter_relevant_evidence(
        evidence, topic_terms, intent_data, plan, message=message
    )
    if evidence and topic_terms and not relevant_evidence:
        logger.warning("TopicRelevanceUnverified=true | no retrieved evidence matched topic terms")
        _record("relevance_filter", "no_match")
        logger.info(f"Trajectory | {trajectory}")
        return _build_unverified_relevance_response(message, evidence)
    if relevant_evidence:
        evidence = relevant_evidence
    # Independent backstop: the filter above derives terms from the (possibly
    # drifted) plan, so it can validate off-topic evidence against itself. This
    # check compares the pack to the LITERAL question and blocks synthesis when
    # they do not match — catching planner/classifier drift the filter misses.
    if _pack_is_off_topic(message, intent_data, evidence):
        logger.warning(
            "PackOffTopic=true | pack did not match question topic — refusing "
            f"synthesis | question={message.strip()[:120]!r}"
        )
        _record("relevance_filter", "off_topic")
        logger.info(f"Trajectory | {trajectory}")
        return _build_off_topic_pack_response(message, evidence)
    if _is_current_fact_request(intent_data, temporal_context):
        evidence = _filter_current_fact_evidence(evidence)
    _record("relevance_filter", "ok", kept=len(evidence))

    # Step 4 — conflict resolution on normalized evidence
    conflict_summary = resolve_evidence_conflicts(evidence, intent_data, classifier_llm, message)
    _record("conflict", conflict_summary.get("status", "unknown"))

    # Fail closed for current facts when we do not have an authoritative Salesforce source.
    if _is_current_fact_request(intent_data, temporal_context) and not _has_authoritative_salesforce_source(evidence):
        logger.warning("CurrentFactUnverified=true | no authoritative Salesforce source found")
        _record("current_fact_gate", "blocked")
        logger.info(f"Trajectory | {trajectory}")
        return _build_unverified_current_fact_response(message, evidence, temporal_context)
    _record("current_fact_gate", "passed")

    # Step 5 — select model tier for synthesis
    tier = intent_data.get("model_tier", "standard")
    provider_label, synthesis_llm = get_llm(tier)
    trajectory["provider_used"] = provider_label
    if not synthesis_llm:
        trajectory["fallback_reason"] = "no_synthesis_provider"
        _record(
            "generate",
            "fallback",
            provider=provider_label,
            provider_used=provider_label,
            fallback_reason="no_synthesis_provider",
        )
        logger.info(f"Trajectory | {trajectory}")
        return "❌ All providers failed. Please configure your API keys or verify Ollama is active."

    # Step 6 — generate answer (session-isolated thread_id)
    thread_id = _get_thread_id(history)
    logger.info(f"FinalGeneration | Provider={provider_label} | Tier={tier} | Thread={thread_id}")
    try:
        answer = generate_answer(
            message,
            evidence,
            intent_data,
            conflict_summary,
            temporal_context,
            synthesis_llm,
            thread_id,
            solution_plan=plan,
        )
    except Exception as e:
        logger.exception(f"Generation failed | {e}")
        trajectory["fallback_reason"] = "generation_error"
        logger.info(f"Trajectory | {trajectory}")
        return f"❌ Generation error: {e}"
    fallback_reason = "generation_fallback_content" if _is_fallback_content(answer) else None
    trajectory["fallback_reason"] = fallback_reason
    if fallback_reason:
        logger.warning("ResponseUsedFallbackContent=true | kind=synthesis")
    _record(
        "generate",
        "fallback" if fallback_reason else "ok",
        provider=provider_label,
        provider_used=provider_label,
        fallback_reason=fallback_reason,
    )

    # Step 7 — grounding check (lightweight, uses fast LLM)
    answer, grounding_status = validate_grounding(answer, evidence, classifier_llm, message, intent_data)
    _record("grounding", grounding_status)

    # Post-processing: for volatile limit questions, always append a release notes
    # verification note — the MCP index may not have the latest release notes yet.
    if _is_volatile_limit_question(message) and temporal_context.get("current_docs_required"):
        rn_url = "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm"
        has_rn = any(
            "release-notes" in (item.get("url") or "").lower() or
            "release-notes" in (item.get("document_path") or "").lower()
            for item in evidence
        )
        if not has_rn:
            answer += (
                f"\n\n> **⚠️ Verify against latest release notes:** Governor limits change "
                f"release-to-release. Confirm the current value at "
                f"[Salesforce Release Notes]({rn_url})."
            )
            logger.info("VolatileLimitCaveat=appended | no release notes in evidence")

    logger.success(
        f"ResponseComplete | Provider={provider_label} | "
        f"Intent={intent_data.get('intent')} | "
        f"EvidenceCount={len(evidence)} | "
        f"ConflictStatus={conflict_summary.get('status')} | "
        f"TemporalValidation={temporal_context.get('temporal_validation_required')} | "
        f"Grounding={grounding_status} | "
        f"SynthesisFallback={_is_fallback_content(answer)} | "
        f"SessionCostCents={_session_cost(thread_id):.4f}"
    )
    logger.info(f"Trajectory | {trajectory}")

    return f"[{provider_label}] {answer}"

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else "What is a Platform Event?"
    print(run_agent(question, []))
