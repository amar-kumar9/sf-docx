import os
import json
import re
import time
import requests
import tiktoken
from dotenv import load_dotenv
load_dotenv()

from loguru import logger
import gradio as gr

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from prompts import (
    INTENT_CLASSIFIER_PROMPT,
    QUERY_PLANNER_PROMPT,
    TEMPORAL_FACT_VALIDATION_PROMPT,
    EVIDENCE_CONFLICT_RESOLUTION_PROMPT,
    DEFINITION_FIRST_SYNTHESIS_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    ARCHITECTURE_REASONING_PROMPT,
    CODE_REVIEW_PROMPT,
    GROUNDING_CHECK_PROMPT,
    GROUNDING_REWRITE_PROMPT,
)

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
# Definition questions always fetch the top doc regardless of depth tier.
FETCH_BUDGET_DEFINITION = 1
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

# Facts that change release-to-release — LLM classifier must NOT override these.
# The static docs are always stale for these; release notes must be consulted.
_VOLATILE_LIMIT_PATTERNS = re.compile(
    r"\b(heap\s*size|governor\s*limit|platform\s*limit|apex\s*limit|soql\s*limit|"
    r"dml\s*limit|cpu\s*limit|callout\s*limit|api\s*version|api\s*limit|"
    r"bulk\s*api\s*limit|max\s*query|query\s*limit|heap\s*limit)\b",
    re.IGNORECASE,
)


def _is_volatile_limit_question(message: str) -> bool:
    """Returns True if the question asks about a Salesforce limit that changes per release."""
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


def _parse_mcp_document_payload(raw: str) -> dict:
    parsed = {
        "title": "",
        "url": "",
        "document_path": "",
        "excerpt": raw,
    }
    try:
        data = json.loads(raw)
    except Exception:
        return parsed

    if not isinstance(data, dict):
        return parsed

    chunks = data.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return parsed

    contents = []
    first_chunk = chunks[0] if isinstance(chunks[0], dict) else {}
    for chunk in chunks:
        if isinstance(chunk, dict):
            content = chunk.get("content")
            if content:
                contents.append(str(content))

    parsed["excerpt"] = _smart_truncate("\n\n".join(contents) if contents else raw, MCP_MAX_TOKENS)
    parsed["document_path"] = (
        str(first_chunk.get("documentPath") or first_chunk.get("document_path") or "").strip()
    )
    parsed["title"] = (
        str(first_chunk.get("title") or first_chunk.get("name") or parsed["document_path"] or "").strip()
    )
    parsed["url"] = str(first_chunk.get("url") or first_chunk.get("link") or "").strip()
    return parsed


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
        parsed["excerpt"] = _smart_truncate(str(content), MCP_MAX_TOKENS)
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
    if kind == "conflict":
        return json.dumps({
            "status": "no_conflict",
            "applicable_fact": None,
            "conflicts": [],
        })
    if kind == "grounding":
        return "GROUNDING: passed"
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


def validate_temporal_context(message: str, intent_data: dict, llm) -> dict:
    fallback = _determine_temporal_context(message, intent_data)
    definition_question = _is_definition_question(message, intent_data)
    lower = message.lower()
    explicit_current_scope = "current" in lower or "latest" in lower
    historical_release = _find_historical_release(message)
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


def mcp_search(query: str) -> dict:
    logger.info(f"Search | query='{query}'")
    raw, ok = _call_mcp("salesforce_docs_search", {"query": query, "limit": 3})
    if not ok or not raw:
        logger.warning(f"Search empty | query='{query}'")
        return {}
    parsed = _parse_mcp_document_payload(raw)
    return {
        "query": query,
        "excerpt": parsed["excerpt"],
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": parsed["title"],
        "url": parsed["url"],
        "document_path": parsed["document_path"],
        "relevance": None,
    }


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
# Provider / Model Selection
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        requests.get(OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _make_groq(model: str = None):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model or GROQ_MODEL,
        groq_api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
        max_tokens=2048,
    )


def _make_ollama(model: str = None):
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model or OLLAMA_MODEL, temperature=0)


# MCP tool risk levels (per OpenAI agent guide — tools should be rated by risk).
# All MCP tools here are READ-ONLY (no writes, no side effects) → LOW risk.
# If write tools are added in future, rate them MEDIUM or HIGH and add confirmation.
MCP_TOOL_RISK = {
    "salesforce_docs_search": "low",   # read-only semantic search
    "salesforce_docs_fetch":  "low",   # read-only document fetch
}


def get_llm(tier: str = "standard") -> tuple[str, object]:
    """
    Return (label, llm) for the requested tier.
    Priority: Groq first (capable), Ollama as fallback (local).
    Tiers: fast | standard | reasoning
    """
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    ollama_up = _ollama_available()

    # Groq first for all tiers — it follows prompt instructions correctly.
    # Ollama is fallback only (local models too small for conditional format rules).
    if has_groq:
        try:
            llm = _make_groq()
            logger.info(f"ModelTier={tier} | Provider=Groq | Model={GROQ_MODEL}")
            return f"Groq/{GROQ_MODEL}", llm
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    if ollama_up:
        model = OLLAMA_MODEL_FAST if tier == "fast" else OLLAMA_MODEL
        try:
            llm = _make_ollama(model)
            logger.info(f"ModelTier={tier} | Provider=Ollama | Model={model}")
            return f"Ollama/{model}", llm
        except Exception as e:
            logger.warning(f"Ollama init failed: {e}")

    logger.error("All providers unavailable")
    return None, None

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
    r"heroku|slack|einstein|omni.?channel|service cloud|sales cloud|experience cloud)\b",
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
                return _SAFE_FALLBACK_INTENT
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
        return _SAFE_FALLBACK_INTENT
    except Exception as e:
        logger.warning(f"Intent classification failed: {e} — using safe fallback")
        return _SAFE_FALLBACK_INTENT

# ---------------------------------------------------------------------------
# Search Query Planning
# ---------------------------------------------------------------------------

def build_search_queries(message: str, intent_data: dict, llm) -> list[str]:
    depth = intent_data.get("research_depth", "standard")
    max_q = SEARCH_BUDGET.get(depth, 2)
    features = intent_data.get("salesforce_features", [])
    temporal = _determine_temporal_context(message, intent_data)
    boosters = _temporal_search_boosters(message, temporal)
    definition_intent = _is_definition_question(message, intent_data)
    definition_boosters = _definition_search_boosters(message, intent_data) if definition_intent else []

    logger.debug(
        f"QueryPlan | definition_intent={definition_intent} | "
        f"core_concept={_extract_core_concept(message) if definition_intent else None} | "
        f"boosters={definition_boosters} | temporal={temporal}"
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
        if temporal["temporal_validation_required"] and not temporal["historical_release_requested"]:
            current_query = f"{message} current Salesforce documentation"
            if current_query not in result:
                result.insert(0, current_query)
        result = result[:max_q]
        logger.info(f"Planned queries: {result}")
        return result

    try:
        prompt_input = (
            f"question={json.dumps(message)}, "
            f"intent={json.dumps(intent_data.get('intent'))}, "
            f"features={json.dumps(features)}, "
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
            for booster in reversed(boosters):
                if booster not in result:
                    result.insert(0, booster)
            if temporal["temporal_validation_required"] and not temporal["historical_release_requested"]:
                current_query = f"{message} current Salesforce documentation"
                if current_query not in result:
                    result.insert(0, current_query)
            if definition_intent:
                core = _extract_core_concept(message)
                for query in _definition_query_variants(core, intent_data):
                    if query not in result:
                        result.insert(0, query)
            result = result[:max_q]
            logger.info(f"Planned queries: {result}")
            return result
    except Exception as e:
        logger.warning(f"Query planning failed: {e} — using message as query")

    return [message]


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


def _topic_terms(message: str, intent_data: dict) -> set[str]:
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
    }
    terms: set[str] = set()
    sources = []
    sources.extend(intent_data.get("topics", []) or [])
    sources.extend(intent_data.get("salesforce_features", []) or [])
    sources.append(message)
    for source in sources:
        for word in re.findall(r"[a-zA-Z]{3,}", str(source).lower()):
            if word not in stopwords:
                terms.add(word)
    return terms


def _is_relevant(item: dict, terms: set[str], min_matches: int = 2) -> bool:
    if not terms:
        return True
    blob = _evidence_blob(item)
    matches = sum(1 for term in terms if term in blob)
    min_matches = max(2, len(terms) // 3)
    return matches >= min_matches


def _filter_relevant_evidence(evidence: list[dict], terms: set[str]) -> list[dict]:
    if not terms:
        return evidence
    return [item for item in evidence if _is_relevant(item, terms)]


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


def retrieve_evidence(message: str, intent_data: dict, llm, temporal_context: dict = None) -> list[dict]:
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

    # Build focused queries
    queries = build_search_queries(message, intent_data, llm)
    topic_terms = _topic_terms(message, intent_data)
    if temporal.get("historical_release_requested") and temporal.get("requested_release"):
        release_query = f"{message} {temporal['requested_release']} Salesforce documentation"
        if release_query not in queries:
            queries.insert(0, release_query)
    elif temporal.get("temporal_validation_required"):
        current_query = f"{message} current Salesforce documentation release notes"
        if current_query not in queries:
            queries.insert(0, current_query)

    logger.info(f"SearchBudget={search_budget} | Queries={len(queries)}")

    # Execute searches within budget
    for i, query in enumerate(queries[:search_budget], 1):
        logger.info(f"Search #{i} | query='{query}'")
        result = mcp_search(query)
        if result:
            evidence.append(result)

        if GOOGLE_SEARCH_MODE in {"hybrid", "always"} and _google_available():
            google_hits = google_search(query)
            if google_hits:
                logger.info(
                    f"GoogleSearchSupplement | query='{query}' | hits={len(google_hits)}"
                )
                _merge_unique_evidence(evidence, google_hits)

        definition_satisfied = True
        if _is_definition_question(message, intent_data):
            definition_satisfied = any(_definition_page_bonus(item) > 0 for item in evidence)

        if _is_sufficient(
            evidence,
            intent_data,
            require_authoritative=require_authoritative,
        ) and any(_is_relevant(item, topic_terms) for item in evidence) and definition_satisfied:
            logger.info(f"EvidenceSufficient=true after {i} search(es)")
            break

    if GOOGLE_SEARCH_MODE == "fallback" and _google_available():
        needs_google = (
            not evidence
            or not any(_is_relevant(item, topic_terms) for item in evidence)
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
                logger.info(f"RescueSearch | query='{query}'")
                result = mcp_search(query)
                if result:
                    evidence.append(result)

                if GOOGLE_SEARCH_MODE in {"hybrid", "always"} and _google_available():
                    google_hits = google_search(query)
                    if google_hits:
                        logger.info(
                            f"GoogleSearchSupplement | rescue_query='{query}' | hits={len(google_hits)}"
                        )
                        _merge_unique_evidence(evidence, google_hits)

            if _definition_evidence_is_sufficient(evidence, topic_terms):
                logger.info("DefinitionRetrievalRetry=resolved")
            else:
                logger.warning("DefinitionRetrievalRetry=insufficient | continuing with best available evidence")

    # Always fetch the top doc when we have a document path — search excerpts are
    # capped at MCP_MAX_TOKENS and too thin for synthesis + grounding on their own.
    fetch_budget = max(fetch_budget, FETCH_BUDGET_DEFINITION)

    # Optional fetch: only spend fetch budget on a relevant result that has a real path.
    if fetch_budget > 0 and evidence:
        ranked_for_fetch = _sort_evidence_for_intent(evidence, intent_data, message)
        fetch_source = next(
            (item for item in ranked_for_fetch if item.get("document_path") and _is_relevant(item, topic_terms)),
            None,
        )
        if fetch_source:
            doc_path = fetch_source.get("document_path")
            logger.debug(
                f"FetchSelection | title={fetch_source.get('title')} | "
                f"path={doc_path} | url={fetch_source.get('url')} | "
                f"definition={_is_definition_question(message, intent_data)}"
            )
            logger.info(f"Fetch | hinted path='{doc_path}'")
            fetched = mcp_fetch(doc_path)
            if fetched:
                evidence.append(fetched)

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
    if len(evidence) < 2:
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

def build_synthesis_prompt(message: str, intent_data: dict) -> str:
    parts = [RESEARCH_SYNTHESIS_PROMPT]
    if _is_definition_question(message, intent_data):
        parts.append(DEFINITION_FIRST_SYNTHESIS_PROMPT)
    intent = intent_data.get("intent", "general")

    if intent == "architecture" or intent_data.get("requires_architecture_analysis"):
        parts.append(ARCHITECTURE_REASONING_PROMPT)

    if intent in ("code_review", "troubleshooting") or intent_data.get("requires_code_analysis"):
        parts.append(CODE_REVIEW_PROMPT)

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


def generate_answer(message: str, evidence: list[dict], intent_data: dict, conflict_summary: dict, temporal_context: dict, llm, thread_id: str = "default") -> str:
    system_prompt = build_synthesis_prompt(message, intent_data)
    synthesis_evidence = evidence
    if _is_current_fact_request(intent_data, temporal_context):
        synthesis_evidence = _filter_current_fact_evidence(evidence)
    ordered_evidence = _sort_evidence_for_intent(synthesis_evidence, intent_data, message)

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
            "release-to-release. The retrieved documentation may reflect an older release. "
            "If the evidence does not include release notes, explicitly note in your answer that "
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
        return answer, "passed"
    except Exception as e:
        logger.warning(f"GroundingCheck failed: {e} — skipping")
        return answer, "error"

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
    _, classifier_llm = get_llm("fast")
    if not classifier_llm:
        return "❌ No LLM provider available. Configure GROQ_API_KEY or start Ollama."

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

    raw_evidence = retrieve_evidence(message, intent_data, classifier_llm, temporal_context)
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

    topic_terms = _topic_terms(message, intent_data)
    relevant_evidence = _filter_relevant_evidence(evidence, topic_terms)
    if evidence and topic_terms and not relevant_evidence:
        logger.warning("TopicRelevanceUnverified=true | no retrieved evidence matched topic terms")
        _record("relevance_filter", "no_match")
        logger.info(f"Trajectory | {trajectory}")
        return _build_unverified_relevance_response(message, evidence)
    if relevant_evidence:
        evidence = relevant_evidence
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
    if not synthesis_llm:
        return "❌ All providers failed. Please configure your API keys or verify Ollama is active."

    # Step 6 — generate answer (session-isolated thread_id)
    thread_id = _get_thread_id(history)
    logger.info(f"FinalGeneration | Provider={provider_label} | Tier={tier} | Thread={thread_id}")
    try:
        answer = generate_answer(message, evidence, intent_data, conflict_summary, temporal_context, synthesis_llm, thread_id)
    except Exception as e:
        logger.exception(f"Generation failed | {e}")
        return f"❌ Generation error: {e}"
    if _is_fallback_content(answer):
        logger.warning("ResponseUsedFallbackContent=true | kind=synthesis")
    _record("generate", "fallback" if _is_fallback_content(answer) else "ok", provider=provider_label)

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

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def set_groq_key(key: str):
    if key and isinstance(key, str) and key.strip():
        os.environ["GROQ_API_KEY"] = key.strip()
        return "✅ Groq key saved."
    return "⚠️ No key provided."


with gr.Blocks(fill_height=True) as demo:
    with gr.Sidebar():
        gr.Markdown("## ☁️ SFDC Architect Agent")
        gr.Markdown(
            "Salesforce Technical Architect AI — grounded in live "
            "Salesforce documentation via MCP."
        )
        gr.HTML("<hr>")
        gr.Markdown("### ⚙️ Providers")
        groq_input = gr.Textbox(
            label="Groq API Key",
            type="password",
            placeholder="gsk_...",
        )
        gr.Markdown("_Ollama is used as fallback._")
        status_label = gr.Label(value="Status: Ready.")
        groq_input.change(fn=set_groq_key, inputs=[groq_input], outputs=[status_label])
        gr.HTML("<hr>")
        gr.Markdown(
            "**Pipeline per turn:**\n"
            "1. Classify intent + entities\n"
            "2. Validate temporal scope\n"
            "3. Plan search queries\n"
            "4. Retrieve evidence (bounded)\n"
            "5. Normalize evidence\n"
            "6. Resolve evidence conflicts\n"
            "7. Generate answer (tier-routed)\n"
            "8. Grounding check\n\n"
            "**Model tiers:**\n"
            "- Fast — facts, definitions\n"
            "- Standard — comparisons, research\n"
            "- Reasoning — architecture, complex code"
        )

    gr.ChatInterface(
        fn=run_agent,
        fill_height=True,
        examples=[
            "Agentforce Coworker Benefits and Use Cases",
            "What is the latest Salesforce API version?",
            "What is a Platform Event?",
            "Can Flow replace an Apex trigger?",
            "We process 500,000 records every night. What Salesforce architecture should we use?",
            "Why am I getting Too many SOQL queries?",
        ],
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
