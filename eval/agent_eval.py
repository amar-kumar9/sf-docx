from __future__ import annotations

import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

import app as app_module
from app import (
    classify_intent,
    generate_answer,
    get_llm,
    normalize_evidence,
    resolve_evidence_conflicts,
    retrieve_evidence,
    validate_grounding,
    validate_temporal_context,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "golden_dataset.json"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "artifacts" / "eval" / "agent_eval_results.csv"
DEFAULT_COMPARE_OUTPUT_PATH = ROOT_DIR / "artifacts" / "eval" / "agent_eval_comparison.csv"
DEFAULT_SUMMARY_PATH = ROOT_DIR / "artifacts" / "eval" / "agent_eval_summary.json"
DEFAULT_MCP_FIXTURES_PATH = EVAL_DIR / "eval_mcp_fixtures.json"

STOPWORDS = {
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
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "about",
}

REQUIRED_TRACE_STEPS = (
    "guardrail",
    "classify",
    "temporal",
    "retrieve",
    "relevance_filter",
    "conflict",
    "current_fact_gate",
    "generate",
    "grounding",
)

TYPE_THRESHOLDS: dict[str, float] = {
    "current_fact": 0.42,
    "historical_release": 0.40,
    "explanation": 0.32,
    "design": 0.35,
    "comparison": 0.34,
    "default": 0.30,
}

RUBRIC_NAMES = (
    "final_response_quality",
    "groundedness",
    "tool_use_quality",
    "trajectory_order",
)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _word_set(text: str) -> set[str]:
    words = set()
    for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9'.-]*", _normalize_text(text)):
        if token in STOPWORDS:
            continue
        words.add(token)
    return words


def _clone_mapping(value: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cloned = json.loads(json.dumps(value))
    cloned.update(extra)
    return cloned


def _load_mcp_fixtures(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MCP fixture file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MCP fixture file must contain a JSON array.")
    fixtures: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"MCP fixture #{index} is not an object.")
        fixtures.append(item)
    return fixtures


def _fixture_patterns(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _fixture_matches(query: str, patterns: list[str]) -> bool:
    normalized_query = _normalize_text(query)
    if not patterns:
        return False
    return any(_normalize_text(pattern) in normalized_query for pattern in patterns)


def _fixture_search_response(query: str, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    for fixture in fixtures:
        patterns = _fixture_patterns(fixture.get("match")) + _fixture_patterns(fixture.get("search_match"))
        if not _fixture_matches(query, patterns):
            continue
        payload = fixture.get("search") or fixture.get("result") or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Search fixture '{fixture.get('name', 'unnamed')}' must be an object.")
        response = _clone_mapping(payload, query=query)
        response.setdefault("fixture_name", fixture.get("name"))
        return response
    return {}


def _fixture_fetch_response(document_path: str, fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_path = _normalize_text(document_path)
    for fixture in fixtures:
        patterns = _fixture_patterns(fixture.get("document_path")) + _fixture_patterns(fixture.get("fetch_match"))
        if not patterns:
            search_payload = fixture.get("search") if isinstance(fixture.get("search"), dict) else {}
            fetch_payload = fixture.get("fetch") if isinstance(fixture.get("fetch"), dict) else {}
            patterns = _fixture_patterns(search_payload.get("document_path")) + _fixture_patterns(fetch_payload.get("document_path"))
        if not patterns:
            continue
        if not any(_normalize_text(pattern) in normalized_path for pattern in patterns):
            continue
        payload = fixture.get("fetch") or fixture.get("search") or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Fetch fixture '{fixture.get('name', 'unnamed')}' must be an object.")
        response = _clone_mapping(payload, document_path=document_path)
        response.setdefault("fixture_name", fixture.get("name"))
        return response
    return {}


def _load_samples(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        raise ValueError(
            "Dataset must be a JSON array of objects or a JSON object containing "
            "a 'samples', 'data', or 'items' array."
        )

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(payload, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample #{index} is not an object.")
        query = sample.get("user_input") or sample.get("question")
        reference = sample.get("reference") or sample.get("expected_answer")
        if not query:
            raise ValueError(f"Sample #{index} is missing 'user_input'.")
        if not reference:
            raise ValueError(f"Sample #{index} is missing 'reference'.")

        row = dict(sample)
        row["user_input"] = str(query).strip()
        row["reference"] = str(reference).strip()
        row.setdefault("type", row.get("question_type", "default"))
        row.setdefault("reference_contexts", [])
        samples.append(row)
    return samples


def _reference_texts(sample: dict[str, Any]) -> list[str]:
    texts = [sample.get("reference", "")]
    reference_contexts = sample.get("reference_contexts") or []
    if isinstance(reference_contexts, list):
        texts.extend(str(item) for item in reference_contexts if item)
    return [text for text in texts if text]


def _best_overlap(candidate: str, reference_texts: list[str]) -> float:
    candidate_words = _word_set(candidate)
    if not candidate_words or not reference_texts:
        return 0.0
    best = 0.0
    for reference in reference_texts:
        ref_words = _word_set(reference)
        if not ref_words:
            continue
        overlap = len(candidate_words & ref_words) / max(1, len(ref_words))
        if overlap > best:
            best = overlap
    return best


def _trajectory_from_steps(steps: list[dict[str, Any]]) -> list[str]:
    return [str(step.get("step", "")) for step in steps]


def _schema_failures(result: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    required_keys = ("response", "trace", "evidence", "grounding_status")
    for key in required_keys:
        if key not in result:
            failures.append(f"missing_key:{key}")

    trace = result.get("trace")
    if not isinstance(trace, dict):
        failures.append("trace_not_object")
        return failures

    if not isinstance(trace.get("steps"), list):
        failures.append("trace_steps_not_list")
        return failures

    for idx, step in enumerate(trace["steps"], start=1):
        if not isinstance(step, dict):
            failures.append(f"trace_step_not_object:{idx}")
            continue
        if not step.get("step"):
            failures.append(f"trace_step_missing_name:{idx}")
        if not step.get("outcome"):
            failures.append(f"trace_step_missing_outcome:{idx}")

    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        failures.append("evidence_not_list")
    else:
        for idx, item in enumerate(evidence, start=1):
            if not isinstance(item, dict):
                failures.append(f"evidence_item_not_object:{idx}")
                continue
            if "excerpt" not in item:
                failures.append(f"evidence_item_missing_excerpt:{idx}")

    return failures


def _trajectory_failures(trace: dict[str, Any]) -> list[str]:
    steps = _trajectory_from_steps(trace.get("steps", []))
    failures: list[str] = []
    if not steps:
        return ["empty_trace"]

    required = [step for step in REQUIRED_TRACE_STEPS if step in steps]
    if len(required) < len(REQUIRED_TRACE_STEPS):
        missing = [step for step in REQUIRED_TRACE_STEPS if step not in steps]
        failures.append(f"missing_steps:{','.join(missing)}")

    order_index = {step: idx for idx, step in enumerate(steps)}
    ordered_chain = [step for step in REQUIRED_TRACE_STEPS if step in order_index]
    for left, right in zip(ordered_chain, ordered_chain[1:]):
        if order_index[left] > order_index[right]:
            failures.append(f"out_of_order:{left}>{right}")
            break
    return failures


def _rubric_score(scale: float) -> int:
    if scale >= 0.9:
        return 5
    if scale >= 0.75:
        return 4
    if scale >= 0.55:
        return 3
    if scale >= 0.35:
        return 2
    return 1


def _score_rubrics(sample: dict[str, Any], result: dict[str, Any], failures: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    sample_type = str(sample.get("type", "default")).strip().lower()
    response = str(result.get("response", ""))
    evidence = result.get("evidence", []) or []
    trace = result.get("trace", {}) or {}
    grounding_status = str(result.get("grounding_status", ""))
    coverage = float(meta.get("reference_coverage", 0.0) or 0.0)
    threshold = float(meta.get("required_coverage", _session_threshold(sample_type)) or _session_threshold(sample_type))
    trace_steps = meta.get("trace_steps", [])
    trajectory_failures = _trajectory_failures(trace)
    schema_failures = _schema_failures(result)

    evidence_count = len(evidence)
    required_doc_case = sample_type in {"current_fact", "historical_release", "explanation", "design", "comparison"}
    unverified = "[Unverified]" in response
    fallback = bool(meta.get("response_is_fallback"))

    final_response_quality = 5
    if fallback:
        final_response_quality = 1
    elif unverified and sample_type in {"current_fact", "historical_release"}:
        final_response_quality = 2
    elif coverage < threshold:
        final_response_quality = 2 if coverage < (threshold / 2) else 3
    elif coverage >= threshold and not unverified:
        final_response_quality = 5
    elif coverage >= threshold:
        final_response_quality = 4

    groundedness = 5
    if grounding_status == "passed":
        groundedness = 5
    elif grounding_status == "skipped (no evidence)":
        groundedness = 2 if required_doc_case else 3
    elif grounding_status == "flagged":
        groundedness = 4 if sample_type == "explanation" and evidence_count else 2
    elif grounding_status == "rewritten":
        groundedness = 3
    else:
        groundedness = 1
    if unverified and sample_type in {"current_fact", "historical_release"}:
        groundedness = min(groundedness, 2)
    if not evidence_count and required_doc_case:
        groundedness = min(groundedness, 2)

    tool_use_quality = 5
    if schema_failures:
        tool_use_quality = 1
    elif trajectory_failures:
        tool_use_quality = 2
    elif evidence_count == 0 and required_doc_case:
        tool_use_quality = 2
    elif evidence_count == 1 and sample_type in {"current_fact", "historical_release"}:
        tool_use_quality = 3
    elif evidence_count >= 2:
        tool_use_quality = 4
    trajectory_order = 5
    if trajectory_failures:
        trajectory_order = 1 if any(f.startswith("out_of_order") for f in trajectory_failures) else 2
    if "generate" in trace_steps and "retrieve" in trace_steps:
        if trace_steps.index("generate") < trace_steps.index("retrieve"):
            trajectory_order = 1
    if "grounding" in trace_steps and "generate" in trace_steps:
        if trace_steps.index("grounding") < trace_steps.index("generate"):
            trajectory_order = 1

    rubric_scores = {
        "final_response_quality": final_response_quality,
        "groundedness": groundedness,
        "tool_use_quality": tool_use_quality,
        "trajectory_order": trajectory_order,
    }
    rubric_average = round(sum(rubric_scores.values()) / (5.0 * len(rubric_scores)), 4)
    rubric_pass = all(score >= 4 for score in rubric_scores.values()) and not schema_failures and not trajectory_failures
    return {
        "rubric_scores": rubric_scores,
        "rubric_average": rubric_average,
        "rubric_pass": rubric_pass,
    }


def _session_threshold(sample_type: str) -> float:
    return TYPE_THRESHOLDS.get(str(sample_type or "default").strip().lower(), TYPE_THRESHOLDS["default"])


def _grade_session(sample: dict[str, Any], response: str, evidence: list[dict[str, Any]], trace: dict[str, Any], grounding_status: str) -> tuple[bool, list[str], dict[str, Any]]:
    sample_type = str(sample.get("type", "default")).strip().lower()
    reference_texts = _reference_texts(sample)
    coverage = _best_overlap(response, reference_texts)
    trace_steps = _trajectory_from_steps(trace.get("steps", []))
    failures: list[str] = []
    schema_failures = _schema_failures({
        "response": response,
        "trace": trace,
        "evidence": evidence,
        "grounding_status": grounding_status,
    })
    trajectory_failures = _trajectory_failures(trace)

    if not evidence:
        failures.append("no_evidence")
    if app_module._is_fallback_content(response):
        failures.append("fallback_content")
    if sample_type in {"current_fact", "historical_release"} and "[Unverified]" in response:
        failures.append("unverified_current_fact")
    if grounding_status not in {"passed", "skipped (no evidence)"} and not app_module._is_fallback_content(response):
        failures.append(f"grounding_{grounding_status}")
    if schema_failures:
        failures.extend(f"schema:{failure}" for failure in schema_failures)
    if trajectory_failures:
        failures.extend(f"trajectory:{failure}" for failure in trajectory_failures)

    threshold = _session_threshold(sample_type)
    if coverage < threshold:
        failures.append(f"low_reference_coverage:{coverage:.2f}<{threshold:.2f}")

    if sample_type in {"current_fact", "historical_release"} and not app_module._has_authoritative_salesforce_source(evidence):
        failures.append("missing_authoritative_source")

    result = {
        "sample_type": sample_type,
        "reference_coverage": round(coverage, 4),
        "required_coverage": threshold,
        "trace_steps": trace_steps,
        "missing_steps": [step for step in REQUIRED_TRACE_STEPS if step not in trace_steps],
        "evidence_count": len(evidence),
        "grounding_status": grounding_status,
        "response_is_fallback": app_module._is_fallback_content(response),
        "schema_failures": schema_failures,
        "trajectory_failures": trajectory_failures,
    }
    return (len(failures) == 0), failures, result


@contextmanager
def _patched_google_mode(mode: str):
    original_enabled = app_module.GOOGLE_FALLBACK_ENABLED
    original_mode = app_module.GOOGLE_SEARCH_MODE
    normalized = (mode or "").strip().lower()
    if normalized == "none":
        app_module.GOOGLE_FALLBACK_ENABLED = False
        app_module.GOOGLE_SEARCH_MODE = "none"
    elif normalized in {"hybrid", "fallback", "always"}:
        app_module.GOOGLE_FALLBACK_ENABLED = True
        app_module.GOOGLE_SEARCH_MODE = normalized
    try:
        yield
    finally:
        app_module.GOOGLE_FALLBACK_ENABLED = original_enabled
        app_module.GOOGLE_SEARCH_MODE = original_mode


@contextmanager
def _patched_eval_provider_preference():
    """Prefer local Ollama during eval so Groq outages do not confound results."""
    original_value = os.environ.get("GROQ_API_KEY")
    had_key = "GROQ_API_KEY" in os.environ
    os.environ["GROQ_API_KEY"] = ""
    try:
        yield
    finally:
        if had_key:
            os.environ["GROQ_API_KEY"] = original_value or ""
        else:
            os.environ.pop("GROQ_API_KEY", None)


@contextmanager
def _patched_eval_mcp(mode: str, fixtures_path: Path):
    normalized = (mode or "").strip().lower()
    if normalized != "fixture":
        yield
        return

    fixtures = _load_mcp_fixtures(fixtures_path)
    original_mcp_search = app_module.mcp_search
    original_mcp_fetch = app_module.mcp_fetch

    def fixture_search(query: str) -> dict[str, Any]:
        result = _fixture_search_response(query, fixtures)
        if result:
            logger.info(
                "MCPFixtureSearch | query='{}' | fixture='{}'".format(
                    query,
                    result.get("fixture_name") or "unnamed",
                )
            )
        else:
            logger.info(f"MCPFixtureSearch | query='{query}' | fixture='empty'")
        return result

    def fixture_fetch(document_path: str) -> dict[str, Any]:
        result = _fixture_fetch_response(document_path, fixtures)
        if result:
            logger.info(
                "MCPFixtureFetch | path='{}' | fixture='{}'".format(
                    document_path,
                    result.get("fixture_name") or "unnamed",
                )
            )
        else:
            logger.info(f"MCPFixtureFetch | path='{document_path}' | fixture='empty'")
        return result

    app_module.mcp_search = fixture_search
    app_module.mcp_fetch = fixture_fetch
    try:
        yield
    finally:
        app_module.mcp_search = original_mcp_search
        app_module.mcp_fetch = original_mcp_fetch


def _run_session(sample: dict[str, Any]) -> dict[str, Any]:
    query = sample["user_input"]
    trace: dict[str, Any] = {"query": query, "steps": []}

    def record(step: str, outcome: str, **payload: Any) -> None:
        entry = {"step": step, "outcome": outcome}
        entry.update(payload)
        trace["steps"].append(entry)

    should_continue, early_response = app_module.input_guardrail(query)
    if not should_continue:
        record("guardrail", "blocked")
        return {
            "user_input": query,
            "reference": sample["reference"],
            "response": early_response,
            "retrieved_contexts": [],
            "trace": trace,
            "evidence": [],
            "grounding_status": "blocked",
            "provider": None,
            "provider_used": None,
            "fallback_reason": None,
        }
    record("guardrail", "passed")

    _, fast_llm = get_llm("fast")
    if fast_llm is None:
        response = "❌ No LLM provider available. Configure GROQ_API_KEY or start Ollama."
        trace["provider_used"] = None
        trace["fallback_reason"] = "no_fast_llm"
        return {
            "user_input": query,
            "reference": sample["reference"],
            "response": response,
            "retrieved_contexts": [],
            "trace": trace,
            "evidence": [],
            "grounding_status": "no_llm",
            "provider": None,
            "provider_used": None,
            "fallback_reason": "no_fast_llm",
        }

    intent_data = classify_intent(query, fast_llm)
    record(
        "classify",
        "ok",
        intent=intent_data.get("intent"),
        tier=intent_data.get("model_tier"),
        depth=intent_data.get("research_depth"),
        requires_docs=intent_data.get("requires_documentation"),
    )

    temporal_context = validate_temporal_context(query, intent_data, fast_llm)
    record(
        "temporal",
        "ok",
        temporal_required=temporal_context.get("temporal_validation_required"),
        current_docs=temporal_context.get("current_docs_required"),
    )

    raw_evidence = retrieve_evidence(query, intent_data, fast_llm, temporal_context)
    evidence = normalize_evidence(raw_evidence)
    record("retrieve", "ok", evidence_count=len(evidence))

    topic_terms = app_module._topic_terms(query, intent_data)
    relevant_evidence = app_module._filter_relevant_evidence(evidence, topic_terms)
    if evidence and topic_terms and not relevant_evidence:
        record("relevance_filter", "no_match")
        return {
            "user_input": query,
            "reference": sample["reference"],
            "response": app_module._build_unverified_relevance_response(query, evidence),
            "retrieved_contexts": [item.get("excerpt", "") for item in evidence if item.get("excerpt")],
            "trace": trace,
            "evidence": evidence,
            "grounding_status": "blocked",
            "provider": None,
            "provider_used": None,
            "fallback_reason": None,
        }
    if relevant_evidence:
        evidence = relevant_evidence
    if app_module._is_current_fact_request(intent_data, temporal_context):
        evidence = app_module._filter_current_fact_evidence(evidence)
    record("relevance_filter", "ok", kept=len(evidence))

    conflict_summary = resolve_evidence_conflicts(evidence, intent_data, fast_llm, query)
    record("conflict", conflict_summary.get("status", "unknown"))

    if app_module._is_current_fact_request(intent_data, temporal_context) and not app_module._has_authoritative_salesforce_source(evidence):
        record("current_fact_gate", "blocked")
        response = app_module._build_unverified_current_fact_response(query, evidence, temporal_context)
        return {
            "user_input": query,
            "reference": sample["reference"],
            "response": response,
            "retrieved_contexts": [item.get("excerpt", "") for item in evidence if item.get("excerpt")],
            "trace": trace,
            "evidence": evidence,
            "grounding_status": "blocked",
            "provider": None,
            "provider_used": None,
            "fallback_reason": None,
        }
    record("current_fact_gate", "passed")

    tier = intent_data.get("model_tier", "standard")
    provider_label, synthesis_llm = get_llm(tier)
    if synthesis_llm is None:
        response = "❌ All providers failed. Please configure your API keys or verify Ollama is active."
        trace["provider_used"] = provider_label
        trace["fallback_reason"] = "no_synthesis_provider"
        record(
            "generate",
            "fallback",
            provider=provider_label,
            provider_used=provider_label,
            fallback_reason="no_synthesis_provider",
        )
        return {
            "user_input": query,
            "reference": sample["reference"],
            "response": response,
            "retrieved_contexts": [item.get("excerpt", "") for item in evidence if item.get("excerpt")],
            "trace": trace,
            "evidence": evidence,
            "grounding_status": "no_synthesis_llm",
            "provider": None,
            "provider_used": provider_label,
            "fallback_reason": "no_synthesis_provider",
        }

    thread_id = app_module._get_thread_id([])
    generation_error: str | None = None
    try:
        response = generate_answer(query, evidence, intent_data, conflict_summary, temporal_context, synthesis_llm, thread_id)
    except Exception as exc:
        logger.exception(f"Generation failed | {exc}")
        response = f"❌ Generation error: {exc}"
        generation_error = str(exc)
    fallback_reason = "generation_fallback_content" if app_module._is_fallback_content(response) else None
    if generation_error:
        fallback_reason = "generation_error"
    trace["provider_used"] = provider_label
    trace["fallback_reason"] = fallback_reason
    if fallback_reason:
        record("generate", "fallback", provider=provider_label, provider_used=provider_label, fallback_reason=fallback_reason)
    else:
        record("generate", "ok", provider=provider_label, provider_used=provider_label, fallback_reason=None)

    response, grounding_status = validate_grounding(response, evidence, fast_llm, query, intent_data)
    record("grounding", grounding_status)

    if app_module._is_volatile_limit_question(query) and temporal_context.get("current_docs_required"):
        rn_url = "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm"
        has_rn = any(
            "release-notes" in (item.get("url") or "").lower() or "release-notes" in (item.get("document_path") or "").lower()
            for item in evidence
        )
        if not has_rn:
            response += (
                f"\n\n> **⚠️ Verify against latest release notes:** Governor limits change "
                f"release-to-release. Confirm the current value at "
                f"[Salesforce Release Notes]({rn_url})."
            )

    return {
        "user_input": query,
        "reference": sample["reference"],
        "response": response,
        "retrieved_contexts": [item.get("excerpt", "") for item in evidence if item.get("excerpt")],
        "trace": trace,
        "evidence": evidence,
        "grounding_status": grounding_status,
        "provider": provider_label,
        "provider_used": provider_label,
        "fallback_reason": fallback_reason,
        "intent": intent_data.get("intent"),
        "sample_type": sample.get("type", "default"),
    }


def _run_rows(
    samples: list[dict[str, Any]],
    google_mode: str,
    variant: str,
    mcp_mode: str,
    mcp_fixtures_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _patched_google_mode(google_mode), _patched_eval_provider_preference(), _patched_eval_mcp(mcp_mode, mcp_fixtures_path):
        for sample in samples:
            result = _run_session(sample)
            passed, failures, meta = _grade_session(
                sample,
                result["response"],
                result["evidence"],
                result["trace"],
                result["grounding_status"],
            )
            rubric = _score_rubrics(sample, result, failures, meta)
            row = {
                **result,
                "variant": variant,
                "google_mode": google_mode,
                "overall_pass": passed,
                "failed_checks": failures,
                "schema_pass": not meta["schema_failures"],
                "trajectory_pass": not meta["trajectory_failures"],
                "rubric_pass": rubric["rubric_pass"],
                "rubric_average": rubric["rubric_average"],
                "reference_coverage": meta["reference_coverage"],
                "required_coverage": meta["required_coverage"],
                "trace_steps": json.dumps(meta["trace_steps"], ensure_ascii=False),
                "missing_steps": json.dumps(meta["missing_steps"], ensure_ascii=False),
                "schema_failures": json.dumps(meta["schema_failures"], ensure_ascii=False),
                "trajectory_failures": json.dumps(meta["trajectory_failures"], ensure_ascii=False),
                "evidence_count": meta["evidence_count"],
                "response_is_fallback": meta["response_is_fallback"],
                "sample_type": meta["sample_type"],
                "provider_used": result.get("provider_used") or result.get("provider"),
                "fallback_reason": result.get("fallback_reason") or result.get("trace", {}).get("fallback_reason"),
            }
            for rubric_name, score in rubric["rubric_scores"].items():
                row[rubric_name] = score
            row["trace_json"] = json.dumps(result["trace"], ensure_ascii=False)
            row["evidence_json"] = json.dumps(result["evidence"], ensure_ascii=False)
            row["retrieved_contexts"] = json.dumps(result["retrieved_contexts"], ensure_ascii=False)
            row.pop("trace", None)
            row.pop("evidence", None)
            rows.append(row)
    return rows


def _write_report(rows: list[dict[str, Any]], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(output_path, index=False)
    return df


def _print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n=================== {label.upper()} SUMMARY ===================")
    if df.empty:
        print("No rows evaluated.")
        return
    print(f"Overall pass rate: {df['overall_pass'].mean():.2%}")
    if "rubric_average" in df.columns:
        print(f"Average rubric score: {df['rubric_average'].mean():.3f}")
    for metric in RUBRIC_NAMES:
        if metric in df.columns:
            print(f"  {metric}: {df[metric].mean():.2f}")
    if "sample_type" in df.columns:
        print("Pass rate by sample_type:")
        for sample_type, group in df.groupby("sample_type", dropna=False):
            print(f"  {sample_type}: {group['overall_pass'].mean():.2%} ({len(group)} cases)")
    if "grounding_status" in df.columns:
        print("Grounding status counts:")
        print(df["grounding_status"].value_counts(dropna=False).to_string())
    if "failed_checks" in df.columns:
        failed = df["failed_checks"].astype(str).value_counts()
        print("Top failure patterns:")
        print(failed.head(10).to_string())
    if "schema_failures" in df.columns:
        schema_counts = df["schema_failures"].astype(str).value_counts()
        print("Top schema failure patterns:")
        print(schema_counts.head(5).to_string())
    if "trajectory_failures" in df.columns:
        trajectory_counts = df["trajectory_failures"].astype(str).value_counts()
        print("Top trajectory failure patterns:")
        print(trajectory_counts.head(5).to_string())


def _build_summary(df: pd.DataFrame, label: str, source: str | None = None) -> dict[str, Any]:
    if df.empty:
        return {
            "label": label,
            "source": source,
            "row_count": 0,
            "overall_pass_rate": 0.0,
            "rubric_average": 0.0,
            "rubric_scores": {},
            "grounding_status_counts": {},
            "sample_type_pass_rates": {},
            "failed_checks_top": [],
            "schema_failures_top": [],
            "trajectory_failures_top": [],
        }

    rubric_scores = {
        metric: round(float(df[metric].mean()), 4)
        for metric in RUBRIC_NAMES
        if metric in df.columns
    }
    sample_type_pass_rates: dict[str, float] = {}
    if "sample_type" in df.columns:
        for sample_type, group in df.groupby("sample_type", dropna=False):
            sample_type_pass_rates[str(sample_type)] = round(float(group["overall_pass"].mean()), 4)

    def _top_counts(column: str, limit: int = 5) -> list[dict[str, Any]]:
        if column not in df.columns:
            return []
        counts = df[column].astype(str).value_counts().head(limit)
        return [{"value": str(index), "count": int(count)} for index, count in counts.items()]

    return {
        "label": label,
        "source": source,
        "row_count": int(len(df)),
        "overall_pass_rate": round(float(df["overall_pass"].mean()), 4) if "overall_pass" in df.columns else 0.0,
        "rubric_average": round(float(df["rubric_average"].mean()), 4) if "rubric_average" in df.columns else 0.0,
        "rubric_scores": rubric_scores,
        "grounding_status_counts": (
            df["grounding_status"].astype(str).value_counts().to_dict() if "grounding_status" in df.columns else {}
        ),
        "sample_type_pass_rates": sample_type_pass_rates,
        "failed_checks_top": _top_counts("failed_checks"),
        "schema_failures_top": _top_counts("schema_failures"),
        "trajectory_failures_top": _top_counts("trajectory_failures"),
    }


def _write_summary_json(summary: dict[str, Any], summary_path: Path) -> None:
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _compare_metrics(baseline_df: pd.DataFrame, candidate_df: pd.DataFrame) -> dict[str, Any]:
    baseline_pass = float(baseline_df["overall_pass"].mean()) if not baseline_df.empty else 0.0
    candidate_pass = float(candidate_df["overall_pass"].mean()) if not candidate_df.empty else 0.0
    baseline_rubric = float(baseline_df["rubric_average"].mean()) if not baseline_df.empty else 0.0
    candidate_rubric = float(candidate_df["rubric_average"].mean()) if not candidate_df.empty else 0.0
    return {
        "baseline_overall_pass_rate": round(baseline_pass, 4),
        "candidate_overall_pass_rate": round(candidate_pass, 4),
        "overall_pass_delta": round(candidate_pass - baseline_pass, 4),
        "baseline_rubric_average": round(baseline_rubric, 4),
        "candidate_rubric_average": round(candidate_rubric, 4),
        "rubric_average_delta": round(candidate_rubric - baseline_rubric, 4),
    }


def _should_fail_regression(compare_metrics: dict[str, Any], max_pass_drop: float, max_rubric_drop: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if compare_metrics["overall_pass_delta"] < -abs(max_pass_drop):
        reasons.append(
            f"overall_pass_rate dropped by {abs(compare_metrics['overall_pass_delta']):.4f}, "
            f"which exceeds the allowed drop of {max_pass_drop:.4f}"
        )
    if compare_metrics["rubric_average_delta"] < -abs(max_rubric_drop):
        reasons.append(
            f"rubric_average dropped by {abs(compare_metrics['rubric_average_delta']):.4f}, "
            f"which exceeds the allowed drop of {max_rubric_drop:.4f}"
        )
    return bool(reasons), reasons


def run_eval(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    google_mode: str = "none",
    mcp_mode: str = "fixture",
    mcp_fixtures_path: str = str(DEFAULT_MCP_FIXTURES_PATH),
) -> pd.DataFrame:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    samples = _load_samples(dataset_file)
    if not samples:
        raise ValueError("Dataset is empty.")

    rows = _run_rows(
        samples,
        google_mode=google_mode,
        variant="baseline",
        mcp_mode=mcp_mode,
        mcp_fixtures_path=Path(mcp_fixtures_path),
    )
    df = _write_report(rows, Path(output_path))
    summary = _build_summary(df, label="session_eval", source=str(dataset_file))
    _write_summary_json(summary, Path(summary_path))
    print(f"Session eval complete. Results written to: {output_path}")
    print(f"Summary written to: {summary_path}")
    _print_summary(df, "session eval")
    return df


def run_compare(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    output_path: str = str(DEFAULT_COMPARE_OUTPUT_PATH),
    summary_path: str = str(DEFAULT_SUMMARY_PATH),
    baseline_google_mode: str = "none",
    candidate_google_mode: str = "none",
    mcp_mode: str = "fixture",
    mcp_fixtures_path: str = str(DEFAULT_MCP_FIXTURES_PATH),
    fail_on_regression: bool = False,
    max_pass_drop: float = 0.0,
    max_rubric_drop: float = 0.0,
) -> pd.DataFrame:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
    samples = _load_samples(dataset_file)
    if not samples:
        raise ValueError("Dataset is empty.")

    baseline_rows = _run_rows(
        samples,
        google_mode=baseline_google_mode,
        variant="baseline",
        mcp_mode=mcp_mode,
        mcp_fixtures_path=Path(mcp_fixtures_path),
    )
    candidate_rows = _run_rows(
        samples,
        google_mode=candidate_google_mode,
        variant="candidate",
        mcp_mode=mcp_mode,
        mcp_fixtures_path=Path(mcp_fixtures_path),
    )
    combined = _write_report(baseline_rows + candidate_rows, Path(output_path))
    baseline_df = combined[combined["variant"] == "baseline"].copy()
    candidate_df = combined[combined["variant"] == "candidate"].copy()
    compare_metrics = _compare_metrics(baseline_df, candidate_df)
    summary = {
        "label": "comparison",
        "source": str(dataset_file),
        "row_count": int(len(combined)),
        "comparison": compare_metrics,
        "baseline": _build_summary(baseline_df, label="baseline", source=str(dataset_file)),
        "candidate": _build_summary(candidate_df, label="candidate", source=str(dataset_file)),
    }
    _write_summary_json(summary, Path(summary_path))
    print(f"Comparison complete. Results written to: {output_path}")
    print(f"Summary written to: {summary_path}")
    _print_summary(combined[combined["variant"] == "baseline"], "baseline")
    _print_summary(combined[combined["variant"] == "candidate"], "candidate")
    if {"variant", "rubric_average"}.issubset(combined.columns):
        pivot = combined.pivot_table(index="user_input", columns="variant", values="rubric_average", aggfunc="first")
        if {"baseline", "candidate"}.issubset(pivot.columns):
            pivot["delta"] = pivot["candidate"].astype(float) - pivot["baseline"].astype(float)
            pivot["winner"] = pivot["delta"].apply(
                lambda value: "candidate" if value > 0.02 else "baseline" if value < -0.02 else "tie"
            )
            print("\nPairwise rubric tournament:")
            print(pivot.to_string())
            print("\nWin/tie summary:")
            print(pivot["winner"].value_counts(dropna=False).to_string())

    if fail_on_regression:
        failed, reasons = _should_fail_regression(compare_metrics, max_pass_drop=max_pass_drop, max_rubric_drop=max_rubric_drop)
        if failed:
            message = "Regression gate failed:\n- " + "\n- ".join(reasons)
            logger.error(message)
            raise SystemExit(1)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a session-level eval for the Salesforce QA agent.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH), help="Path to the JSON golden dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="CSV output path.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH), help="JSON summary output path.")
    parser.add_argument("--google-mode", default="none", help="Google mode for a single run: none|hybrid|fallback|always.")
    parser.add_argument("--mcp-mode", default="fixture", help="MCP mode for eval: fixture|live.")
    parser.add_argument("--mcp-fixtures", default=str(DEFAULT_MCP_FIXTURES_PATH), help="Path to the MCP fixture JSON file.")
    parser.add_argument("--compare", action="store_true", help="Run baseline and candidate variants and compare them.")
    parser.add_argument("--baseline-google-mode", default="none", help="Google mode for the baseline comparison run.")
    parser.add_argument("--candidate-google-mode", default="none", help="Google mode for the candidate comparison run.")
    parser.add_argument("--fail-on-regression", action="store_true", help="Exit non-zero if the candidate regresses.")
    parser.add_argument("--max-pass-drop", type=float, default=0.0, help="Allowed drop in overall pass rate before failing.")
    parser.add_argument("--max-rubric-drop", type=float, default=0.0, help="Allowed drop in rubric average before failing.")
    args = parser.parse_args()

    if args.compare:
        run_compare(
            dataset_path=args.dataset,
            output_path=args.output,
            summary_path=args.summary,
            baseline_google_mode=args.baseline_google_mode,
            candidate_google_mode=args.candidate_google_mode,
            mcp_mode=args.mcp_mode,
            mcp_fixtures_path=args.mcp_fixtures,
            fail_on_regression=args.fail_on_regression,
            max_pass_drop=args.max_pass_drop,
            max_rubric_drop=args.max_rubric_drop,
        )
    else:
        run_eval(
            dataset_path=args.dataset,
            output_path=args.output,
            summary_path=args.summary,
            google_mode=args.google_mode,
            mcp_mode=args.mcp_mode,
            mcp_fixtures_path=args.mcp_fixtures,
        )


if __name__ == "__main__":
    main()
