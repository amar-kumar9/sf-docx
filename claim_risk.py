"""Claim-level volatility for Salesforce answers.

Rules-only: classify the *shape* of a claim (numeric platform constraint,
edition availability, etc.) rather than a hard-coded topic list like "heap".
"""
from __future__ import annotations

import re
from typing import Any, Callable

MAX_CLAIMS = 8

# Mutable claim shapes (not topic names). Avoid bare digits alone — architecture
# questions often mention volumes (e.g. 500000 records) without a platform limit.
_MUTABLE_SHAPE_RE = re.compile(
    r"(?:"
    r"\b(?:limit|quota|threshold|maximum|minimum|max|min|timeout|duration)\b|"
    r"\b(?:heap|governor|soql|dml|cpu|callout|api)\s*(?:size|limit|version)?\b|"
    r"\b(?:supported|unsupported|enabled|disabled|available|unavailable)\b|"
    r"\b(?:enterprise|unlimited|professional|starter|developer)\s+edition\b|"
    r"\b(?:deprecat|retir|end[\s-]of[\s-]support|eos)\w*\b|"
    r"\b(?:spring|summer|winter|fall)\s*['’]?\d{2}\b|"
    r"\bapi\s*version\b|"
    r"\b\d+(?:\.\d+)?\s*(?:mb|gb|kb|ms|seconds?|minutes?|hours?|days?|%)\b|"
    r"\b(?:limit|quota|maximum|minimum|max|min|threshold|heap|governor|"
    r"soql|dml|cpu|callout|timeout)\b[^.\n]{0,40}?\b\d{1,3}(?:,\d{3})*\b|"
    r"\b\d{1,3}(?:,\d{3})*\b[^.\n]{0,40}?\b(?:limit|quota|maximum|minimum|"
    r"queries|statements|callouts|mb|gb)\b"
    r")",
    re.IGNORECASE,
)

_PLATFORM_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"salesforce|apex|soql|sosl|dml|flow|governor|platform|lightning|"
    r"api\s*version|bulk\s*api|queueable|batch\s*apex|visualforce|"
    r"enterprise\s+edition|unlimited\s+edition|professional\s+edition|"
    r"heap|callout|cpu\s*time|event\s*bus|platform\s*event"
    r")\b",
    re.IGNORECASE,
)

# Not Salesforce-controlled platform policy (protocol / language / domain).
_NOT_SF_CONTROLLED_RE = re.compile(
    r"(?:"
    r"\bhttp\s*(?:status\s*)?(?:code\s*)?\d{3}\b|"
    r"\b(?:404|200|201|301|302|400|401|403|500)\b.*\b(?:not\s+found|ok|created)\b|"
    r"\b(?:not\s+found|ok|created)\b.*\b(?:404|200|201)\b|"
    r"\benglish\s+alphabet\b|"
    r"\b26\s+letters\b|"
    r"\b(?:json|xml|html|css|tcp|udp|dns)\b|"
    r"\blist\s+is\s+ordered\b|"
    r"\bcase[\s-]insensitive\b|"
    r"\bstandard\s+object\b"  # taxonomic fact, usually stable
    r")",
    re.IGNORECASE,
)

_QUANTITY_CLAIM_RE = re.compile(
    r"(?P<span>"
    r".{0,80}?"
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*(?:mb|gb|kb|ms|seconds?|minutes?|hours?|days?|%)\b|"
    r"\b\d{1,3}(?:,\d{3})+\b|"
    r"(?<![.\w])\d{1,6}(?![.\w])\s*(?:queries|statements|callouts|records|jobs|licenses)?\b|"
    r"\bapi\s*version\s*\d+\b|"
    r"\b(?:maximum|minimum|max|min|limit|quota)\b[^.\n]{0,60}\b\d+"
    r")"
    r".{0,80}?"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_BOOLEAN_CLAIM_RE = re.compile(
    r"(?P<span>"
    r".{0,100}?"
    r"\b(?:is|are|supports?|does\s+not\s+support|unsupported|supported|"
    r"available|unavailable|enabled|disabled|retired|deprecated)\b"
    r".{0,100}?"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_EDITION_CLAIM_RE = re.compile(
    r"(?P<span>.{0,80}?\bavailable\s+in\b.{0,60}?\bedition\b.{0,40}?)",
    re.IGNORECASE | re.DOTALL,
)

_RELEASE_NOTES_META_RE = re.compile(
    r"release[-_\s]?notes|\brn_[a-z]|salesforce_release_notes",
    re.IGNORECASE,
)

ReleaseNotesFn = Callable[[dict], bool]


def question_suggests_mutable_platform_fact(message: str) -> bool:
    """Early heuristic for search budget — claim shape on the question text."""
    text = str(message or "").strip()
    if not text:
        return False
    if _NOT_SF_CONTROLLED_RE.search(text) and not _PLATFORM_CONTEXT_RE.search(text):
        return False
    if not _PLATFORM_CONTEXT_RE.search(text) and not re.search(
        r"\b(?:limit|quota|heap|governor|api\s*version|edition)\b", text, re.I
    ):
        return False
    return bool(_MUTABLE_SHAPE_RE.search(text))


def _normalize_span(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _evidence_text(item: dict) -> str:
    parts = [
        str(item.get("title") or ""),
        str(item.get("excerpt") or ""),
        str(item.get("url") or ""),
        str(item.get("document_path") or ""),
    ]
    return "\n".join(parts)


def _default_is_release_notes(item: dict) -> bool:
    blob = " ".join(
        [
            str(item.get("title") or ""),
            str(item.get("url") or ""),
            str(item.get("document_path") or ""),
            str(item.get("authority") or ""),
            str(item.get("source_type") or ""),
        ]
    )
    return bool(_RELEASE_NOTES_META_RE.search(blob))


def extract_claims(question: str, evidence: list[dict] | None = None) -> list[dict]:
    """Pull candidate claim spans from the question and top evidence excerpts."""
    candidates: list[dict] = []
    seen: set[str] = set()

    def _add(span: str, source: str, source_url: str = "") -> None:
        cleaned = _normalize_span(span)
        if len(cleaned) < 8:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "text": cleaned[:240],
                "source": source,
                "source_url": source_url,
            }
        )

    q = str(question or "").strip()
    if q:
        _add(q, "question")
        for match in _QUANTITY_CLAIM_RE.finditer(q):
            _add(match.group("span"), "question")
        for match in _EDITION_CLAIM_RE.finditer(q):
            _add(match.group("span"), "question")

    for item in (evidence or [])[:6]:
        body = _evidence_text(item)
        url = str(item.get("url") or "")
        for match in _QUANTITY_CLAIM_RE.finditer(body):
            _add(match.group("span"), "evidence", url)
            if len(candidates) >= MAX_CLAIMS * 2:
                break
        for match in _EDITION_CLAIM_RE.finditer(body):
            _add(match.group("span"), "evidence", url)
        # Boolean capability lines only when platform context is nearby.
        if _PLATFORM_CONTEXT_RE.search(body):
            for match in _BOOLEAN_CLAIM_RE.finditer(body):
                span = match.group("span")
                if _MUTABLE_SHAPE_RE.search(span) or re.search(
                    r"\b(?:supports?|available|deprecated|retired)\b", span, re.I
                ):
                    _add(span, "evidence", url)
        if len(candidates) >= MAX_CLAIMS * 2:
            break

    return candidates[: MAX_CLAIMS * 2]


def classify_claim(claim: dict | str) -> dict[str, Any]:
    """Classify one claim's volatility from semantic shape."""
    if isinstance(claim, str):
        text = claim
        source = "unknown"
        source_url = ""
    else:
        text = str(claim.get("text") or "")
        source = str(claim.get("source") or "unknown")
        source_url = str(claim.get("source_url") or "")

    lower = text.lower()
    result: dict[str, Any] = {
        "text": _normalize_span(text)[:240],
        "source": source,
        "source_url": source_url,
        "claim_type": "other",
        "salesforce_controlled": False,
        "volatility": "stable",
        "requires_current_release": False,
        "reason": "",
    }

    if not text.strip():
        result["reason"] = "empty claim"
        return result

    # Intrinsic / non-SF protocol facts first.
    if _NOT_SF_CONTROLLED_RE.search(text):
        # Exception: Salesforce edition/limit language wins over "standard object"
        # only when mutable platform shapes dominate.
        if re.search(r"\bstandard\s+object\b", lower) and not _MUTABLE_SHAPE_RE.search(text):
            result["claim_type"] = "intrinsic"
            result["salesforce_controlled"] = True
            result["volatility"] = "stable"
            result["reason"] = "taxonomic Salesforce fact; usually stable"
            return result
        if re.search(r"\bhttp\b|\b404\b|\b200\b|\bjson\b|\bxml\b|\balphabet\b", lower):
            result["claim_type"] = "intrinsic"
            result["salesforce_controlled"] = False
            result["volatility"] = "stable"
            result["reason"] = "not a Salesforce-controlled platform property"
            return result
        if re.search(r"\blist\s+is\s+ordered\b", lower):
            result["claim_type"] = "intrinsic"
            result["salesforce_controlled"] = True
            result["volatility"] = "stable"
            result["reason"] = "language semantics; not a seasonal platform policy"
            return result

    sf_controlled = bool(_PLATFORM_CONTEXT_RE.search(text))
    # Bare limit rows from Apex governor tables often omit "Salesforce" in the span.
    if not sf_controlled and re.search(
        r"\b(?:heap|soql|dml|governor|total\s+heap|synchronous\s+limit|"
        r"asynchronous\s+limit|api\s*version)\b",
        lower,
    ):
        sf_controlled = True

    result["salesforce_controlled"] = sf_controlled

    if re.search(r"\b(?:deprecat|retir|end[\s-]of[\s-]support)\w*\b", lower):
        result["claim_type"] = "deprecation"
    elif re.search(r"\bapi\s*version\b|\bversion\s*\d+\b", lower):
        result["claim_type"] = "version"
    elif re.search(r"\bedition\b|\bavailable\s+in\b", lower):
        result["claim_type"] = "availability"
    elif re.search(
        r"\b(?:supported|unsupported|supports?|enabled|disabled)\b", lower
    ) and not re.search(r"\b\d+(?:\.\d+)?\s*(?:mb|gb|kb)\b", lower):
        result["claim_type"] = "boolean_capability"
    elif _MUTABLE_SHAPE_RE.search(text) and re.search(r"\d", text):
        result["claim_type"] = "numeric_constraint"
    elif _MUTABLE_SHAPE_RE.search(text):
        result["claim_type"] = "boolean_capability"
    else:
        result["claim_type"] = "other"

    mutable_shape = bool(_MUTABLE_SHAPE_RE.search(text))
    # Feature availability / edition without an explicit digit still mutable.
    if result["claim_type"] in {"availability", "deprecation", "version"}:
        mutable_shape = True
    if result["claim_type"] == "boolean_capability" and sf_controlled:
        mutable_shape = True

    if not sf_controlled:
        result["volatility"] = "stable"
        result["requires_current_release"] = False
        result["reason"] = "not a Salesforce-controlled platform property"
        return result

    if mutable_shape:
        result["volatility"] = "potentially_mutable"
        result["requires_current_release"] = True
        result["reason"] = (
            f"{result['claim_type']}: Salesforce-controlled value that can change by release"
        )
        return result

    result["volatility"] = "stable"
    result["requires_current_release"] = False
    result["reason"] = "Salesforce context without a mutable claim shape"
    return result


def analyze_evidence_claims(
    question: str,
    evidence: list[dict] | None = None,
    *,
    is_release_notes: ReleaseNotesFn | None = None,
) -> dict[str, Any]:
    """Extract + classify claims; summarize volatility and RN coverage."""
    rn_fn = is_release_notes or _default_is_release_notes
    items = list(evidence or [])
    raw = extract_claims(question, items)
    classified = [classify_claim(c) for c in raw]

    # Prefer classified rows that are actionable; drop near-duplicates.
    claims: list[dict] = []
    seen: set[str] = set()
    for row in classified:
        key = row["text"].lower()[:120]
        if key in seen:
            continue
        seen.add(key)
        claims.append(row)
        if len(claims) >= MAX_CLAIMS:
            break

    # Ensure the question itself is classified when it looks mutable.
    if question_suggests_mutable_platform_fact(question):
        q_claim = classify_claim({"text": question, "source": "question"})
        key = q_claim["text"].lower()[:120]
        if key not in seen:
            claims.insert(0, q_claim)
            claims = claims[:MAX_CLAIMS]

    volatile = [c for c in claims if c.get("requires_current_release")]
    has_rn = any(rn_fn(item) for item in items)

    return {
        "claims": claims,
        "volatile_claims": volatile,
        "requires_current_release": bool(volatile) or question_suggests_mutable_platform_fact(question),
        "has_release_notes_evidence": has_rn,
    }


def verification_from_claim_risk(claim_risk: dict[str, Any]) -> dict[str, str]:
    """Host-facing verification status for the evidence pack."""
    if not claim_risk.get("requires_current_release"):
        return {"status": "ok", "missing": ""}
    if claim_risk.get("has_release_notes_evidence"):
        return {"status": "ok", "missing": ""}
    return {"status": "incomplete", "missing": "current_release_notes"}


def host_instructions_for_claim_risk(claim_risk: dict[str, Any], verification: dict[str, str]) -> str:
    """Extra host instructions when volatile claims lack current-release evidence."""
    if verification.get("status") != "incomplete":
        if claim_risk.get("requires_current_release") and claim_risk.get("has_release_notes_evidence"):
            return (
                " Volatile Salesforce platform claims are present; release-notes evidence "
                "was retrieved. Prefer release notes over standing developer-guide digests "
                "when they conflict."
            )
        return ""
    examples = []
    for claim in claim_risk.get("volatile_claims") or []:
        text = str(claim.get("text") or "").strip()
        if text:
            examples.append(text[:120])
        if len(examples) >= 3:
            break
    example_bit = ""
    if examples:
        example_bit = " Volatile claims include: " + " | ".join(examples) + "."
    return (
        " Claim-risk: one or more Salesforce-controlled mutable claims "
        "(limits, quotas, sizes, durations, edition availability, API version, "
        "support/deprecation) appear in the pack without current-release-notes evidence."
        + example_bit
        + " Do not present those values as definitively current. You may cite the "
        "standing developer guide as a digest that often lags. Verify via Help "
        "release notes (release-notes.rn_* / help.salesforce.com) before stating "
        "a number or availability as current. If verification fails, say the "
        "digest may lag and refuse a definitive latest value."
    )
