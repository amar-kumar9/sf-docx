"""Translate business requirements into Salesforce technical search targets.

A solution architect does not retrieve against the user's wording. They name the
object, the capability layers (know / act / show / connect / scale / trust),
and only then look up documentation. This module is that intake step.

It is capability-driven, not story-driven: Case risk, Account credit hold, and
Opportunity stuck-notify share the same rules.
"""
from __future__ import annotations

import re
from typing import Any

STANDARD_OBJECTS = (
    "Work Order",
    "WorkOrder",
    "Service Appointment",
    "ServiceAppointment",
    "Knowledge",
    "Opportunity",
    "Account",
    "Contact",
    "Campaign",
    "Contract",
    "Product",
    "Quote",
    "Order",
    "Lead",
    "Case",
    "User",
    "Asset",
    "Task",
    "Event",
    "Entitlement",
)

OBJECT_ALIASES = {
    "work order": "WorkOrder",
    "workorder": "WorkOrder",
    "service appointment": "ServiceAppointment",
    "serviceappointment": "ServiceAppointment",
}

# Overloaded English that means platform security only when the ask is about
# the org/session. On a CRM record it usually means a business process.
PLATFORM_SECURITY_LANGUAGE = re.compile(
    r"\b(shield|event monitoring|security center|threat detection|"
    r"transaction security|mfa|multi-factor|login|session|org-wide|"
    r"tenant security|event log)\b",
    re.IGNORECASE,
)
PROCESS_LANGUAGE = re.compile(
    r"\b(legal|complaint|customer|support rep|sales rep|cxo|board|"
    r"employee|social media|subject(?: line)?|description|credit hold|"
    r"churn|escalate|playbook|indicator)\b",
    re.IGNORECASE,
)
PLATFORM_SECURITY_EVIDENCE = re.compile(
    r"security.?center|threat.?detection|event.?monitoring|"
    r"shield.?threat|tenant.?security|security_center_alerts",
    re.IGNORECASE,
)

HOW_TO_RE = re.compile(
    r"\bhow (?:do i|do we|to|can i|can we|should i|should we|would i|would we)\b",
    re.IGNORECASE,
)
# A factual data-access question ("how do I GET/READ the <value>") is a lookup,
# not a design problem. It must retrieve against the named subject, never be
# capability-translated (which drifts the topic — e.g. "queue position" -> Omni
# routing). Design/build verbs (show, notify, implement, create) are NOT here on
# purpose; those stay solution questions.
FACT_LOOKUP_RE = re.compile(
    r"\bhow (?:do i|do we|can i|can we|to|would i|should i)\s+"
    r"(?:get|retrieve|read|fetch|obtain|access|query|look up|pull|grab)\b",
    re.IGNORECASE,
)
# When any of these appear, the question is building/showing something, so a bare
# lookup verb does not demote it out of the solution path.
_DESIGN_OVERRIDE_RE = re.compile(
    r"\b(architecture|well-?architected|decision guide|should we use|should i use|"
    r"vs apex|ldv|large data|integrat)\b",
    re.IGNORECASE,
)
IMPLEMENT_RE = re.compile(
    r"\b(implement(?:ation)?|we need (?:to|a)|build (?:a|an)|design (?:a|an)|"
    r"user stor(?:y|ies)|requirement)\b",
    re.IGNORECASE,
)
TROUBLESHOOTING_RE = re.compile(
    r"\b(why am i getting|error|exception|stack trace|soql 101|"
    r"too many (?:soql|dml)|unexpected exception|"
    r"steps to reproduce|actual behavio(?:u)?r|expected behavio(?:u)?r|"
    r"does not (?:trigger|fire|work|update|search|clear)|"
    r"doesn't (?:trigger|fire|work)|not triggered|"
    r"is there a way to fix|how (?:do i|to) fix|workaround|"
    r"known (?:issue|quirk|bug))\b",
    re.IGNORECASE,
)
DEFINITION_PREFIXES = (
    "what is",
    "what are",
    "explain",
    "define",
    "tell me about",
    "give an overview of",
    "give me an overview of",
)

TEXT_SIGNAL = re.compile(
    r"\b(subject(?: line)?|description|email body|free text|unstructured|"
    r"details and subject|from (?:the )?text)\b",
    re.IGNORECASE,
)
CLASSIFY_SIGNAL = re.compile(
    r"\b(classif|categoriz|detect(?:ion)?|identif(?:y|ication)|flag|label|score|"
    r"risk types?|sentiment)\b",
    re.IGNORECASE,
)
UI_SIGNAL = re.compile(
    r"\b(visual indicator|indicator|highlight|banner|badge|icon|"
    r"show (?:it |that )?(?:on|to)|rep can see|agent can see|"
    r"on the (?:record|page|console)|lightning page)\b",
    re.IGNORECASE,
)
AUTOMATION_SIGNAL = re.compile(
    r"\b(when (?:a |the )?record|automatically|on create|on update|"
    r"record-triggered|playbook|trigger|before save|after save)\b",
    re.IGNORECASE,
)
ASSIGN_SIGNAL = re.compile(
    r"\b(assign|route|omni-?channel|queue|owner)\b",
    re.IGNORECASE,
)
NOTIFY_SIGNAL = re.compile(
    r"\b(notif(?:y|ication)|email alert|alert the|ping the|tell the)\b",
    re.IGNORECASE,
)
FIELD_SIGNAL = re.compile(
    r"\b(custom field|picklist|checkbox|data model|new field|object manager)\b",
    re.IGNORECASE,
)
INTEGRATION_SIGNAL = re.compile(
    r"\b(integrat|erp|named credential|callout|platform event|"
    r"change data capture|\bcdc\b|\bmcp\b|rest api|soap)\b",
    re.IGNORECASE,
)
SCALE_SIGNAL = re.compile(
    r"\b(ldv|large data|bulk api|500,?000|million records|high volume|"
    r"nightly)\b",
    re.IGNORECASE,
)

# Well-known Salesforce products for (capability, object). This is catalog
# knowledge, not a one-off story. Unknown objects fall back to generic queries.
NAMED_SEARCHES: dict[tuple[str, str], list[str]] = {
    ("text_classification", "Case"): [
        "Einstein Case Classification Case Subject Description",
        "Salesforce Case classification Subject Description",
        "Salesforce Case Classification with AI options key considerations",
    ],
    ("text_classification", "Lead"): [
        "Einstein Lead Scoring",
        "Salesforce Lead scoring",
    ],
    ("scoring", "Opportunity"): [
        "Einstein Opportunity Scoring",
        "Salesforce Opportunity scoring",
    ],
    ("assignment", "Case"): [
        "Salesforce Case assignment rules",
        "Omni-Channel Case routing",
    ],
}

EVIDENCE_OK: dict[str, re.Pattern[str]] = {
    "text_classification": re.compile(
        r"classif|scoring|prediction|recommend(?:ed|ation)? field|picklist",
        re.IGNORECASE,
    ),
    "scoring": re.compile(r"scoring|einstein opportunity|prediction", re.IGNORECASE),
    "record_ui": re.compile(
        r"lightning (?:record )?page|highlight panel|compact layout|"
        r"dynamic form|service console|record page",
        re.IGNORECASE,
    ),
    "record_automation": re.compile(
        r"record-triggered|flow vs apex|before-save|after-save|"
        r"record triggered automation",
        re.IGNORECASE,
    ),
    "assignment": re.compile(r"assignment rule|omni-channel|queue", re.IGNORECASE),
    "notification": re.compile(
        r"custom notification|email alert|notify|notification",
        re.IGNORECASE,
    ),
    "data_model": re.compile(r"custom field|object manager|picklist|schema", re.IGNORECASE),
    "integration": re.compile(
        r"integration|rest api|named credential|platform event|"
        r"change data capture|\bmcp\b",
        re.IGNORECASE,
    ),
    "scale": re.compile(r"bulk api|batch apex|large data|\bldv\b|queueable", re.IGNORECASE),
    "platform_security": re.compile(
        r"security center|shield|event monitoring|threat detection",
        re.IGNORECASE,
    ),
}

EMPTY_PLAN = {
    "active": False,
    "skip_raw": False,
    "full_protocol": False,
    "business_outcome": "",
    "business_capability": "",
    "objects": [],
    "actors": [],
    "capabilities": [],
    "technical_requirements": [],
    "search_queries": [],
    "avoid_platform_security": False,
    "in_scope": [],
    "out_of_scope": [],
    "discovery_questions": [],
    "separate_detection_from_policy": False,
}


def _blob(item: dict) -> str:
    return " ".join(
        str(part or "")
        for part in (
            item.get("title"),
            item.get("source_type"),
            item.get("authority"),
            item.get("url"),
            item.get("document_path"),
            item.get("excerpt"),
        )
    )


def extract_objects(message: str) -> list[str]:
    found: list[str] = []
    lower = message.lower()
    lower = re.sub(r"\baccount (?:owner|team|executive)\b", " ", lower)
    for name in STANDARD_OBJECTS:
        key = name.lower()
        canonical = OBJECT_ALIASES.get(key, name.replace(" ", ""))
        if re.search(rf"\b{re.escape(key)}\b", lower) and canonical not in found:
            found.append(canonical)
    if not found and re.search(r"\b(subject(?: line)?|support rep|service cloud)\b", lower):
        found.append("Case")
    return found


def extract_actors(message: str) -> list[str]:
    actors: list[str] = []
    lower = message.lower()
    mapping = (
        ("support rep", "support representative"),
        ("customer support", "support representative"),
        ("sales rep", "sales representative"),
        ("agent", "agent"),
        ("customer", "customer"),
    )
    for token, label in mapping:
        if token in lower and label not in actors:
            actors.append(label)
    return actors


def is_troubleshooting_question(message: str) -> bool:
    return bool(TROUBLESHOOTING_RE.search(message))


def is_fact_lookup(message: str) -> bool:
    """True for factual data-access questions ("how do I get the <value>").

    These are named-subject retrievals, not solution/architecture designs. Kept
    separate so the planner does not capability-translate them (the drift that
    turned "Live Agent queue position" into "assignment rules").
    """
    if is_troubleshooting_question(message):
        return False
    if not FACT_LOOKUP_RE.search(message):
        return False
    if IMPLEMENT_RE.search(message) or _DESIGN_OVERRIDE_RE.search(message):
        return False
    if UI_SIGNAL.search(message) or NOTIFY_SIGNAL.search(message) or CLASSIFY_SIGNAL.search(message):
        return False
    return True


def is_definition_shaped(message: str) -> bool:
    lower = message.lower().strip()
    return lower.startswith(DEFINITION_PREFIXES)


def is_business_requirement(message: str) -> bool:
    """True when the user stated an outcome, not a named Salesforce product."""
    if is_troubleshooting_question(message):
        return False
    if is_fact_lookup(message):
        return False
    if is_definition_shaped(message) and not IMPLEMENT_RE.search(message):
        return False
    return bool(HOW_TO_RE.search(message) or IMPLEMENT_RE.search(message))


def is_solution_question(message: str) -> bool:
    """How-to / implement / design questions a solution architect should intake."""
    if is_troubleshooting_question(message):
        return False
    if is_fact_lookup(message):
        return False
    lower = message.lower().strip()
    if is_definition_shaped(message):
        return bool(IMPLEMENT_RE.search(message) or HOW_TO_RE.search(message))
    if HOW_TO_RE.search(message) or IMPLEMENT_RE.search(message):
        return True
    if re.search(
        r"\b(architecture|well-?architected|decision guide|should we use|"
        r"should i use|vs apex|mcp or api|ldv|large data)\b",
        lower,
    ):
        return True
    return any(token in lower for token in ("500000", "500,000", "integrat"))


def _detect_capabilities(message: str) -> list[str]:
    caps: list[str] = []

    def add(name: str) -> None:
        if name not in caps:
            caps.append(name)

    if CLASSIFY_SIGNAL.search(message) and (
        TEXT_SIGNAL.search(message) or extract_objects(message)
    ):
        add("text_classification")
    if UI_SIGNAL.search(message):
        add("record_ui")
    if AUTOMATION_SIGNAL.search(message) or "text_classification" in caps:
        add("record_automation")
    if ASSIGN_SIGNAL.search(message):
        add("assignment")
    if NOTIFY_SIGNAL.search(message) and not PLATFORM_SECURITY_LANGUAGE.search(message):
        add("notification")
    if FIELD_SIGNAL.search(message):
        add("data_model")
    if INTEGRATION_SIGNAL.search(message):
        add("integration")
    if SCALE_SIGNAL.search(message):
        add("scale")
    if PLATFORM_SECURITY_LANGUAGE.search(message) and not PROCESS_LANGUAGE.search(message):
        add("platform_security")
    if "text_classification" in caps and "data_model" not in caps:
        add("data_model")
    if "record_ui" in caps and "data_model" not in caps:
        add("data_model")
    return caps


def _collapse_repeated_words(query: str) -> str:
    """Collapse adjacent duplicate words (case-insensitive).

    When no concrete object is known, `target` defaults to "Salesforce" and the
    templates already prefix "Salesforce", yielding "Salesforce Salesforce
    assignment rules". Dedup adjacent repeats so retrieval sees a clean query.
    """
    words = query.split()
    out: list[str] = []
    for word in words:
        if out and out[-1].lower() == word.lower():
            continue
        out.append(word)
    return " ".join(out)


def _queries_for(capability: str, obj: str | None) -> list[str]:
    target = obj or "Salesforce"
    named = NAMED_SEARCHES.get((capability, obj or ""))
    if named:
        extra = {
            "text_classification": [f"Salesforce record-triggered Flow vs Apex {target}"],
            "record_ui": [f"Lightning record page {target} highlight panel"],
        }.get(capability, [])
        return [_collapse_repeated_words(q) for q in named + extra]
    generic = {
        "text_classification": [
            f"Salesforce {target} classification Subject Description",
            f"Salesforce record-triggered Flow vs Apex {target}",
            "Salesforce classification with AI options key considerations",
        ],
        "scoring": [f"Salesforce {target} scoring"],
        "record_ui": [
            f"Lightning record page {target} highlight panel",
            f"Salesforce compact layout {target} field",
        ],
        "record_automation": [f"Salesforce record-triggered Flow vs Apex {target}"],
        "assignment": [f"Salesforce {target} assignment rules"],
        "notification": [f"Salesforce custom notification {target}", f"Salesforce email alert {target}"],
        "data_model": [f"Salesforce custom field {target}", f"Salesforce Object Manager {target} field"],
        "integration": [f"Salesforce {target} integration patterns", "Salesforce REST API named credential"],
        "scale": [
            "Salesforce Bulk API 2.0 large data volume",
            "Batch Apex governor limits async processing",
        ],
        "platform_security": [
            "Salesforce Shield Event Monitoring",
            "Salesforce Security Center Threat Detection",
        ],
    }
    return [_collapse_repeated_words(q) for q in generic.get(capability, [f"Salesforce {target} {capability}"])]


def _technical_requirements(capabilities: list[str], objects: list[str], actors: list[str]) -> list[str]:
    obj = objects[0] if objects else "the record"
    actor = actors[0] if actors else "the user"
    labels = {
        "data_model": f"Persist the outcome on {obj} (fields the process can key off).",
        "text_classification": f"Derive a classification from {obj} text (Subject/Description or similar).",
        "record_automation": f"Evaluate {obj} on create/update with record-triggered automation (Flow vs Apex).",
        "record_ui": f"Make the result visible to {actor} on the Lightning record page.",
        "assignment": f"Route or assign the {obj} using assignment or Omni-Channel.",
        "notification": f"Notify {actor} when the condition is met.",
        "integration": f"Integrate {obj} with the external system using a documented integration pattern.",
        "scale": "Use a high-volume data pattern (Bulk API, Batch, Queueable) rather than record-by-record UI API loops.",
        "platform_security": "Use org/session security products (Shield, Event Monitoring, Security Center).",
        "scoring": f"Score {obj} with the documented Einstein/scoring feature if it exists.",
    }
    return [labels[name] for name in capabilities if name in labels]


def _avoid_platform_security(message: str, capabilities: list[str]) -> bool:
    if "platform_security" in capabilities:
        return False
    if PLATFORM_SECURITY_LANGUAGE.search(message) and not PROCESS_LANGUAGE.search(message):
        return False
    crm_caps = {"text_classification", "record_ui", "record_automation", "data_model", "notification", "assignment"}
    if crm_caps.intersection(capabilities) and (
        PROCESS_LANGUAGE.search(message) or extract_objects(message)
    ):
        return True
    return False


def _business_capability(
    objects: list[str],
    actors: list[str],
    capabilities: list[str],
) -> str:
    obj = objects[0] if objects else "the business record"
    actor = actors[0] if actors else "the user"
    if "text_classification" in capabilities and "record_ui" in capabilities:
        return (
            f"Identify material {obj} situations early enough that {actor} "
            "can follow the appropriate handling path."
        )
    if "text_classification" in capabilities:
        return f"Identify and structure material {obj} situations from unstructured content."
    if "record_ui" in capabilities:
        return (
            f"Make a material {obj} condition impossible for {actor} to miss "
            "without alarming every record."
        )
    if "notification" in capabilities:
        return f"Notify {actor} when a material {obj} condition occurs."
    if "integration" in capabilities:
        return f"Exchange {obj} data with an external system using a documented pattern."
    if "scale" in capabilities:
        return f"Process {obj} volume without per-record UI API loops."
    return f"Design an operable {obj} capability that can evolve with the business."


def _scope(capabilities: list[str]) -> tuple[list[str], list[str]]:
    in_scope = []
    mapping = {
        "data_model": "Persist structured outcomes the process can key off",
        "text_classification": "Detect signals from unstructured record content",
        "record_automation": "Evaluate the record on create/update",
        "record_ui": "Surface the outcome to the user on the record",
        "assignment": "Route or assign after the outcome is known",
        "notification": "Notify a user after the outcome is known",
        "integration": "Connect to an external system",
        "scale": "Handle volume",
        "scoring": "Score the record",
        "platform_security": "Org/session security",
    }
    for name in capabilities:
        if name in mapping:
            in_scope.append(mapping[name])
    out_of_scope = [
        "Downstream playbooks, legal process, and customer-response operations unless the user asked to automate them",
        "Choosing a Salesforce AI product before taxonomy, labelled data, and error-cost are known",
    ]
    return in_scope, out_of_scope


def _discovery_questions(capabilities: list[str], objects: list[str]) -> list[str]:
    obj = objects[0] if objects else "the record"
    questions = [
        "What is the business capability in one sentence (outcome, not a Salesforce feature)?",
        "What decisions does this capability make, and which error is more expensive: miss or false alarm?",
        "What is in scope versus downstream process (playbook, routing, customer response)?",
    ]
    if "text_classification" in capabilities or "scoring" in capabilities:
        questions.extend(
            [
                f"Is the {obj} outcome a taxonomy (type, severity, evidence, confidence) or a single flag?",
                "Should probabilistic detection be separated from deterministic business policy?",
                f"What is the input boundary for {obj} today, and what might be added later (comments, email, history, account)?",
                "Is there enough consistently labelled history to support predictive classification?",
                "Which licensed Salesforce classification options exist (rules, Einstein, generative, Data Cloud)?",
            ]
        )
    if "record_ui" in capabilities:
        questions.append(
            "How does the user see why a condition fired, not only that it fired?"
        )
    if "data_model" in capabilities:
        questions.append(
            f"Can one {obj} have multiple concurrent outcomes that need their own lifecycle, or are fields on {obj} enough?"
        )
    questions.append(
        "What audit, override, and data-residency constraints apply before customer content is sent to an AI service?"
    )
    return questions


def heuristic_plan(message: str) -> dict[str, Any]:
    if not is_solution_question(message):
        return dict(EMPTY_PLAN)
    objects = extract_objects(message)
    actors = extract_actors(message)
    capabilities = _detect_capabilities(message)
    skip_raw = is_business_requirement(message)
    if skip_raw and not capabilities:
        capabilities = ["data_model", "record_automation"]
    buckets = [_queries_for(cap, objects[0] if objects else None) for cap in capabilities]
    queries: list[str] = []
    index = 0
    while True:
        added = False
        for bucket in buckets:
            if index < len(bucket):
                query = bucket[index]
                if query not in queries:
                    queries.append(query)
                added = True
        if not added:
            break
        index += 1
    in_scope, out_of_scope = _scope(capabilities)
    full_protocol = skip_raw and (
        len(capabilities) >= 2 or "text_classification" in capabilities or "scoring" in capabilities
    )
    return {
        "active": True,
        "skip_raw": skip_raw,
        "full_protocol": full_protocol,
        "business_outcome": message.strip(),
        "business_capability": _business_capability(objects, actors, capabilities),
        "objects": objects,
        "actors": actors,
        "capabilities": capabilities,
        "technical_requirements": _technical_requirements(capabilities, objects, actors),
        "search_queries": queries,
        "avoid_platform_security": _avoid_platform_security(message, capabilities),
        "in_scope": in_scope,
        "out_of_scope": out_of_scope,
        "discovery_questions": _discovery_questions(capabilities, objects) if full_protocol else [],
        "separate_detection_from_policy": "text_classification" in capabilities,
    }


def merge_llm_plan(heuristic: dict[str, Any], payload: dict) -> dict[str, Any]:
    plan = dict(heuristic)
    objects = [str(item) for item in (payload.get("objects") or []) if str(item).strip()]
    for obj in objects:
        canonical = OBJECT_ALIASES.get(obj.lower(), obj)
        if canonical not in plan["objects"]:
            plan["objects"].append(canonical)
    queries = [str(item).strip() for item in (payload.get("search_queries") or []) if str(item).strip()]
    merged_queries: list[str] = []
    for query in queries + plan["search_queries"]:
        if query and query not in merged_queries:
            merged_queries.append(query)
    if merged_queries:
        plan["search_queries"] = merged_queries
    requirements = [
        str(item).strip()
        for item in (payload.get("technical_requirements") or [])
        if str(item).strip()
    ]
    if requirements:
        plan["technical_requirements"] = requirements
    capability = str(payload.get("business_capability") or "").strip()
    if capability:
        plan["business_capability"] = capability
    questions = [
        str(item).strip()
        for item in (payload.get("discovery_questions") or [])
        if str(item).strip()
    ]
    if questions:
        plan["discovery_questions"] = questions
        plan["full_protocol"] = True
    if payload.get("separate_detection_from_policy") is True:
        plan["separate_detection_from_policy"] = True
    capabilities = [
        str(item).strip()
        for item in (payload.get("capabilities") or [])
        if str(item).strip()
    ]
    for name in capabilities:
        if name not in plan["capabilities"]:
            plan["capabilities"].append(name)
    for key in ("in_scope", "out_of_scope"):
        items = [str(item).strip() for item in (payload.get(key) or []) if str(item).strip()]
        if items:
            plan[key] = items
    avoid = payload.get("avoid_products") or []
    if any("security center" in str(item).lower() or "threat detection" in str(item).lower() for item in avoid):
        plan["avoid_platform_security"] = True
    plan["active"] = True
    return plan


def is_avoided_evidence(plan: dict[str, Any], item: dict) -> bool:
    if not plan.get("avoid_platform_security"):
        return False
    return bool(PLATFORM_SECURITY_EVIDENCE.search(_blob(item)))


def has_capability_evidence(plan: dict[str, Any], evidence: list[dict]) -> bool:
    needed = [name for name in plan.get("capabilities") or [] if name in EVIDENCE_OK]
    if not needed:
        return False
    for item in evidence:
        if not item.get("excerpt"):
            continue
        blob = _blob(item)
        if any(EVIDENCE_OK[name].search(blob) for name in needed):
            return True
    return False


def evidence_gap(plan: dict[str, Any], evidence: list[dict]) -> dict[str, Any]:
    if not plan.get("active") or not plan.get("skip_raw"):
        return {"needs_rescue": False, "rescue_queries": [], "wrong_family": False}
    aligned = has_capability_evidence(plan, evidence)
    wrong = any(is_avoided_evidence(plan, item) for item in evidence)
    return {
        "needs_rescue": not aligned,
        "rescue_queries": list(plan.get("search_queries") or []),
        "wrong_family": wrong,
    }


def drop_avoided_evidence(plan: dict[str, Any], evidence: list[dict]) -> list[dict]:
    if not evidence or not plan.get("avoid_platform_security"):
        return evidence
    kept = [item for item in evidence if not is_avoided_evidence(plan, item)]
    if kept and len(kept) < len(evidence):
        evidence[:] = kept
    return evidence
