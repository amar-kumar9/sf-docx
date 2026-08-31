import json

import app
import prompts


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, response_map=None, default_response="[]"):
        self.response_map = response_map or {}
        self.default_response = default_response
        self.invocations = []

    def invoke(self, messages):
        system_prompt = messages[0].content
        human_prompt = messages[1].content
        self.invocations.append((system_prompt, human_prompt))
        content = self.default_response
        for key, value in self.response_map.items():
            if key in system_prompt:
                content = value
                break
        return FakeResponse(content)


class ExplodingLLM:
    def invoke(self, messages):
        raise RuntimeError("connection error")


def test_current_governor_limit_temporal_context_and_queries():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "limits",
                    "question_type": "current_fact",
                    "topics": ["Apex heap size"],
                    "salesforce_features": ["Apex"],
                    "is_salesforce_specific": True,
                    "requires_documentation": True,
                    "requires_current_docs": True,
                    "temporal_validation_required": True,
                    "requested_release": None,
                    "requires_multiple_sources": False,
                    "requires_code_analysis": False,
                    "requires_architecture_analysis": False,
                    "research_depth": "quick",
                    "model_tier": "fast",
                }
            ),
            app.TEMPORAL_FACT_VALIDATION_PROMPT: json.dumps(
                {
                    "temporal_validation_required": True,
                    "current_docs_required": True,
                    "historical_release_requested": False,
                    "requested_release": None,
                    "reason": "Heap size changes over time",
                }
            ),
            app.QUERY_PLANNER_PROMPT: json.dumps(["Apex heap size current governor limit"]),
        }
    )

    intent = app.classify_intent("What is the Apex heap size?", llm)
    temporal = app.validate_temporal_context("What is the Apex heap size?", intent, llm)
    queries = app.build_search_queries("What is the Apex heap size?", intent, llm)

    assert intent["requires_current_docs"] is True
    assert temporal["temporal_validation_required"] is True
    assert temporal["current_docs_required"] is True
    assert any("current Salesforce documentation" in q for q in queries)


def test_historical_release_keeps_release_scope():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "limits",
                    "question_type": "current_fact",
                    "topics": ["Apex heap size"],
                    "salesforce_features": ["Apex"],
                    "is_salesforce_specific": True,
                    "requires_documentation": True,
                    "requires_current_docs": False,
                    "temporal_validation_required": True,
                    "requested_release": None,
                    "requires_multiple_sources": False,
                    "requires_code_analysis": False,
                    "requires_architecture_analysis": False,
                    "research_depth": "quick",
                    "model_tier": "fast",
                }
            ),
            app.TEMPORAL_FACT_VALIDATION_PROMPT: json.dumps(
                {
                    "temporal_validation_required": True,
                    "current_docs_required": False,
                    "historical_release_requested": True,
                    "requested_release": "Winter '26",
                    "reason": "User asked for a historical release",
                }
            ),
            app.QUERY_PLANNER_PROMPT: json.dumps(["Apex heap size Winter '26 release notes"]),
        }
    )

    intent = app.classify_intent("What was the Apex heap size in Winter '26?", llm)
    temporal = app.validate_temporal_context("What was the Apex heap size in Winter '26?", intent, llm)
    queries = app.build_search_queries("What was the Apex heap size in Winter '26?", intent, llm)

    assert temporal["historical_release_requested"] is True
    assert temporal["requested_release"] == "Winter '26"
    assert any("Winter '26" in q for q in queries)
    assert not intent["requires_current_docs"]


def test_architecture_prompt_explicitly_prefers_bullets_over_tables():
    assert "Prefer bullets for single recommendations" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "Never merge headers into one cell" in prompts.RESEARCH_SYNTHESIS_PROMPT


def test_fallback_content_detection_distinguishes_provider_failure():
    assert app._is_fallback_content(
        "## Answer\n[Unverified] I could not generate a verified answer because the local model provider was unavailable."
    ) is True
    assert app._is_fallback_content(
        "## Answer\n[Unverified] I could not confirm the current Salesforce value from authoritative Salesforce documentation."
    ) is False


def test_current_api_version_flags_temporal_validation():
    temporal = app._determine_temporal_context("What is the latest Salesforce API version?", {})

    assert temporal["temporal_validation_required"] is True
    assert temporal["current_docs_required"] is True


def test_current_fact_retrieval_uses_standard_budget_even_when_classifier_says_quick():
    intent = {
        "intent": "release",
        "question_type": "current_fact",
        "requires_documentation": True,
        "research_depth": "quick",
    }
    temporal = {
        "temporal_validation_required": True,
        "current_docs_required": True,
        "historical_release_requested": False,
        "requested_release": None,
    }
    seen = {"searches": 0, "fetches": 0}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["query one", "query two"]

        def fake_search(query):
            seen["searches"] += 1
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "/docs/current-api-version\nCurrent API version docs excerpt.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": "/docs/current-api-version",
            }

        def fake_fetch(path):
            seen["fetches"] += 1
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "Fetched current API version docs.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": path,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = fake_fetch

        evidence = app.retrieve_evidence("What is the latest Salesforce API version?", intent, object(), temporal)

        assert seen["searches"] == 1
        assert seen["fetches"] == 1
        assert len(evidence) == 2
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_current_fact_retrieval_keeps_searching_until_authoritative_source():
    intent = {
        "intent": "release",
        "question_type": "current_fact",
        "requires_documentation": True,
        "research_depth": "quick",
    }
    temporal = {
        "temporal_validation_required": True,
        "current_docs_required": True,
        "historical_release_requested": False,
        "requested_release": None,
    }
    seen = {"searches": 0}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["query one", "query two"]

        def fake_search(query):
            seen["searches"] += 1
            if seen["searches"] == 1:
                return {
                    "title": "MuleSoft Salesforce Connector Release Notes",
                    "excerpt": "Supported Salesforce API versions through v47.0.",
                    "source_type": "Salesforce Documentation",
                    "authority": "Salesforce Documentation",
                    "is_authoritative": True,
                }
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "Official current API version docs.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = lambda path: {}

        evidence = app.retrieve_evidence("What is the latest Salesforce API version?", intent, object(), temporal)

        assert seen["searches"] == 2
        assert len(evidence) == 2
        assert any("MuleSoft" in item["title"] for item in evidence)
        assert any("Salesforce Developer Documentation" in item["title"] for item in evidence)
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_historical_release_retrieval_uses_standard_budget_for_quick_depth():
    intent = {
        "intent": "release",
        "question_type": "current_fact",
        "requires_documentation": True,
        "research_depth": "quick",
    }
    temporal = {
        "temporal_validation_required": True,
        "current_docs_required": False,
        "historical_release_requested": True,
        "requested_release": "Winter '27",
    }
    seen = {"searches": 0, "fetches": 0}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["query one", "query two"]

        def fake_search(query):
            seen["searches"] += 1
            if seen["searches"] == 1:
                return {
                    "title": "ApexTestResultLimits",
                    "excerpt": "Field reference for Apex test result limits, documentation overview and notes.",
                    "source_type": "Salesforce Documentation",
                    "authority": "Salesforce Documentation",
                    "is_authoritative": True,
                    "document_path": "/docs/apex-test-result-limits",
                }
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "/docs/winter-27\nApex heap size Winter 27 release notes excerpt.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": "/docs/winter-27",
            }

        def fake_fetch(path):
            seen["fetches"] += 1
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "Fetched Winter '27 docs.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": path,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = fake_fetch

        evidence = app.retrieve_evidence("What was the Apex heap size in Winter '27?", intent, object(), temporal)

        assert seen["searches"] == 2
        assert seen["fetches"] == 1
        assert len(evidence) == 3
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_relevance_gate_ignores_boilerplate_terms_and_keeps_searching():
    intent = {
        "intent": "release",
        "question_type": "current_fact",
        "requires_documentation": True,
        "research_depth": "quick",
        "topics": ["Apex heap size"],
        "salesforce_features": ["Apex"],
    }
    temporal = {
        "temporal_validation_required": True,
        "current_docs_required": False,
        "historical_release_requested": True,
        "requested_release": "Winter '27",
    }
    seen = {"searches": 0, "fetches": 0}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["Apex heap size governor limit", "Apex heap size field reference"]

        def fake_search(query):
            seen["searches"] += 1
            if seen["searches"] == 1:
                return {
                    "title": "ApexTestResultLimits",
                    "excerpt": "Field reference for Apex test result limits, documentation overview and notes.",
                    "source_type": "Salesforce Documentation",
                    "authority": "Salesforce Documentation",
                    "is_authoritative": True,
                    "document_path": "/docs/apex-test-result-limits",
                }
            return {
                "title": "Apex Governor Limits",
                "excerpt": "Governor limits for Apex heap size in Winter '27 release notes.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": "/docs/apex-governor-limits",
            }

        def fake_fetch(path):
            seen["fetches"] += 1
            return {
                "title": "Apex Governor Limits",
                "excerpt": "Fetched Apex governor limits.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": path,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = fake_fetch

        evidence = app.retrieve_evidence("Whats the apex heap size asper winter 27 docuementation", intent, object(), temporal)

        assert seen["searches"] == 2
        assert seen["fetches"] == 1
        assert any(item.get("title") == "Apex Governor Limits" for item in evidence)
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_mcp_search_parses_document_metadata_from_payload():
    original_call_mcp = app._call_mcp
    try:
        app._call_mcp = lambda tool_name, arguments: (
            json.dumps(
                {
                    "chunks": [
                        {
                            "content": "CaseClassification\n\nEnables Einstein Case Classification.",
                            "documentPath": "/docs/data/caseclassification",
                            "title": "Case Classification",
                            "url": "https://developer.salesforce.com/docs/data/caseclassification",
                        }
                    ]
                }
            ),
            True,
        )

        result = app.mcp_search("case classification")

        assert result["title"] == "Case Classification"
        assert result["document_path"] == "/docs/data/caseclassification"
        assert result["url"].startswith("https://developer.salesforce.com")
        assert "Enables Einstein Case Classification" in result["excerpt"]
    finally:
        app._call_mcp = original_call_mcp


def test_retrieve_evidence_fetches_using_document_path_metadata():
    intent = {
        "intent": "research",
        "question_type": "explanation",
        "requires_documentation": True,
        "research_depth": "standard",
        "topics": ["case classification"],
        "salesforce_features": ["Case"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    seen = {"fetch_paths": []}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["case classification overview"]
        app.mcp_search = lambda query: {
            "title": "Case Classification",
            "excerpt": "Enables Einstein Case Classification.",
            "source_type": "Salesforce Documentation",
            "authority": "Salesforce Documentation",
            "is_authoritative": True,
            "document_path": "/docs/data/caseclassification",
        }

        def fake_fetch(path):
            seen["fetch_paths"].append(path)
            return {
                "title": "Case Classification",
                "excerpt": "Full Case Classification documentation.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": path,
            }

        app.mcp_fetch = fake_fetch

        evidence = app.retrieve_evidence("What is Case Classification?", intent, object(), temporal)

        assert seen["fetch_paths"] == ["/docs/data/caseclassification"]
        assert any(item.get("document_path") == "/docs/data/caseclassification" for item in evidence)
        assert len(evidence) == 2
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_agentforce_requires_documentation_and_temporal_planning():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "research",
                    "question_type": "explanation",
                    "topics": ["Agentforce Coworker", "benefits", "use cases"],
                    "salesforce_features": ["Agentforce", "Agentforce Coworker"],
                    "is_salesforce_specific": True,
                    "requires_documentation": True,
                    "requires_current_docs": False,
                    "temporal_validation_required": False,
                    "requested_release": None,
                    "requires_multiple_sources": False,
                    "requires_code_analysis": False,
                    "requires_architecture_analysis": False,
                    "research_depth": "quick",
                    "model_tier": "fast",
                }
            ),
            app.TEMPORAL_FACT_VALIDATION_PROMPT: json.dumps(
                {
                    "temporal_validation_required": False,
                    "current_docs_required": False,
                    "historical_release_requested": False,
                    "requested_release": None,
                    "reason": "Concept question",
                }
            ),
            app.QUERY_PLANNER_PROMPT: json.dumps(["Agentforce Coworker benefits use cases"]),
        }
    )

    intent = app.classify_intent("Agentforce Coworker Benefits and Use Cases", llm)
    temporal = app.validate_temporal_context("Agentforce Coworker Benefits and Use Cases", intent, llm)
    queries = app.build_search_queries("Agentforce Coworker Benefits and Use Cases", intent, llm)

    assert intent["is_salesforce_specific"] is True
    assert intent["requires_documentation"] is True
    assert temporal["temporal_validation_required"] is False
    assert len(queries) == 1
    assert "current Salesforce documentation" in queries[0]


def test_architecture_case_routes_multiple_queries_and_conflict_resolution():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "architecture",
                    "question_type": "design",
                    "topics": ["bulk data", "data loading", "high volume", "LDV"],
                    "salesforce_features": ["Bulk API", "Batch Apex", "Queueable Apex"],
                    "is_salesforce_specific": True,
                    "requires_documentation": True,
                    "requires_current_docs": False,
                    "temporal_validation_required": True,
                    "requested_release": None,
                    "requires_multiple_sources": True,
                    "requires_code_analysis": False,
                    "requires_architecture_analysis": True,
                    "research_depth": "deep",
                    "model_tier": "reasoning",
                }
            ),
            app.TEMPORAL_FACT_VALIDATION_PROMPT: json.dumps(
                {
                    "temporal_validation_required": True,
                    "current_docs_required": True,
                    "historical_release_requested": False,
                    "requested_release": None,
                    "reason": "Architecture may depend on current limits",
                }
            ),
            app.QUERY_PLANNER_PROMPT: json.dumps(
                [
                    "Salesforce Bulk API 2.0 large data volume",
                    "Batch Apex governor limits async processing",
                    "Salesforce LDV best practices",
                ]
            ),
            app.EVIDENCE_CONFLICT_RESOLUTION_PROMPT: json.dumps(
                {
                    "status": "no_conflict",
                    "applicable_fact": "No conflicting facts detected",
                    "conflicts": [],
                }
            ),
        }
    )

    intent = app.classify_intent(
        "We process 5 million records per day. What Salesforce integration architecture should we use?",
        llm,
    )
    temporal = app.validate_temporal_context(
        "We process 5 million records per day. What Salesforce integration architecture should we use?",
        intent,
        llm,
    )
    queries = app.build_search_queries(
        "We process 5 million records per day. What Salesforce integration architecture should we use?",
        intent,
        llm,
    )
    conflict = app.resolve_evidence_conflicts(
        [
            {
                "title": "Doc A",
                "document_path": "/a",
                "excerpt": "Apex heap size is 6 MB synchronous.",
                "authority": "Salesforce Developer Documentation",
            },
            {
                "title": "Doc B",
                "document_path": "/b",
                "excerpt": "Apex heap size is 10 MB synchronous.",
                "authority": "Salesforce Release Notes",
            },
        ],
        intent,
        llm,
        "We process 5 million records per day. What Salesforce integration architecture should we use?",
    )
    normalized = app.normalize_evidence(
        [
            {
                "title": "Doc A",
                "document_path": "/a",
                "excerpt": "Apex heap size is 6 MB synchronous.",
                "authority": "Salesforce Developer Documentation",
                "release": "Winter '26",
                "api_version": "63.0",
                "published_date": "2026-01-15",
                "last_updated": "2026-01-20",
                "query": "Apex heap size",
                "relevance": 0.91,
            }
        ]
    )

    assert intent["requires_architecture_analysis"] is True
    assert temporal["temporal_validation_required"] is True
    assert len(queries) == 3
    assert conflict["status"] == "no_conflict"
    assert normalized[0]["authority"] == "Salesforce Developer Documentation"
    assert normalized[0]["release"] == "Winter '26"


def test_source_trust_prioritizes_salesforce_docs_over_connector_notes():
    evidence = app.normalize_evidence(
        [
            {
                "title": "MuleSoft Salesforce Connector Release Notes",
                "excerpt": "Supported Salesforce API versions through v47.0.",
            },
            {
                "title": "Salesforce Developer Documentation",
                "excerpt": "Current Salesforce API version information.",
            },
        ]
    )

    ordered = app._sort_evidence_by_trust(evidence)

    assert ordered[0]["title"] == "Salesforce Developer Documentation"
    assert app._has_authoritative_salesforce_source(evidence) is True


def test_source_trust_rejects_connector_only_current_fact_evidence():
    evidence = app.normalize_evidence(
        [
            {
                "title": "MuleSoft Salesforce Connector Release Notes",
                "excerpt": "Supported Salesforce API versions through v47.0.",
            }
        ]
    )

    assert app._has_authoritative_salesforce_source(evidence) is False


def test_source_trust_counts_official_mcp_style_evidence():
    evidence = app.normalize_evidence(
        [
            {
                "title": "",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "excerpt": "Salesforce Developer Documentation for Apex governor limits.",
            }
        ]
    )

    assert app._has_authoritative_salesforce_source(evidence) is True
    assert app._sort_evidence_by_trust(evidence)[0]["is_authoritative"] is True


def test_current_fact_filter_keeps_official_docs_and_drops_connector_notes():
    evidence = app.normalize_evidence(
        [
            {
                "title": "MuleSoft Salesforce Connector Release Notes",
                "excerpt": "Supported Salesforce API versions through v47.0.",
            },
            {
                "title": "Salesforce Developer Documentation",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "excerpt": "Official documentation for current Salesforce API behavior.",
            },
        ]
    )

    filtered = app._filter_current_fact_evidence(evidence)

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Salesforce Developer Documentation"
    assert "v47.0" not in filtered[0]["excerpt"]


def test_current_fact_filter_ignores_body_noise_on_official_docs():
    evidence = app.normalize_evidence(
        [
            {
                "title": "Salesforce Developer Documentation",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "excerpt": "Official documentation references the Trailblazer Community blog in a related-links section.",
            }
        ]
    )

    filtered = app._filter_current_fact_evidence(evidence)

    assert len(filtered) == 1
    assert filtered[0]["title"] == "Salesforce Developer Documentation"


def test_topic_relevance_filters_out_case_queue_noise():
    message = "I would like to classify the cases based on the subject, description, and other fields."
    intent = {
        "topics": ["case classification", "subject", "description", "fields"],
        "salesforce_features": ["Case"],
    }
    evidence = app.normalize_evidence(
        [
            {
                "title": "Viewing Case Queues",
                "excerpt": "Salesforce creates a list view for each case queue you create.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
            }
        ]
    )
    terms = app._topic_terms(message, intent)

    assert app._is_relevant(evidence[0], terms) is False
    assert app._filter_relevant_evidence(evidence, terms) == []


def test_google_search_marks_official_salesforce_results_authoritative():
    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    original_get = app.requests.get
    original_api_key = app.GOOGLE_CSE_API_KEY
    original_cx = app.GOOGLE_CSE_CX
    original_enabled = app.GOOGLE_FALLBACK_ENABLED

    try:
        app.GOOGLE_CSE_API_KEY = "key"
        app.GOOGLE_CSE_CX = "cx"
        app.GOOGLE_FALLBACK_ENABLED = True

        def fake_get(url, params=None, timeout=None):
            return FakeResponse(
                {
                    "items": [
                        {
                            "title": "Case Classification",
                            "link": "https://developer.salesforce.com/docs/data/caseclassification",
                            "snippet": "Official Salesforce documentation.",
                        }
                    ]
                }
            )

        app.requests.get = fake_get
        results = app.google_search("case classification")

        assert len(results) == 1
        assert results[0]["is_authoritative"] is True
        assert results[0]["source_type"] == "Google Search Result"
        assert results[0]["url"].startswith("https://developer.salesforce.com")
    finally:
        app.requests.get = original_get
        app.GOOGLE_CSE_API_KEY = original_api_key
        app.GOOGLE_CSE_CX = original_cx
        app.GOOGLE_FALLBACK_ENABLED = original_enabled


def test_retrieve_evidence_can_supplement_mcp_with_google_results():
    intent = {
        "intent": "research",
        "question_type": "explanation",
        "requires_documentation": True,
        "research_depth": "quick",
        "topics": ["case classification"],
        "salesforce_features": ["Case"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_google_search = app.google_search
    original_enabled = app.GOOGLE_FALLBACK_ENABLED
    original_api_key = app.GOOGLE_CSE_API_KEY
    original_cx = app.GOOGLE_CSE_CX

    try:
        app.build_search_queries = lambda message, intent_data, llm: ["case classification overview"]
        app.mcp_search = lambda query: {
            "title": "Viewing Case Queues",
            "excerpt": "Queue list view details.",
            "source_type": "Salesforce Documentation",
            "authority": "Salesforce Documentation",
            "is_authoritative": True,
        }
        app.google_search = lambda query: [
            {
                "title": "Case Classification",
                "url": "https://developer.salesforce.com/docs/data/caseclassification",
                "excerpt": "Official Case Classification docs.",
                "source_type": "Google Search Result",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
            }
        ]
        app.GOOGLE_FALLBACK_ENABLED = True
        app.GOOGLE_CSE_API_KEY = "key"
        app.GOOGLE_CSE_CX = "cx"

        evidence = app.retrieve_evidence("classify cases based on subject and description", intent, object(), temporal)

        assert any(item.get("title") == "Case Classification" for item in evidence)
        assert any(item.get("title") == "Viewing Case Queues" for item in evidence)
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.google_search = original_google_search
        app.GOOGLE_FALLBACK_ENABLED = original_enabled
        app.GOOGLE_CSE_API_KEY = original_api_key
        app.GOOGLE_CSE_CX = original_cx


def test_mulesoft_marker_in_excerpt_rejects_current_fact_evidence():
    evidence = app.normalize_evidence(
        [
            {
                "title": "Salesforce Developer Documentation",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "excerpt": "This excerpt mentions MuleSoft API mappings for comparison.",
            }
        ]
    )

    assert app._is_non_authoritative_salesforce_evidence(evidence[0]) is True


def test_current_fact_short_circuits_when_only_non_authoritative_sources_exist():
    evidence = app.normalize_evidence(
        [
            {
                "title": "MuleSoft Salesforce Connector Release Notes",
                "excerpt": "Supported Salesforce API versions through v47.0.",
            }
        ]
    )
    intent = {
        "intent": "release",
        "question_type": "current_fact",
        "requires_current_docs": True,
    }
    temporal = {
        "temporal_validation_required": True,
        "current_docs_required": True,
        "historical_release_requested": False,
        "requested_release": None,
    }

    assert app._is_current_fact_request(intent, temporal) is True
    response = app._build_unverified_current_fact_response(
        "What is the latest Salesforce API version?",
        evidence,
        temporal,
    )

    assert "v47.0" not in response
    assert "[Unverified]" in response
    assert "Untitled source" not in response
    assert "No authoritative Salesforce source was found." in response


def test_grounding_check_can_rewrite_flagged_answers():
    llm = FakeLLM(
        response_map={
            app.GROUNDING_CHECK_PROMPT: "UNSUPPORTED: Apex heap size is 6 MB synchronous and 12 MB asynchronous.\nACTION: label_as_unverified",
            app.GROUNDING_REWRITE_PROMPT: "## Answer\n[Unverified] The retrieved evidence does not confirm the Apex heap size.",
        }
    )
    evidence = app.normalize_evidence(
        [
            {
                "title": "Salesforce Developer Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "excerpt": "Apex limits documentation excerpt.",
            }
        ]
    )

    answer, status = app.validate_grounding("The Apex heap size is 6 MB.", evidence, llm)

    assert status == "rewritten"
    assert "[Unverified]" in answer


def test_resilient_llm_falls_back_on_provider_failure():
    resp = app._invoke_resilient(
        ExplodingLLM(),
        [
            type("Msg", (), {"content": app.INTENT_CLASSIFIER_PROMPT})(),
            type("Msg", (), {"content": "What is the latest Salesforce API version?"})(),
        ],
        "intent",
        "What is the latest Salesforce API version?",
    )

    assert hasattr(resp, "content")
    assert '"intent": "research"' in resp.content
