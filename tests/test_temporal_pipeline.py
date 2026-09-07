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
    queries = app.build_search_queries("What is the Apex heap size?", intent, llm, temporal)

    assert intent["requires_current_docs"] is True
    assert temporal["temporal_validation_required"] is True
    assert temporal["current_docs_required"] is True
    assert any("governor limit" in q.lower() for q in queries)
    assert any("release note" in q.lower() for q in queries)


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
    queries = app.build_search_queries("What was the Apex heap size in Winter '26?", intent, llm, temporal)

    assert temporal["historical_release_requested"] is True
    assert temporal["requested_release"] == "Winter '26"
    assert any("Winter '26" in q for q in queries)
    assert not intent["requires_current_docs"]


def test_heuristic_heap_question_searches_release_notes_first():
    question = "Whats the apex heap size?"
    intent = app.classify_intent_heuristic(question)
    temporal = app.validate_temporal_context(question, intent, None)
    queries = app.build_search_queries(question, intent, None, temporal)
    assert intent["intent"] == "limits"
    assert temporal["current_docs_required"] is True
    assert any("release note" in query.lower() for query in queries)
    assert queries[0].lower().find("release note") != -1


def test_limits_cheatsheet_alone_is_not_enough_for_current_heap():
    question = "Whats the apex heap size?"
    intent = {"intent": "limits", "question_type": "current_fact", "research_depth": "quick"}
    cheatsheet = {
        "title": "Apex Governor Limits",
        "excerpt": "Total heap size 6 MB synchronous 12 MB asynchronous.",
        "url": "https://developer.salesforce.com/docs/atlas.en-us.salesforce_app_limits_cheatsheet.meta/salesforce_app_limits_cheatsheet/salesforce_app_limits_platform_apexgov.htm",
        "document_path": "salesforce_app_limits_cheatsheet/salesforce_app_limits_platform_apexgov.html",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    notes = {
        "title": "Salesforce Release Notes",
        "excerpt": "Winter release notes increased Apex heap size.",
        "url": "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm",
        "document_path": "release-notes/salesforce_release_notes.html",
        "source_type": "Salesforce Release Notes",
        "authority": "Salesforce Release Notes",
        "is_authoritative": True,
    }
    terms = app._topic_terms(question, intent)
    first = app._evidence_sufficiency_stage([cheatsheet], intent, question, terms)
    assert first["sufficient"] is False
    assert first["reason"] == "release_notes_missing"
    ranked = app._sort_evidence_by_trust([cheatsheet, notes])
    assert app._is_release_notes_evidence(ranked[0]) is True

    hub = {
        "title": "Salesforce Release Notes",
        "excerpt": "What's new in the current Salesforce release.",
        "url": "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm",
        "document_path": "release-notes/salesforce_release_notes.html",
        "source_type": "Salesforce Release Notes",
        "authority": "Salesforce Release Notes",
        "is_authoritative": True,
    }
    mixed = app._evidence_sufficiency_stage([hub, cheatsheet], intent, question, terms)
    assert mixed["sufficient"] is False
    assert mixed["reason"] == "release_notes_off_topic"
    covered = app._evidence_sufficiency_stage([notes, cheatsheet], intent, question, terms)
    assert covered["sufficient"] is True


def test_seasonal_queries_keep_a_statute_slot():
    question = "Whats the apex heap size?"
    intent = app.classify_intent_heuristic(question)
    temporal = app.validate_temporal_context(question, intent, None)
    queries = app.build_search_queries(question, intent, None, temporal)
    assert queries[0].lower() != "salesforce release notes current release"
    assert any("release note" in query.lower() for query in queries)
    assert any(
        "developer documentation" in query.lower() or "developer docs" in query.lower()
        for query in queries
    )


def test_atlas_citation_maps_to_standing_guide_path():
    url = (
        "https://developer.salesforce.com/docs/atlas.en-us.262.0.apexcode.meta/"
        "apexcode/apex_gov_limits.htm"
    )
    assert app._mcp_path_from_atlas_url(url) == "apexcode/apex_gov_limits.html"
    excerpt = (
        "Read up on Apex limits details in "
        f"[Execution Governors and Limits]({url})"
    )
    cited = app._cited_standing_guides(excerpt)
    assert cited
    assert cited[0]["document_path"] == "apexcode/apex_gov_limits.html"


def test_heap_question_ranks_governor_limits_guide_over_apex_toc():
    question = "Whats the apex heap size?"
    guide = {
        "title": "Execution Governors and Limits",
        "document_path": "apexcode/apex_gov_limits.html",
        "url": "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_gov_limits.htm",
        "excerpt": "Per-transaction Apex limits.",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    toc = {
        "title": "Apex Developer Guide",
        "document_path": "apexcode/apex_dev_guide.html",
        "url": "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_dev_guide.htm",
        "excerpt": "Apex is a strongly typed, object-oriented programming language.",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    ranked = sorted(
        [toc, guide],
        key=lambda item: app._seasonal_authority_rank(item, question),
        reverse=True,
    )
    assert ranked[0]["document_path"] == "apexcode/apex_gov_limits.html"
    assert app._statute_on_topic(guide, question) is True
    assert app._statute_on_topic(toc, question) is False


def test_cheatsheet_fetch_follows_cited_standing_guide():
    question = "Whats the apex heap size?"
    intent = {
        "intent": "limits",
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
    cheatsheet_path = "salesforce_app_limits_cheatsheet/salesforce_app_limits_platform_apexgov.html"
    statute_path = "apexcode/apex_gov_limits.html"
    statute_url = (
        "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/"
        "apexcode/apex_gov_limits.htm"
    )
    seen = {"fetch_paths": []}
    original_build = app.build_search_queries
    original_search = app.mcp_search
    original_fetch = app.mcp_fetch
    try:
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: [
            "Salesforce Apex heap size limit release notes"
        ]

        def fake_search(query):
            return {
                "title": "Apex Governor Limits",
                "excerpt": "Total heap size 6 MB synchronous 12 MB asynchronous.",
                "url": (
                    "https://developer.salesforce.com/docs/"
                    "atlas.en-us.salesforce_app_limits_cheatsheet.meta/"
                    "salesforce_app_limits_cheatsheet/"
                    "salesforce_app_limits_platform_apexgov.htm"
                ),
                "document_path": cheatsheet_path,
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
            }

        def fake_fetch(path):
            seen["fetch_paths"].append(path)
            if path == statute_path:
                return {
                    "title": "Execution Governors and Limits",
                    "excerpt": "Total heap size for the current release.",
                    "document_path": path,
                    "source_type": "Salesforce Documentation",
                    "authority": "Salesforce Documentation",
                    "is_authoritative": True,
                }
            return {
                "title": "Apex Governor Limits",
                "excerpt": (
                    "Total heap size 6 MB. Read up on Apex limits details in "
                    f"[Execution Governors and Limits]({statute_url})"
                ),
                "document_path": path,
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = fake_fetch
        evidence = app.retrieve_evidence(question, intent, object(), temporal)
        assert cheatsheet_path in seen["fetch_paths"]
        assert statute_path in seen["fetch_paths"]
        assert any(item.get("document_path") == statute_path for item in evidence)
    finally:
        app.build_search_queries = original_build
        app.mcp_search = original_search
        app.mcp_fetch = original_fetch


def test_architecture_prompt_explicitly_prefers_bullets_over_tables():
    assert "Prefer bullets for single recommendations" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "Never merge headers into one cell" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "Trusted" in prompts.ARCHITECTURE_REASONING_PROMPT
    assert "Easy" in prompts.ARCHITECTURE_REASONING_PROMPT
    assert "Adaptable" in prompts.ARCHITECTURE_REASONING_PROMPT
    assert "## Well-Architected" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "## Business capability" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "Detection vs decision" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "decision deferred" in prompts.RESEARCH_SYNTHESIS_PROMPT
    assert "consequences of being wrong" in prompts.SOLUTION_ARCHITECTURE_PROTOCOL
    assert "narrow Salesforce how-to" in prompts.SOLUTION_ARCHITECTURE_PROTOCOL_SHORT


def test_capability_design_synthesis_uses_full_protocol():
    intent = {
        "intent": "architecture",
        "question_type": "design",
        "requires_architecture_analysis": True,
    }
    prompt = app.build_synthesis_prompt(CASE_RISK_QUESTION, intent)
    assert "consequences of being wrong" in prompt
    assert "Separate detection from decision" in prompt
    assert "narrow Salesforce how-to" not in prompt


def test_narrow_how_to_synthesis_skips_full_protocol():
    intent = {
        "intent": "architecture",
        "question_type": "design",
        "requires_architecture_analysis": True,
    }
    prompt = app.build_synthesis_prompt("How do I create a custom field on Account?", intent)
    assert "narrow Salesforce how-to" in prompt
    assert "consequences of being wrong" not in prompt


def test_host_instructions_defer_product_choice_for_capability_design():
    plan = app.translate_requirement(
        CASE_RISK_QUESTION,
        {"intent": "architecture", "requires_architecture_analysis": True},
        None,
    )
    host = app._host_instructions_for_plan(plan)
    assert "Business capability:" in host
    assert "decision deferred" in host
    assert "Do not pick a product" in host
    pack = app._public_solution_plan(plan)
    assert pack["full_protocol"] is True
    assert pack["discovery_questions"]
    assert pack["business_capability"] != CASE_RISK_QUESTION


def test_diagnosis_queries_use_named_subject_not_the_repro():
    question = (
        "lightning-record-picker does not trigger search when pasting the same search term "
        "after clearing selection. Steps to reproduce: enter abcd, select a record, click X, "
        "paste abcd. Actual behavior: search is not triggered. Is there a way to fix this?"
    )
    intent = app.classify_intent_heuristic(question)
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries(question, intent, None, temporal)
    assert intent["intent"] == "troubleshooting"
    assert question not in queries
    assert all(len(query) <= 140 for query in queries)
    blob = " ".join(queries).lower()
    assert "lightning-record-picker" in blob
    assert "reference" in blob


def test_apex_diagnosis_uses_the_same_subject_rule():
    question = (
        "Database.executeBatch never calls the start method. "
        "Steps to reproduce. Is there a way to fix this?"
    )
    intent = app.classify_intent_heuristic(question)
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries(question, intent, None, temporal)
    assert intent["intent"] == "troubleshooting"
    blob = " ".join(queries)
    assert "Database.executeBatch" in blob
    assert "reference" in blob.lower()
    assert "lightning-record-picker" not in blob.lower()


def test_diagnosis_overview_is_not_sufficient_without_contract_or_symptom():
    question = (
        "lightning-record-picker does not trigger search when pasting the same term. "
        "Steps to reproduce. Is there a way to fix this?"
    )
    intent = {"intent": "troubleshooting", "research_depth": "standard"}
    overview = {
        "title": "Record Picker",
        "excerpt": "lightning-record-picker searches Salesforce records using the GraphQL wire adapter.",
        "url": "https://developer.salesforce.com/docs/platform/lightning-component-reference/guide/lightning-record-picker.html",
        "document_path": "lightning-record-picker.html",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    contract = {
        **overview,
        "excerpt": "clearSelection clears the selected record. This method does not clear the search term.",
    }
    terms = app._topic_terms(question, intent)
    first = app._evidence_sufficiency_stage([overview], intent, question, terms)
    assert first["sufficient"] is False
    assert first["reason"] == "diagnosis_uncovered"
    second = app._evidence_sufficiency_stage([overview, contract], intent, question, terms)
    assert second["sufficient"] is True


def test_soql_101_questions_get_bulkification_hint():
    intent = {
        "intent": "troubleshooting",
        "question_type": "diagnosis",
        "requires_code_analysis": True,
        "requires_documentation": True,
        "research_depth": "standard",
    }
    prompt = app.build_synthesis_prompt("what is best way to avoide soql 101 error", intent)

    assert "SOQL 101 / BULKIFICATION" in prompt or "SOQL/SOQL" in prompt
    assert "collect record Ids into a Set" in prompt or "collect Ids into a Set" in prompt


def test_definition_sufficiency_stage_requires_an_overview_page():
    intent = {
        "intent": "research",
        "question_type": "explanation",
        "requires_documentation": True,
    }
    evidence = [
        {
            "title": "Agentforce Coworker Review User Mapping",
            "excerpt": "Identity Resolution ruleset and user mapping.",
            "source_type": "Salesforce Documentation",
            "authority": "Salesforce Documentation",
            "is_authoritative": True,
            "document_path": "/docs/data/agentforce-coworker-review-user-mapping",
        }
    ]
    result = app._evidence_sufficiency_stage(
        evidence,
        intent,
        "What is Agentforce Coworker?",
        {"agentforce", "coworker"},
        require_authoritative=True,
    )

    assert result["sufficient"] is False
    assert result["reason"] == "definition_not_satisfied"


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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["query one", "query two"]

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

        assert seen["searches"] == 2
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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["query one", "query two"]

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


def test_retrieve_evidence_does_not_add_duplicate_temporal_queries():
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
    seen = {"queries": []}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch

    try:
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: [
            "What is the latest Salesforce API version? current Salesforce documentation release notes",
            "Salesforce API version latest release",
        ]

        def fake_search(query):
            seen["queries"].append(query)
            return {
                "title": "Salesforce Developer Documentation",
                "excerpt": "Current API version docs excerpt.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "document_path": "/docs/current-api-version",
            }

        app.mcp_search = fake_search
        app.mcp_fetch = lambda path: {}

        app.retrieve_evidence("What is the latest Salesforce API version?", intent, object(), temporal)

        assert seen["queries"][0] == (
            "What is the latest Salesforce API version? current Salesforce documentation release notes"
        )
        assert len(seen["queries"]) == len(set(seen["queries"]))
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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["query one", "query two"]

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
        assert seen["fetches"] == 2
        assert len(evidence) == 4
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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["Apex heap size governor limit", "Apex heap size field reference"]

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
        assert seen["fetches"] == 2
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


def test_mcp_search_keeps_separate_documents_from_mixed_chunks():
    original_call_mcp = app._call_mcp
    try:
        app._call_mcp = lambda tool_name, arguments: (
            json.dumps(
                {
                    "chunks": [
                        {
                            "content": "Cheatsheet heap row.",
                            "documentPath": "salesforce_app_limits_cheatsheet/apexgov.html",
                            "title": "Apex Governor Limits",
                            "url": "https://developer.salesforce.com/docs/cheatsheet",
                        },
                        {
                            "content": "Winter release notes increased Apex heap size.",
                            "documentPath": "release-notes/salesforce_release_notes.html",
                            "title": "Salesforce Release Notes",
                            "url": "https://help.salesforce.com/s/articleView?id=release-notes.salesforce_release_notes.htm",
                        },
                    ]
                }
            ),
            True,
        )
        result = app.mcp_search("Apex heap size")
        hits = result if isinstance(result, list) else [result]
        paths = {item["document_path"] for item in hits}
        assert "salesforce_app_limits_cheatsheet/apexgov.html" in paths
        assert "release-notes/salesforce_release_notes.html" in paths
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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["case classification overview"]
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
        assert len(evidence) >= 2
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
    queries = app.build_search_queries("Agentforce Coworker Benefits and Use Cases", intent, llm, temporal)

    assert intent["is_salesforce_specific"] is True
    assert intent["requires_documentation"] is True
    assert temporal["temporal_validation_required"] is False
    assert len(queries) == 1
    assert queries[0] == "Agentforce Coworker benefits use cases"


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
        temporal,
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
    assert len(queries) >= 3
    assert any("Well-Architected" in query for query in queries)
    assert any("Architecture Center" in query for query in queries)
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
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: ["case classification overview"]
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


def test_architect_salesforce_url_is_official_and_authoritative():
    url = "https://architect.salesforce.com/docs/architect/well-architected/guide/overview.html"
    evidence = app.normalize_evidence(
        [
            {
                "title": "Overview",
                "url": url,
                "excerpt": "Salesforce Well-Architected is organized around Trusted, Easy, and Adaptable.",
                "source_type": "Salesforce Documentation",
            }
        ]
    )

    assert app._is_official_salesforce_url(url) is True
    assert app._is_architecture_center_evidence(evidence[0]) is True
    assert app._has_authoritative_salesforce_source(evidence) is True


def test_architecture_queries_interleave_well_architected_boosters():
    llm = FakeLLM(
        response_map={
            app.QUERY_PLANNER_PROMPT: json.dumps(
                [
                    "Salesforce Bulk API 2.0 large data volume",
                    "Batch Apex governor limits async processing",
                    "Salesforce LDV best practices",
                ]
            ),
        }
    )
    intent = {
        "intent": "architecture",
        "question_type": "design",
        "requires_architecture_analysis": True,
        "research_depth": "deep",
        "topics": ["LDV", "bulk data"],
        "salesforce_features": ["Bulk API", "Batch Apex"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries(
        "We process 500000 records every night. What Salesforce architecture should we use?",
        intent,
        llm,
        temporal,
    )

    assert queries[0] == "Salesforce Bulk API 2.0 large data volume"
    assert "Salesforce Well-Architected Framework Trusted Easy Adaptable" in queries
    assert "Salesforce Architecture Center integration patterns" in queries
    assert len(queries) == 4


def test_definition_queries_do_not_force_architecture_center():
    llm = FakeLLM(default_response=json.dumps(["Salesforce Platform Events overview"]))
    intent = {
        "intent": "quick_fact",
        "question_type": "explanation",
        "requires_architecture_analysis": False,
        "research_depth": "quick",
        "topics": ["Platform Events"],
        "salesforce_features": ["Platform Events"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries("What is a Platform Event?", intent, llm, temporal)

    assert all("Well-Architected" not in query for query in queries)
    assert all("Architecture Center" not in query for query in queries)


def test_architecture_center_evidence_survives_topic_relevance_filter():
    intent = {
        "intent": "architecture",
        "requires_architecture_analysis": True,
        "topics": ["bulk", "queueable", "nightly"],
        "salesforce_features": ["Bulk API"],
    }
    item = {
        "title": "Overview",
        "url": "https://architect.salesforce.com/docs/architect/well-architected/guide/overview.html",
        "excerpt": "Trusted solutions protect stakeholders. Easy solutions deliver value fast.",
        "document_path": "well-architected/guides/overview.html",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    terms = {"bulk", "queueable", "nightly", "records"}

    assert app._is_relevant(item, terms, intent_data=intent) is True
    assert app._filter_relevant_evidence([item], terms, intent) == [item]


def test_architecture_retrieval_keeps_searching_until_architecture_center():
    intent = {
        "intent": "architecture",
        "question_type": "design",
        "requires_documentation": True,
        "requires_architecture_analysis": True,
        "research_depth": "deep",
        "topics": ["bulk", "ldv"],
        "salesforce_features": ["Bulk API"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    seen = {"searches": 0}
    original_build_search_queries = app.build_search_queries
    original_mcp_search = app.mcp_search
    original_mcp_fetch = app.mcp_fetch
    try:
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: [
            "Salesforce Bulk API 2.0 large data volume",
            "Salesforce Well-Architected Framework Trusted Easy Adaptable",
            "Salesforce Architecture Center integration patterns",
        ]

        def fake_search(query):
            seen["searches"] += 1
            if "Well-Architected" in query:
                return {
                    "query": query,
                    "excerpt": "Salesforce Well-Architected is organized to help you build solutions that are Trusted, Easy, and Adaptable.",
                    "source_type": "Salesforce Documentation",
                    "authority": "Salesforce Documentation",
                    "is_authoritative": True,
                    "title": "Overview",
                    "url": "https://architect.salesforce.com/docs/architect/well-architected/guide/overview.html",
                    "document_path": "well-architected/guides/overview.html",
                }
            return {
                "query": query,
                "excerpt": "Bulk API 2.0 provides a programmatic option to asynchronously insert large datasets in your Salesforce org.",
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
                "title": "Bulk API 2.0",
                "url": "https://developer.salesforce.com/docs/atlas.en-us.api_asynch.meta/api_asynch/bulk_api_2_0.htm",
                "document_path": "/docs/bulk",
            }

        app.mcp_search = fake_search
        app.mcp_fetch = lambda path: {}
        evidence = app.retrieve_evidence(
            "We process 500000 records every night. What architecture should we use?",
            intent,
            object(),
            temporal,
        )
        assert seen["searches"] >= 2
        assert any("architect.salesforce.com" in (item.get("url") or "") for item in evidence)
    finally:
        app.build_search_queries = original_build_search_queries
        app.mcp_search = original_mcp_search
        app.mcp_fetch = original_mcp_fetch


def test_google_search_marks_architecture_center_results_authoritative():
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
                            "title": "Well-Architected Overview",
                            "link": "https://architect.salesforce.com/docs/architect/well-architected/guide/overview.html",
                            "snippet": "Trusted, Easy, and Adaptable.",
                        }
                    ]
                }
            )

        app.requests.get = fake_get
        results = app.google_search("well-architected")

        assert len(results) == 1
        assert results[0]["is_authoritative"] is True
        assert "architect.salesforce.com" in results[0]["url"]
    finally:
        app.requests.get = original_get
        app.GOOGLE_CSE_API_KEY = original_api_key
        app.GOOGLE_CSE_CX = original_cx
        app.GOOGLE_FALLBACK_ENABLED = original_enabled


CASE_RISK_QUESTION = (
    "I would like to implement Risk detection based on case details and subject line. "
    "Risk types: legal notice, social media highlight, knows CXO or board of directors, "
    "complaint related to an employee. Visual indicator for customer support rep."
)


def _security_center_hit(query=""):
    return {
        "query": query,
        "excerpt": "Security Center Threat Detection alerts on tenant security events.",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": "Security Center Alert Use Cases",
        "url": "https://help.salesforce.com/s/articleView?id=xcloud.security_center_alerts_use_cases.htm",
        "document_path": "/docs/security_center_alerts_use_cases",
    }


def _case_classification_hit(query=""):
    return {
        "query": query,
        "excerpt": "Einstein Case Classification recommends a classification value from Case Subject and Description.",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": "Einstein Case Classification",
        "url": "https://help.salesforce.com/s/articleView?id=platform.lightning_page_components.htm",
        "document_path": "/docs/lightning_page_components",
    }


def _well_architected_hit(query=""):
    return {
        "query": query,
        "excerpt": "Salesforce Well-Architected is organized to help you build solutions that are Trusted, Easy, and Adaptable.",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
        "title": "Well-Architected Overview",
        "url": "https://architect.salesforce.com/docs/architect/well-architected/guide/overview.html",
        "document_path": "well-architected/guides/overview.html",
    }


def test_llm_research_intent_upgraded_for_case_risk():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "research",
                    "question_type": "explanation",
                    "topics": ["risk detection"],
                    "salesforce_features": ["Case"],
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
            ),
        }
    )
    intent = app.classify_intent(CASE_RISK_QUESTION, llm)
    assert intent["intent"] == "architecture"
    assert intent["requires_architecture_analysis"] is True
    assert intent["research_depth"] == "deep"


def test_troubleshooting_intent_not_upgraded_to_architecture():
    llm = FakeLLM(
        response_map={
            app.INTENT_CLASSIFIER_PROMPT: json.dumps(
                {
                    "intent": "troubleshooting",
                    "question_type": "diagnosis",
                    "topics": ["SOQL"],
                    "salesforce_features": ["Apex"],
                    "is_salesforce_specific": True,
                    "requires_documentation": True,
                    "requires_current_docs": False,
                    "temporal_validation_required": True,
                    "requested_release": None,
                    "requires_multiple_sources": False,
                    "requires_code_analysis": True,
                    "requires_architecture_analysis": False,
                    "research_depth": "standard",
                    "model_tier": "standard",
                }
            ),
        }
    )
    intent = app.classify_intent("Why am I getting Too many SOQL queries?", llm)
    assert intent["intent"] == "troubleshooting"


def test_case_risk_queries_prefer_capability_language():
    intent = app.classify_intent_heuristic(CASE_RISK_QUESTION)
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries(CASE_RISK_QUESTION, intent, None, temporal)
    blob = " ".join(queries).lower()
    assert "classification" in blob
    assert "record-triggered" in blob or "flow" in blob
    assert any("Well-Architected" in query for query in queries)
    assert CASE_RISK_QUESTION not in queries


def test_account_hold_queries_are_account_not_case_classification():
    question = "How do I show a visual indicator on Account when the customer is on credit hold?"
    intent = app.classify_intent_heuristic(question)
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    queries = app.build_search_queries(question, intent, None, temporal)
    blob = " ".join(queries).lower()
    assert intent["intent"] == "architecture"
    assert "account" in blob
    assert "einstein case classification" not in blob
    assert question not in queries


def test_security_center_is_wrong_family_for_crm_process():
    assert app._is_wrong_family_item(CASE_RISK_QUESTION, _security_center_hit()) is True
    assert app._is_wrong_family_item(CASE_RISK_QUESTION, _case_classification_hit()) is False
    gap = app._architecture_family_gap(
        CASE_RISK_QUESTION,
        {"intent": "architecture", "requires_architecture_analysis": True},
        [_security_center_hit()],
    )
    assert gap["needs_rescue"] is True
    assert gap["wrong_family"] is True
    dropped = app._drop_wrong_family_evidence(
        CASE_RISK_QUESTION,
        [_security_center_hit(), _case_classification_hit()],
    )
    assert dropped == [_case_classification_hit()]


def test_case_risk_retrieve_does_not_stop_on_security_center():
    intent = app.classify_intent_heuristic(CASE_RISK_QUESTION)
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    seen = {"queries": []}
    original_build = app.build_search_queries
    original_search = app.mcp_search
    original_fetch = app.mcp_fetch
    original_google_mode = app.GOOGLE_SEARCH_MODE
    try:
        app.GOOGLE_SEARCH_MODE = "off"
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: [
            "Case risk detection threat",
            "Salesforce Well-Architected Framework Trusted Easy Adaptable",
        ]

        def fake_search(query):
            seen["queries"].append(query)
            if "Case Classification" in query or "record-triggered" in query:
                return _case_classification_hit(query)
            if "Well-Architected" in query or "Architecture Center" in query:
                return _well_architected_hit(query)
            return _security_center_hit(query)

        app.mcp_search = fake_search
        app.mcp_fetch = lambda path: {}
        evidence = app.retrieve_evidence(CASE_RISK_QUESTION, intent, None, temporal)
        urls = [item.get("url") or "" for item in evidence]
        assert any("lightning_page_components" in url for url in urls)
        assert any("architect.salesforce.com" in url for url in urls)
        assert not any("security_center" in url for url in urls)
        assert any("Case Classification" in query or "record-triggered" in query for query in seen["queries"])
        assert len(seen["queries"]) > 1
    finally:
        app.build_search_queries = original_build
        app.mcp_search = original_search
        app.mcp_fetch = original_fetch
        app.GOOGLE_SEARCH_MODE = original_google_mode


def test_architecture_second_pass_fetches_product_and_center():
    intent = {
        "intent": "architecture",
        "question_type": "design",
        "requires_documentation": True,
        "requires_architecture_analysis": True,
        "research_depth": "deep",
        "topics": ["Case", "risk"],
        "salesforce_features": ["Case"],
    }
    temporal = {
        "temporal_validation_required": False,
        "current_docs_required": False,
        "historical_release_requested": False,
        "requested_release": None,
    }
    fetched = []
    original_search = app.mcp_search
    original_fetch = app.mcp_fetch
    original_build = app.build_search_queries
    original_google_mode = app.GOOGLE_SEARCH_MODE
    try:
        app.GOOGLE_SEARCH_MODE = "off"
        app.build_search_queries = lambda message, intent_data, llm, temporal_context=None: [
            "Einstein Case Classification Case Subject Description",
            "Salesforce Well-Architected Framework Trusted Easy Adaptable",
        ]

        def fake_search(query):
            if "Well-Architected" in query:
                return _well_architected_hit(query)
            return _case_classification_hit(query)

        def fake_fetch(path):
            fetched.append(path)
            return {
                "excerpt": f"fetched {path}",
                "title": path,
                "url": f"https://example.invalid{path}",
                "document_path": path,
                "source_type": "Salesforce Documentation",
                "authority": "Salesforce Documentation",
                "is_authoritative": True,
            }

        app.mcp_search = fake_search
        app.mcp_fetch = fake_fetch
        app.retrieve_evidence(CASE_RISK_QUESTION, intent, None, temporal)
        assert "well-architected/guides/overview.html" in fetched
        assert "/docs/lightning_page_components" in fetched
    finally:
        app.mcp_search = original_search
        app.mcp_fetch = original_fetch
        app.build_search_queries = original_build
        app.GOOGLE_SEARCH_MODE = original_google_mode
