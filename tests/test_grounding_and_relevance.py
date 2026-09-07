"""Unit tests for the correctness fixes:

- fact-lookup bypass of capability translation (solution_architect.is_fact_lookup)
- the independent off-topic gate (_pack_is_off_topic and friends)
- fail-closed grounding (validate_grounding never emits an unverified draft as passed)
- named-subject preservation on the query planner
- adjacent-word dedup in planned queries ("Salesforce Salesforce" bug)

These reproduce the live drift ("Live Agent queue position" → assignment rules)
deterministically, without needing MCP or an LLM.
"""

import app
import solution_architect as sa


LIVE_AGENT_QUEUE = "How do I get the Live Agent chat queue position for a session?"

OFF_TOPIC_DOC = {
    "title": "Omni-Channel Routing",
    "excerpt": (
        "Configure assignment rules and routing configurations to distribute "
        "work items to agents based on capacity."
    ),
    "url": "https://help.salesforce.com/s/articleView?id=omnichannel.htm",
    "source_type": "Salesforce Documentation",
    "is_authoritative": True,
}
ON_TOPIC_DOC = {
    "title": "Live Agent clientChatQueuePosition",
    "excerpt": (
        "The clientChatQueuePosition property returns the position of the chat "
        "in the queue for a Live Agent session."
    ),
    "url": "https://developer.salesforce.com/docs/live-agent.htm",
    "source_type": "Salesforce Documentation",
    "is_authoritative": True,
}


# --------------------------------------------------------------------------- #
# Fact-lookup bypass (root-cause fix)
# --------------------------------------------------------------------------- #

def test_fact_lookup_is_not_capability_translated():
    assert sa.is_fact_lookup(LIVE_AGENT_QUEUE) is True
    # A named-subject data lookup must NOT be intaken as a solution/architecture
    # design (that was the drift that reached assignment/omni rules).
    assert sa.is_solution_question(LIVE_AGENT_QUEUE) is False
    assert sa.is_business_requirement(LIVE_AGENT_QUEUE) is False


def test_fact_lookup_does_not_swallow_design_or_ui_questions():
    # "show a visual indicator" is a UI design, not a data lookup.
    assert sa.is_fact_lookup(
        "How do I show a visual indicator on Account when on credit hold?"
    ) is False
    # "notify the owner" is a notification design, not a data lookup.
    assert sa.is_fact_lookup(
        "How do I notify the account owner when an opportunity is stuck?"
    ) is False
    # Troubleshooting stays troubleshooting.
    assert sa.is_fact_lookup("Why am I getting Too many SOQL queries?") is False


# --------------------------------------------------------------------------- #
# Independent off-topic gate
# --------------------------------------------------------------------------- #

def test_off_topic_pack_is_flagged():
    assert app._evidence_covers_question(LIVE_AGENT_QUEUE, [OFF_TOPIC_DOC]) is False
    assert app._pack_is_off_topic(LIVE_AGENT_QUEUE, {"intent": "quick_fact"}, [OFF_TOPIC_DOC]) is True


def test_on_topic_pack_is_not_flagged():
    assert app._evidence_covers_question(LIVE_AGENT_QUEUE, [ON_TOPIC_DOC]) is True
    assert app._pack_is_off_topic(LIVE_AGENT_QUEUE, {"intent": "quick_fact"}, [ON_TOPIC_DOC]) is False


def test_empty_pack_is_not_off_topic():
    # Empty packs are handled by the no-evidence paths, not this gate.
    assert app._pack_is_off_topic(LIVE_AGENT_QUEUE, {"intent": "quick_fact"}, []) is False


def test_architecture_questions_are_exempt_unless_fact_lookup():
    arch_intent = {"intent": "architecture", "requires_architecture_analysis": True}
    design_q = "What Salesforce architecture should we use for 500000 records nightly?"
    # Conceptual design docs legitimately have low lexical overlap — do not block.
    assert app._pack_is_off_topic(design_q, arch_intent, [OFF_TOPIC_DOC]) is False
    # But a fact lookup misrouted to architecture is still checked against the topic.
    assert app._pack_is_off_topic(LIVE_AGENT_QUEUE, arch_intent, [OFF_TOPIC_DOC]) is True


def test_off_topic_response_refuses_and_keeps_sources():
    response = app._build_off_topic_pack_response(LIVE_AGENT_QUEUE, [OFF_TOPIC_DOC])
    assert "[Unverified]" in response
    assert "No on-topic documentation found" in response
    # The retrieved (unmatched) source URL is preserved for the user to verify.
    assert OFF_TOPIC_DOC["url"] in response


# --------------------------------------------------------------------------- #
# Fail-closed grounding
# --------------------------------------------------------------------------- #

class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    def __init__(self, content=None, raise_exc=False):
        self._content = content
        self._raise = raise_exc

    def invoke(self, messages):
        if self._raise:
            raise RuntimeError("provider down")
        return _Resp(self._content)


_EVIDENCE = [{"excerpt": "Some documentation text.", "title": "Doc",
              "url": "https://help.salesforce.com/x"}]


def test_grounding_passes_only_on_explicit_pass():
    answer, status = app.validate_grounding(
        "## Answer\nA claim.", _EVIDENCE, _FakeLLM("GROUNDING: passed"), "q", {}
    )
    assert status == "passed"
    assert "Grounding unverified" not in answer


def test_grounding_fails_closed_on_unrecognized_output():
    answer, status = app.validate_grounding(
        "## Answer\nA claim.", _EVIDENCE, _FakeLLM("total gibberish"), "q", {}
    )
    assert status == "unverified"
    assert "Grounding unverified" in answer


def test_grounding_fails_closed_when_checker_unavailable():
    # LLM raises → resilient fallback returns the "GROUNDING: unverified" sentinel.
    answer, status = app.validate_grounding(
        "## Answer\nA claim.", _EVIDENCE, _FakeLLM(raise_exc=True), "q", {}
    )
    assert status == "unverified"
    assert "Grounding unverified" in answer


def test_grounding_fallback_sentinel_is_unverified():
    assert app._safe_fallback_content("grounding", "") == "GROUNDING: unverified"


def test_grounding_skipped_without_evidence():
    answer, status = app.validate_grounding("x", [], _FakeLLM("anything"), "q", {})
    assert status == "skipped (no evidence)"


def test_grounding_unverified_note_is_idempotent():
    once = app._append_grounding_unverified_note("answer body")
    twice = app._append_grounding_unverified_note(once)
    assert once == twice
    assert "Grounding unverified" in once


def test_definition_keeps_spine_but_is_still_flagged():
    # Definitions do not get the blunt unverified note, but unrecognized output
    # must not be reported as "passed" — it is flagged instead.
    _, status = app.validate_grounding(
        "## Answer\nA Platform Event is...",
        _EVIDENCE,
        _FakeLLM("total gibberish"),
        "What is a Platform Event?",
        {"intent": "quick_fact", "question_type": "explanation"},
    )
    assert status == "flagged"


# --------------------------------------------------------------------------- #
# Named-subject preservation + query dedup
# --------------------------------------------------------------------------- #

def test_named_subject_survives_query_planning():
    result = app._ensure_named_subject_queries(
        ["salesforce omni-channel routing", "salesforce assignment rules"],
        "How do I get `clientChatQueuePosition` for a Live Agent chat?",
        {"intent": "quick_fact"},
    )
    assert result[0] == "clientChatQueuePosition"


def test_named_subject_not_forced_for_architecture():
    # A design question keeps its design queries; the named subject is not injected.
    result = app._ensure_named_subject_queries(
        ["Bulk API 2.0 large data volume"],
        "Should we use `Database.executeBatch` or Bulk API for 500000 records nightly?",
        {"intent": "architecture", "requires_architecture_analysis": True},
    )
    assert result == ["Bulk API 2.0 large data volume"]


def test_collapse_repeated_words():
    assert (
        sa._collapse_repeated_words("Salesforce Salesforce assignment rules")
        == "Salesforce assignment rules"
    )


def test_generic_queries_have_no_doubled_salesforce():
    # With no concrete object, target defaults to "Salesforce"; the template must
    # not emit "Salesforce Salesforce assignment rules".
    queries = sa._queries_for("assignment", None)
    assert queries
    assert all("salesforce salesforce" not in q.lower() for q in queries)


# --------------------------------------------------------------------------- #
# build_evidence_pack wiring
# --------------------------------------------------------------------------- #

def test_build_evidence_pack_flags_off_topic(monkeypatch):
    monkeypatch.setattr(
        app, "classify_intent",
        lambda m, llm=None: {"intent": "quick_fact", "requires_documentation": True},
    )
    monkeypatch.setattr(
        app, "validate_temporal_context",
        lambda m, i, llm=None: {
            "temporal_validation_required": False,
            "current_docs_required": False,
            "historical_release_requested": False,
            "requested_release": None,
        },
    )
    monkeypatch.setattr(app, "translate_requirement", lambda m, i, llm=None: dict(sa.EMPTY_PLAN))
    monkeypatch.setattr(app, "retrieve_evidence", lambda *a, **k: [dict(OFF_TOPIC_DOC)])

    pack = app.build_evidence_pack(LIVE_AGENT_QUEUE, llm=None)
    assert pack["relevance"] == "off_topic"
    # The host must be told NOT to synthesize from the off-topic pack.
    assert "did NOT match" in pack["host_instructions"]
    assert "not synthesize" in pack["host_instructions"].lower()


def test_build_evidence_pack_on_topic_gets_synthesis_directive(monkeypatch):
    monkeypatch.setattr(
        app, "classify_intent",
        lambda m, llm=None: {"intent": "quick_fact", "requires_documentation": True},
    )
    monkeypatch.setattr(
        app, "validate_temporal_context",
        lambda m, i, llm=None: {
            "temporal_validation_required": False,
            "current_docs_required": False,
            "historical_release_requested": False,
            "requested_release": None,
        },
    )
    monkeypatch.setattr(app, "translate_requirement", lambda m, i, llm=None: dict(sa.EMPTY_PLAN))
    monkeypatch.setattr(app, "retrieve_evidence", lambda *a, **k: [dict(ON_TOPIC_DOC)])

    pack = app.build_evidence_pack(LIVE_AGENT_QUEUE, llm=None)
    assert pack["relevance"] == "on_topic"
    assert "Synthesize" in pack["host_instructions"]
