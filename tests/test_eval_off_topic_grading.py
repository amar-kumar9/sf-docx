"""Deterministic checks for the off-topic grading mode in eval/agent_eval.py.

Verifies that the eval instrument REWARDS a correct refusal of an off-topic pack
and FAILS a fluent-but-wrong synthesis from off-topic docs — without needing a
live LLM provider. Skipped when pandas (an eval-only dependency) is unavailable.
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("pandas")

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("agent_eval", _ROOT / "eval" / "agent_eval.py")
agent_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(agent_eval)


OFF_TOPIC_SAMPLE = {
    "user_input": "How do I get the Live Agent chat queue position for a session?",
    "reference": "No on-topic official documentation was found.",
    "reference_contexts": [],
    "type": "off_topic",
}
EVIDENCE = [{
    "excerpt": "Configure assignment rules and routing configurations to distribute work items.",
    "title": "Omni-Channel Routing",
    "url": "https://help.salesforce.com/omni",
}]
REFUSAL = (
    "## No on-topic documentation found\n"
    "[Unverified] The retrieved Salesforce documentation does not appear to "
    "answer this question."
)
FLUENT = (
    "## Answer\nTo get the queue position, configure Omni-Channel assignment "
    "rules and routing configurations that distribute work to available agents."
)


def _trace(steps):
    return {"query": "q", "steps": [{"step": s, "outcome": "ok"} for s in steps]}


# The early-refusal trajectory (gate fires before synthesis).
REFUSAL_STEPS = ["guardrail", "classify", "temporal", "retrieve", "relevance_filter"]
# A full trajectory that (wrongly) synthesized an answer.
FLUENT_STEPS = REFUSAL_STEPS + ["conflict", "current_fact_gate", "generate", "grounding"]


def test_off_topic_refusal_passes():
    passed, failures, meta = agent_eval._grade_session(
        OFF_TOPIC_SAMPLE, REFUSAL, EVIDENCE, _trace(REFUSAL_STEPS), "blocked"
    )
    assert passed is True, failures
    assert meta["sample_type"] == "off_topic"


def test_off_topic_fluent_answer_fails():
    passed, failures, _ = agent_eval._grade_session(
        OFF_TOPIC_SAMPLE, FLUENT, EVIDENCE, _trace(FLUENT_STEPS), "passed"
    )
    assert passed is False
    assert "off_topic_not_refused" in failures


def test_off_topic_rubric_rewards_refusal_and_penalizes_fluent():
    _, failures, meta = agent_eval._grade_session(
        OFF_TOPIC_SAMPLE, REFUSAL, EVIDENCE, _trace(REFUSAL_STEPS), "blocked"
    )
    good = agent_eval._score_rubrics(
        OFF_TOPIC_SAMPLE,
        {"response": REFUSAL, "evidence": EVIDENCE, "trace": _trace(REFUSAL_STEPS), "grounding_status": "blocked"},
        failures,
        meta,
    )
    assert good["rubric_pass"] is True
    assert good["rubric_average"] == 1.0

    _, bad_failures, bad_meta = agent_eval._grade_session(
        OFF_TOPIC_SAMPLE, FLUENT, EVIDENCE, _trace(FLUENT_STEPS), "passed"
    )
    bad = agent_eval._score_rubrics(
        OFF_TOPIC_SAMPLE,
        {"response": FLUENT, "evidence": EVIDENCE, "trace": _trace(FLUENT_STEPS), "grounding_status": "passed"},
        bad_failures,
        bad_meta,
    )
    assert bad["rubric_pass"] is False
