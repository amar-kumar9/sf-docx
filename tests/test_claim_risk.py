"""Rules-based claim volatility classifier."""

import claim_risk
import app
import solution_architect as sa


def _classified(text: str) -> dict:
    return claim_risk.classify_claim({"text": text, "source": "test"})


def test_apex_heap_numeric_is_volatile():
    row = _classified("Apex synchronous heap limit = 6 MB")
    assert row["salesforce_controlled"] is True
    assert row["volatility"] == "potentially_mutable"
    assert row["requires_current_release"] is True
    assert row["claim_type"] == "numeric_constraint"


def test_soql_limit_is_volatile():
    row = _classified("Total number of SOQL queries issued | 100")
    assert row["requires_current_release"] is True


def test_api_version_is_volatile():
    row = _classified("Salesforce API version = 65")
    assert row["requires_current_release"] is True
    assert row["claim_type"] == "version"


def test_edition_availability_is_volatile():
    row = _classified("Feature X is available in Enterprise Edition")
    assert row["salesforce_controlled"] is True
    assert row["requires_current_release"] is True
    assert row["claim_type"] == "availability"


def test_http_404_is_stable():
    row = _classified("HTTP status 404 means Not Found")
    assert row["salesforce_controlled"] is False
    assert row["volatility"] == "stable"
    assert row["requires_current_release"] is False


def test_list_ordered_is_stable():
    row = _classified("Apex List is ordered")
    assert row["volatility"] == "stable"
    assert row["requires_current_release"] is False


def test_account_standard_object_is_stable():
    row = _classified("Account is a standard object")
    assert row["volatility"] == "stable"
    assert row["requires_current_release"] is False


def test_question_heuristic_detects_mutable_shape_without_heap_word():
    assert claim_risk.question_suggests_mutable_platform_fact(
        "What is the maximum SOQL query count per Apex transaction?"
    )
    assert claim_risk.question_suggests_mutable_platform_fact(
        "Whats the latest apex heap size limit ?"
    )


def test_analyze_evidence_marks_incomplete_without_release_notes():
    evidence = [
        {
            "title": "Execution Governors and Limits",
            "url": "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_gov_limits.htm",
            "document_path": "apexcode/apex_gov_limits.html",
            "excerpt": (
                "Per-Transaction Apex Limits\n"
                "| Total heap size | 6 MB | 12 MB |\n"
                "Email services heap size is 50 MB."
            ),
        }
    ]
    analysis = claim_risk.analyze_evidence_claims(
        "Whats the latest apex heap size limit ?",
        evidence,
        is_release_notes=app._is_release_notes_evidence,
    )
    assert analysis["requires_current_release"] is True
    assert analysis["has_release_notes_evidence"] is False
    assert any(c.get("requires_current_release") for c in analysis["claims"])
    verification = claim_risk.verification_from_claim_risk(analysis)
    assert verification["status"] == "incomplete"
    assert verification["missing"] == "current_release_notes"
    host = claim_risk.host_instructions_for_claim_risk(analysis, verification)
    assert "do not present those values as definitively current" in host.lower()
    assert "release notes" in host.lower()


def test_analyze_evidence_ok_when_release_notes_present():
    evidence = [
        {
            "title": "Apex Heap Size Limit — Winter ’27 Release Notes",
            "url": "https://help.salesforce.com/s/articleView?id=release-notes.rn_apex_heap_limit.htm&type=5",
            "document_path": "release-notes/rn_apex_heap_limit.html",
            "excerpt": "Synchronous Apex heap size increased to 10 MB.",
        }
    ]
    analysis = claim_risk.analyze_evidence_claims(
        "What is the Apex heap size?",
        evidence,
        is_release_notes=app._is_release_notes_evidence,
    )
    assert analysis["has_release_notes_evidence"] is True
    verification = claim_risk.verification_from_claim_risk(analysis)
    assert verification["status"] == "ok"


def test_build_evidence_pack_emits_claim_risk(monkeypatch):
    digest = {
        "title": "Execution Governors and Limits",
        "url": "https://developer.salesforce.com/docs/atlas.en-us.apexcode.meta/apexcode/apex_gov_limits.htm",
        "document_path": "apexcode/apex_gov_limits.html",
        "excerpt": "Total heap size^4 | 6 MB | 12 MB",
        "source_type": "Salesforce Documentation",
        "authority": "Salesforce Documentation",
        "is_authoritative": True,
    }
    monkeypatch.setattr(
        app,
        "classify_intent",
        lambda message, llm: {
            "intent": "limits",
            "question_type": "current_fact",
            "topics": ["Apex heap"],
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
        },
    )
    monkeypatch.setattr(
        app,
        "validate_temporal_context",
        lambda message, intent, llm: {
            "temporal_validation_required": True,
            "current_docs_required": True,
            "historical_release_requested": False,
            "requested_release": None,
            "reason": "test",
        },
    )
    monkeypatch.setattr(app, "translate_requirement", lambda m, i, llm=None: dict(sa.EMPTY_PLAN))
    monkeypatch.setattr(app, "retrieve_evidence", lambda *a, **k: [dict(digest)])
    monkeypatch.setattr(app, "_pack_is_off_topic", lambda *a, **k: False)

    pack = app.build_evidence_pack("Whats the latest apex heap size limit ?", llm=None)
    assert pack["verification"]["status"] == "incomplete"
    assert pack["claim_risk"]["requires_current_release"] is True
    assert pack["claims"]
    assert "claim-risk" in pack["host_instructions"].lower()
    assert "definitively current" in pack["host_instructions"].lower()


def test_volatile_limit_question_uses_claim_shape():
    assert app._is_volatile_limit_question("What is the maximum DML statements per transaction?")
    assert app._is_volatile_limit_question("What is the Apex heap size?")
