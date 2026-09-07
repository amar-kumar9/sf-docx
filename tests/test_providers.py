import json
import os

import app
import providers


def test_classify_intent_heuristic_definition_and_architecture():
    definition = app.classify_intent_heuristic("What is a Platform Event?")
    architecture = app.classify_intent_heuristic(
        "We process 500000 records every night. What Salesforce architecture should we use?"
    )
    limits = app.classify_intent_heuristic("What is the Salesforce Apex heap size?")
    case_risk = app.classify_intent_heuristic(
        "I would like to implement Risk detection based on case details and subject line. "
        "Risk types: legal notice, social media highlight, knows CXO or board of directors, "
        "complaint related to an employee. Visual indicator for customer support rep."
    )

    assert definition["intent"] == "quick_fact"
    assert definition["question_type"] == "explanation"
    assert architecture["intent"] == "architecture"
    assert architecture["requires_architecture_analysis"] is True
    assert limits["requires_current_docs"] is True
    assert case_risk["intent"] == "architecture"
    assert case_risk["requires_architecture_analysis"] is True
    assert case_risk["research_depth"] == "deep"
    how_to = app.classify_intent_heuristic(
        "How do I show a visual indicator on Account when the customer is on credit hold?"
    )
    assert how_to["intent"] == "architecture"
    classification_def = app.classify_intent_heuristic("What is Case Classification?")
    assert classification_def["intent"] == "quick_fact"
    soql = app.classify_intent_heuristic("Why am I getting Too many SOQL queries?")
    assert soql["intent"] == "troubleshooting"
    picker = app.classify_intent_heuristic(
        "lightning-record-picker does not trigger search when pasting the same search term "
        "after clearing selection. Steps to reproduce. Is there a way to fix this?"
    )
    assert picker["intent"] == "troubleshooting"
    assert picker["requires_architecture_analysis"] is False
    apex_bug = app.classify_intent_heuristic(
        "Database.executeBatch never calls the start method. Steps to reproduce. Is there a way to fix this?"
    )
    assert apex_bug["intent"] == "troubleshooting"


def test_build_search_queries_without_llm_keeps_architecture_boosters():
    intent = {
        "intent": "architecture",
        "requires_architecture_analysis": True,
        "research_depth": "deep",
        "topics": ["LDV"],
        "salesforce_features": ["Bulk API"],
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
        None,
        temporal,
    )
    assert any("Well-Architected" in query for query in queries)


def test_run_agent_retrieval_only_returns_evidence(monkeypatch):
    monkeypatch.setattr(app, "get_llm", lambda tier="standard": (None, None))
    monkeypatch.setattr(
        app,
        "mcp_search",
        lambda query: {
            "title": "Platform Events",
            "url": "https://developer.salesforce.com/docs/platform-events",
            "excerpt": "Platform Events let you publish and subscribe to business events.",
            "source_type": "Salesforce Documentation",
            "authority": "Salesforce Documentation",
            "is_authoritative": True,
            "document_path": "/docs/pe",
        },
    )
    monkeypatch.setattr(app, "mcp_fetch", lambda path: {})

    answer = app.run_agent("What is a Platform Event?", [])

    assert "[retrieval-only]" in answer
    assert "Platform Events" in answer
    assert "No LLM provider available" not in answer


def test_chat_completions_llm_maps_langchain_roles(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(providers.requests, "post", fake_post)
    llm = providers.ChatCompletionsLLM("https://api.openai.com/v1", "sk-test", "gpt-4o-mini")
    result = llm.invoke(
        [
            type("SystemMessage", (), {"type": "system", "content": "sys"})(),
            type("HumanMessage", (), {"type": "human", "content": "hi"})(),
        ]
    )
    assert result.content == "ok"
    assert captured["json"]["messages"][0]["role"] == "system"
    assert captured["json"]["messages"][1]["role"] == "user"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_save_provider_settings_merges_env(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("MCP_URL=https://example.invalid\nGROQ_API_KEY=old\n", encoding="utf-8")
    monkeypatch.setattr(providers, "ENV_PATH", env_path)
    providers.save_provider_settings({"GROQ_API_KEY": "new", "LLM_PROVIDER": "groq"})
    text = env_path.read_text(encoding="utf-8")
    assert "MCP_URL=https://example.invalid" in text
    assert "GROQ_API_KEY=new" in text
    assert "LLM_PROVIDER=groq" in text


def test_get_llm_respects_explicit_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_http_ok", lambda url, timeout=1.5: False)
    label, llm = providers.get_llm("fast")
    assert label.startswith("openai/")
    assert isinstance(llm, providers.ChatCompletionsLLM)
