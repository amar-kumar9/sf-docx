import solution_architect as sa


CASE_RISK = (
    "I would like to implement Risk detection based on case details and subject line. "
    "Risk types: legal notice, social media highlight, knows CXO or board of directors, "
    "complaint related to an employee. Visual indicator for customer support rep."
)
ACCOUNT_HOLD = (
    "How do I show a visual indicator on Account when the customer is on credit hold?"
)
OPPORTUNITY_NOTIFY = (
    "How do I notify the account owner when a high-value opportunity is stuck?"
)
SHIELD = "How do I enable Salesforce Shield Event Monitoring?"
RECORD_PICKER_BUG = (
    "lightning-record-picker does not trigger search when pasting the same search term "
    "after clearing selection. Steps to reproduce: enter a search term, select a record, "
    "click X to clear, paste the same term. Is there a way to fix this?"
)


def test_definition_is_not_a_solution_question():
    assert sa.is_solution_question("What is a Platform Event?") is False
    assert sa.is_business_requirement("What is a Platform Event?") is False


def test_how_to_and_implement_are_solution_questions():
    assert sa.is_solution_question(CASE_RISK) is True
    assert sa.is_business_requirement(CASE_RISK) is True
    assert sa.is_solution_question(ACCOUNT_HOLD) is True
    assert sa.is_solution_question(OPPORTUNITY_NOTIFY) is True
    assert sa.is_troubleshooting_question("Why am I getting Too many SOQL queries?") is True
    assert sa.is_solution_question("Why am I getting Too many SOQL queries?") is False
    assert sa.is_troubleshooting_question(RECORD_PICKER_BUG) is True
    assert sa.is_solution_question(RECORD_PICKER_BUG) is False
    assert sa.is_business_requirement(RECORD_PICKER_BUG) is False
    apex_bug = (
        "Database.executeBatch never calls the start method. "
        "Steps to reproduce. Is there a way to fix this?"
    )
    assert sa.is_troubleshooting_question(apex_bug) is True
    assert sa.is_solution_question(apex_bug) is False


def test_case_risk_translates_to_classification_automation_ui():
    plan = sa.heuristic_plan(CASE_RISK)
    assert plan["skip_raw"] is True
    assert "Case" in plan["objects"]
    assert "text_classification" in plan["capabilities"]
    assert "record_ui" in plan["capabilities"]
    assert "record_automation" in plan["capabilities"]
    assert plan["avoid_platform_security"] is True
    blob = " ".join(plan["search_queries"]).lower()
    assert "classification" in blob
    assert "flow" in blob or "record-triggered" in blob
    assert CASE_RISK not in plan["search_queries"]


def test_account_credit_hold_uses_same_layers_not_case_products():
    plan = sa.heuristic_plan(ACCOUNT_HOLD)
    assert "Account" in plan["objects"]
    assert "record_ui" in plan["capabilities"]
    blob = " ".join(plan["search_queries"]).lower()
    assert "account" in blob
    assert "lightning" in blob or "highlight" in blob
    assert "einstein case classification" not in blob
    assert plan["avoid_platform_security"] is True


def test_opportunity_notify_is_notification_not_shield():
    plan = sa.heuristic_plan(OPPORTUNITY_NOTIFY)
    assert "Opportunity" in plan["objects"]
    assert "notification" in plan["capabilities"]
    blob = " ".join(plan["search_queries"]).lower()
    assert "notification" in blob or "email alert" in blob
    assert "security center" not in blob


def test_shield_question_does_not_avoid_platform_security():
    plan = sa.heuristic_plan(SHIELD)
    assert "platform_security" in plan["capabilities"]
    assert plan["avoid_platform_security"] is False


def test_security_center_dropped_for_crm_process_not_for_shield_setup():
    shield_hit = {
        "title": "Security Center Alert Use Cases",
        "url": "https://help.salesforce.com/s/articleView?id=xcloud.security_center_alerts_use_cases.htm",
        "excerpt": "Security Center Threat Detection alerts on tenant security events.",
        "document_path": "/docs/security_center",
    }
    lightning_hit = {
        "title": "Lightning Record Page",
        "url": "https://help.salesforce.com/s/articleView?id=platform.lightning_page_components.htm",
        "excerpt": "Add a highlight panel to a Lightning record page.",
        "document_path": "/docs/lightning_page_components",
    }
    case_plan = sa.heuristic_plan(CASE_RISK)
    assert sa.is_avoided_evidence(case_plan, shield_hit) is True
    pack = [shield_hit, lightning_hit]
    sa.drop_avoided_evidence(case_plan, pack)
    assert pack == [lightning_hit]

    shield_plan = sa.heuristic_plan(SHIELD)
    assert sa.is_avoided_evidence(shield_plan, shield_hit) is False


def test_capability_design_uses_full_protocol_not_raw_wording():
    for question in (CASE_RISK, ACCOUNT_HOLD):
        plan = sa.heuristic_plan(question)
        assert plan["full_protocol"] is True
        assert plan["business_capability"]
        assert plan["business_capability"] != question
        assert "Einstein" not in plan["business_capability"]
        assert "Agentforce" not in plan["business_capability"]
        assert plan["discovery_questions"]
        assert plan["in_scope"]
        assert plan["out_of_scope"]
        blob = " ".join(plan["discovery_questions"]).lower()
        assert "false" in blob or "miss" in blob or "alarm" in blob


def test_case_risk_separates_detection_from_policy():
    plan = sa.heuristic_plan(CASE_RISK)
    assert plan["separate_detection_from_policy"] is True
    assert "text_classification" in plan["capabilities"]


def test_account_hold_is_capability_design_without_case_classifier():
    plan = sa.heuristic_plan(ACCOUNT_HOLD)
    assert plan["separate_detection_from_policy"] is False
    assert "einstein case classification" not in " ".join(plan["search_queries"]).lower()


def test_narrow_field_how_to_skips_full_protocol():
    plan = sa.heuristic_plan("How do I create a custom field on Account?")
    assert plan["active"] is True
    assert plan["full_protocol"] is False
    assert plan["discovery_questions"] == []


def test_merge_llm_plan_keeps_capability_and_discovery():
    heuristic = sa.heuristic_plan(ACCOUNT_HOLD)
    merged = sa.merge_llm_plan(
        heuristic,
        {
            "business_capability": "Make a credit-hold Account impossible for the rep to miss.",
            "discovery_questions": ["Is credit hold a field or a related outcome?"],
            "in_scope": ["Show the condition on Account"],
            "out_of_scope": ["Collections playbook"],
            "capabilities": ["record_ui"],
        },
    )
    assert merged["business_capability"].startswith("Make a credit-hold")
    assert merged["full_protocol"] is True
    assert "Is credit hold a field or a related outcome?" in merged["discovery_questions"]
    assert merged["in_scope"] == ["Show the condition on Account"]
    assert merged["out_of_scope"] == ["Collections playbook"]
