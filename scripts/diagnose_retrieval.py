from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import build_search_queries, classify_intent, get_llm, retrieve_evidence, validate_temporal_context


QUESTIONS = [
    "What is Agentforce Coworker?",
    "what is best way to avoide soql 101 error",
    "What is the latest Salesforce API version?",
    "What is the Salesforce Apex heap size?",
    "I would like to classify the cases based on the subject, description and other classification fields on the case object.",
    "Can Flow replace an Apex trigger?",
]


def main() -> None:
    _, llm = get_llm("fast")
    if llm is None:
        raise RuntimeError("No fast-tier LLM is available.")

    for question in QUESTIONS:
        intent = classify_intent(question, llm)
        temporal = validate_temporal_context(question, intent, llm)
        evidence = retrieve_evidence(question, intent, llm, temporal)
        queries = build_search_queries(question, intent, llm, temporal)

        print("\n" + "=" * 96)
        print(f"Q: {question}")
        print(
            "intent={intent} depth={depth} current_docs={current_docs} temporal_validation={temporal_required}".format(
                intent=intent.get("intent"),
                depth=intent.get("research_depth"),
                current_docs=intent.get("requires_current_docs"),
                temporal_required=temporal.get("temporal_validation_required"),
            )
        )
        print(f"planned_queries={json.dumps(queries, ensure_ascii=False)}")
        print(f"evidence_count={len(evidence)}")
        for idx, item in enumerate(evidence, start=1):
            print(f"  {idx}. {item.get('title') or item.get('authority') or 'Untitled'}")
            print(f"     url={item.get('url') or ''}")
            print(f"     path={item.get('document_path') or ''}")
            print(f"     excerpt={str(item.get('excerpt') or '')[:180]}")


if __name__ == "__main__":
    main()
