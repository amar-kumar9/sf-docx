from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd

import app as app_module
from eval_utils import apply_pass_fail_labels
from app import (
    classify_intent,
    generate_answer,
    get_llm,
    normalize_evidence,
    resolve_evidence_conflicts,
    retrieve_evidence,
    validate_temporal_context,
)


DEFAULT_DATASET_PATH = Path("golden_dataset.json")
DEFAULT_OUTPUT_PATH = Path("ragas_baseline_results.csv")
DEFAULT_COMPARE_OUTPUT_PATH = Path("ragas_comparison_results.csv")
DEFAULT_EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", "gemma3:1b")
DEFAULT_EMBEDDINGS_MODEL = os.getenv("RAGAS_EMBEDDINGS_MODEL", "nomic-embed-text")


def _ensure_langchain_vertexai_compat() -> None:
    try:
        from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
        return
    except Exception:
        pass

    package_name = "langchain_community.chat_models"
    module_name = "langchain_community.chat_models.vertexai"

    package_module = sys.modules.get(package_name)
    if package_module is None:
        package_module = types.ModuleType(package_name)
        package_module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[package_name] = package_module

    shim = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - compatibility shim
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI compatibility shim is active because the installed "
                "langchain-community package no longer provides this module."
            )

    shim.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = shim


_ensure_langchain_vertexai_compat()


def _load_samples(dataset_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                payload = value
                break
    if not isinstance(payload, list):
        raise ValueError(
            "Dataset must be a JSON array of objects or a JSON object "
            "containing a 'samples', 'data', or 'items' array."
        )

    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(payload, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample #{index} is not an object.")

        query = sample.get("user_input") or sample.get("question")
        reference = sample.get("reference") or sample.get("expected_answer")
        if not query:
            raise ValueError(f"Sample #{index} is missing 'user_input'.")
        if not reference:
            raise ValueError(f"Sample #{index} is missing 'reference'.")

        sample = dict(sample)
        sample["user_input"] = str(query).strip()
        sample["reference"] = str(reference).strip()
        sample.setdefault("type", sample.get("question_type", "unknown"))
        sample.setdefault("reference_contexts", None)
        samples.append(sample)

    return samples


def _make_ragas_llm():
    provider = os.getenv("RAGAS_EVAL_PROVIDER", "ollama").lower()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=DEFAULT_EVAL_MODEL, temperature=0)
            logger.info(f"RagasEvalLLM=Ollama | Model={DEFAULT_EVAL_MODEL}")
        except Exception as exc:
            logger.warning(f"Local Ollama evaluator unavailable: {exc}")
            llm = None
    else:
        llm = None

    if llm is None:
        _, llm = get_llm("standard")
        if llm is None:
            raise RuntimeError("No evaluator LLM is available.")
        logger.warning("Falling back to app's standard-tier evaluator LLM.")

    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(llm)
    except Exception as exc:
        logger.warning(f"Ragas LLM wrapper unavailable, using raw LLM object: {exc}")
        return llm


def _make_ragas_embeddings():
    provider = os.getenv("RAGAS_EMBEDDINGS_PROVIDER", "ollama").lower()
    if provider == "none":
        return None

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings

            embeddings = OllamaEmbeddings(model=DEFAULT_EMBEDDINGS_MODEL)
            logger.info(f"RagasEvalEmbeddings=Ollama | Model={DEFAULT_EMBEDDINGS_MODEL}")
        except Exception as exc:
            logger.warning(f"Embeddings unavailable from Ollama: {exc}")
            return None
    else:
        logger.warning(
            f"Unsupported RAGAS_EMBEDDINGS_PROVIDER={provider!r}; skipping embeddings."
        )
        return None

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(embeddings)
    except Exception as exc:
        logger.warning(f"Ragas embeddings wrapper unavailable, using raw embeddings: {exc}")
        return embeddings


def _build_ragas_dataset(rows: list[dict[str, Any]]) -> Any:
    try:
        from datasets import Dataset

        return Dataset.from_list(rows)
    except ModuleNotFoundError:
        pass

    try:
        from ragas import EvaluationDataset, SingleTurnSample

        samples = []
        for row in rows:
            samples.append(
                SingleTurnSample(
                    user_input=row.get("user_input"),
                    retrieved_contexts=row.get("retrieved_contexts"),
                    response=row.get("response"),
                    reference=row.get("reference"),
                    reference_contexts=row.get("reference_contexts"),
                )
            )
        try:
            return EvaluationDataset(samples=samples)
        except Exception:
            return EvaluationDataset.from_list(rows)
    except Exception as exc:
        raise RuntimeError(
            "Ragas is not installed in this environment. Install the benchmark "
            "dependencies first, then rerun this script."
        ) from exc


def _load_metrics(evaluator_llm, evaluator_embeddings):
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

        metrics = [context_precision, context_recall, faithfulness]
        if evaluator_embeddings is not None:
            metrics.append(answer_relevancy)
        else:
            logger.warning("Skipping answer_relevancy because no embeddings model is configured.")
        return metrics
    except Exception:
        pass

    try:
        try:
            from ragas.metrics import AnswerRelevancy
        except Exception:
            from ragas.metrics import ResponseRelevancy as AnswerRelevancy

        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    except Exception as exc:
        raise RuntimeError(
            "Could not import Ragas metrics from either the legacy or current API."
        ) from exc

    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
    ]
    if evaluator_embeddings is not None:
        metrics.append(AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings))
    else:
        logger.warning("Skipping answer relevancy because no embeddings model is configured.")
    return metrics


def _run_pipeline_sample(sample: dict[str, Any]) -> dict[str, Any]:
    query = sample["user_input"]
    reference = sample["reference"]

    _, fast_llm = get_llm("fast")
    if fast_llm is None:
        raise RuntimeError("No fast-tier LLM is available.")

    intent_data = classify_intent(query, fast_llm)
    temporal_context = validate_temporal_context(query, intent_data, fast_llm)
    raw_evidence = retrieve_evidence(query, intent_data, fast_llm, temporal_context)
    evidence = normalize_evidence(raw_evidence)
    conflict_summary = resolve_evidence_conflicts(evidence, intent_data, fast_llm, query)
    tier = intent_data.get("model_tier", "standard")
    _, synthesis_llm = get_llm(tier)
    if synthesis_llm is None:
        raise RuntimeError("No synthesis LLM is available.")

    response = generate_answer(
        query,
        evidence,
        intent_data,
        conflict_summary,
        temporal_context,
        synthesis_llm,
    )

    contexts = [item["excerpt"] for item in evidence if item.get("excerpt")]
    row = {
        "user_input": query,
        "retrieved_contexts": contexts if contexts else ["No evidence retrieved."],
        "response": response,
        "reference": reference,
        "reference_contexts": sample.get("reference_contexts"),
        "sample_type": sample.get("type", "unknown"),
    }
    return row


def _evaluate_rows(rows: list[dict[str, Any]], output_path: Path) -> Any:
    try:
        dataset = _build_ragas_dataset(rows)
        evaluator_llm = _make_ragas_llm()
        evaluator_embeddings = _make_ragas_embeddings()
        metrics = _load_metrics(evaluator_llm, evaluator_embeddings)

        from ragas import evaluate
    except Exception as exc:
        fallback_df = pd.DataFrame(rows)
        fallback_df["ragas_available"] = False
        fallback_df["evaluation_status"] = "skipped_ragas_missing"
        fallback_df.to_csv(output_path, index=False)
        print(
            "Ragas is not installed in this environment, so the benchmark "
            "saved the raw pipeline outputs instead of scoring them."
        )
        print(
            "Install the benchmark dependency with: `pip install ragas` "
            "and rerun the command to compute evaluation metrics."
        )
        return None, fallback_df

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    df = apply_pass_fail_labels(results.to_pandas())
    df.to_csv(output_path, index=False)
    return results, df


def _build_rows(samples: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    original_flag = app_module.GOOGLE_FALLBACK_ENABLED
    app_module.GOOGLE_FALLBACK_ENABLED = variant == "google_fallback"
    try:
        rows = []
        for sample in samples:
            row = _run_pipeline_sample(sample)
            row["variant"] = variant
            rows.append(row)
        return rows
    finally:
        app_module.GOOGLE_FALLBACK_ENABLED = original_flag


def run_benchmark(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    output_path: str = str(DEFAULT_OUTPUT_PATH),
) -> None:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    samples = _load_samples(dataset_file)
    if not samples:
        raise ValueError("Dataset is empty.")

    output_file = Path(output_path)
    rows = _build_rows(samples, "baseline")
    results, df = _evaluate_rows(rows, output_file)

    print("Benchmark complete.")
    print(f"Results written to: {output_file}")
    score_cols = [
        col
        for col in df.columns
        if col
        not in {
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
            "reference_contexts",
            "sample_type",
            "variant",
            "failed_metrics",
            "overall_pass",
            "context_precision_pass",
            "context_recall_pass",
            "faithfulness_pass",
            "answer_relevancy_pass",
            "ragas_available",
            "evaluation_status",
        }
    ]
    if score_cols:
        print("Average scores:")
        for col in score_cols:
            try:
                print(f"  {col}: {float(df[col].mean()):.4f}")
            except Exception:
                continue
    if results is None:
        print("Ragas scoring was skipped because the package is not installed.")
        print("The CSV contains raw pipeline outputs plus an evaluation_status column.")
    else:
        print(results)


def run_comparison(
    dataset_path: str = str(DEFAULT_DATASET_PATH),
    output_path: str = str(DEFAULT_COMPARE_OUTPUT_PATH),
) -> None:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    samples = _load_samples(dataset_file)
    if not samples:
        raise ValueError("Dataset is empty.")

    baseline_rows = _build_rows(samples, "baseline")
    google_rows = _build_rows(samples, "google_fallback")

    baseline_results, baseline_df = _evaluate_rows(baseline_rows, Path("ragas_baseline_run.csv"))
    google_results, google_df = _evaluate_rows(google_rows, Path("ragas_google_run.csv"))

    combined = pd.concat([baseline_df, google_df], ignore_index=True)
    combined.to_csv(output_path, index=False)

    print("Comparison complete.")
    print(f"Combined results written to: {output_path}")
    print("Baseline averages:")
    for col in [
        c
        for c in baseline_df.columns
        if c
        not in {
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
            "reference_contexts",
            "sample_type",
            "variant",
            "failed_metrics",
            "overall_pass",
            "context_precision_pass",
            "context_recall_pass",
            "faithfulness_pass",
            "answer_relevancy_pass",
            "ragas_available",
            "evaluation_status",
        }
    ]:
        try:
            print(f"  {col}: {float(baseline_df[col].mean()):.4f}")
        except Exception:
            continue
    print("Google fallback averages:")
    for col in [
        c
        for c in google_df.columns
        if c
        not in {
            "user_input",
            "retrieved_contexts",
            "response",
            "reference",
            "reference_contexts",
            "sample_type",
            "variant",
            "failed_metrics",
            "overall_pass",
            "context_precision_pass",
            "context_recall_pass",
            "faithfulness_pass",
            "answer_relevancy_pass",
            "ragas_available",
            "evaluation_status",
        }
    ]:
        try:
            print(f"  {col}: {float(google_df[col].mean()):.4f}")
        except Exception:
            continue
    print("Baseline result object:")
    print(baseline_results)
    print("Google fallback result object:")
    print(google_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Ragas benchmark against the Salesforce QA pipeline.")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the JSON golden dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the CSV file where benchmark results will be written.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and Google-fallback variants and compare them side by side.",
    )
    args = parser.parse_args()
    if args.compare:
        run_comparison(args.dataset, args.output)
    else:
        run_benchmark(args.dataset, args.output)


if __name__ == "__main__":
    main()
