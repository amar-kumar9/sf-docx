from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "current_fact": {
        "context_precision": 0.75,
        "context_recall": 0.85,
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
    },
    "historical_release": {
        "context_precision": 0.75,
        "context_recall": 0.85,
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
    },
    "explanation": {
        "context_precision": 0.70,
        "context_recall": 0.70,
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
    },
    "design": {
        "context_precision": 0.70,
        "context_recall": 0.70,
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
    },
    "default": {
        "context_precision": 0.70,
        "context_recall": 0.75,
        "faithfulness": 0.90,
        "answer_relevancy": 0.80,
    },
}


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "context_precision": ("context_precision", "ContextPrecision", "contextprecision"),
    "context_recall": ("context_recall", "ContextRecall", "contextrecall"),
    "faithfulness": ("faithfulness", "Faithfulness"),
    "answer_relevancy": ("answer_relevancy", "answer_relevance", "response_relevancy", "ResponseRelevancy", "AnswerRelevancy"),
}


def _pick_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    for candidate in METRIC_ALIASES.get(metric, (metric,)):
        if candidate in df.columns:
            return candidate
    return None


def _thresholds_for_type(sample_type: Any) -> dict[str, float]:
    key = str(sample_type or "default").strip().lower()
    return DEFAULT_THRESHOLDS.get(key, DEFAULT_THRESHOLDS["default"])


def apply_pass_fail_labels(df: pd.DataFrame) -> pd.DataFrame:
    labeled = df.copy()
    if labeled.empty:
        labeled["overall_pass"] = []
        labeled["failed_metrics"] = []
        return labeled

    if "sample_type" not in labeled.columns:
        labeled["sample_type"] = "default"

    failed_metrics: list[list[str]] = []
    overall_pass: list[bool] = []

    for _, row in labeled.iterrows():
        thresholds = _thresholds_for_type(row.get("sample_type"))
        row_failures: list[str] = []
        for metric, cutoff in thresholds.items():
            column = _pick_metric_column(labeled, metric)
            if column is None:
                continue
            value = row.get(column)
            if pd.isna(value) or float(value) < cutoff:
                row_failures.append(metric)
        failed_metrics.append(row_failures)
        overall_pass.append(len(row_failures) == 0)

    labeled["failed_metrics"] = failed_metrics
    labeled["overall_pass"] = overall_pass

    for metric in DEFAULT_THRESHOLDS["default"].keys():
        column = _pick_metric_column(labeled, metric)
        if column is None:
            continue
        pass_col = f"{metric}_pass"
        labeled[pass_col] = False
        for idx, row in labeled.iterrows():
            thresholds = _thresholds_for_type(row.get("sample_type"))
            cutoff = thresholds.get(metric)
            if cutoff is None:
                continue
            value = row.get(column)
            labeled.at[idx, pass_col] = bool(not pd.isna(value) and float(value) >= cutoff)

    return labeled

