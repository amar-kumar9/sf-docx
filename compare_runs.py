from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from eval_utils import apply_pass_fail_labels


METRIC_COLUMNS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
IGNORE_COLUMNS = {
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
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}. Run the benchmark first to generate it."
        )
    return pd.read_csv(path)


def _group_means(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in METRIC_COLUMNS if col in df.columns]
    if not metric_cols:
        return pd.DataFrame()
    group_cols = [col for col in ("variant", "sample_type") if col in df.columns]
    if not group_cols:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False)[metric_cols].mean(numeric_only=True)


def print_regression_report(baseline_csv: str = "ragas_baseline.csv", fallback_csv: str = "ragas_fallback.csv") -> None:
    base = apply_pass_fail_labels(_load_csv(Path(baseline_csv)))
    fall = apply_pass_fail_labels(_load_csv(Path(fallback_csv)))

    if len(base) != len(fall):
        raise ValueError("Baseline and fallback CSVs must contain the same number of rows.")

    metrics = [metric for metric in METRIC_COLUMNS if metric in base.columns and metric in fall.columns]
    print("\n=================== REGRESSION DIFFERENTIAL REPORT ===================")
    for metric in metrics:
        delta = fall[metric] - base[metric]
        diff_df = pd.DataFrame(
            {
                "user_input": base["user_input"],
                "sample_type": base["sample_type"] if "sample_type" in base.columns else "default",
                "baseline": base[metric].round(3),
                "fallback": fall[metric].round(3),
                "delta": delta.round(3),
                "baseline_pass": base.get(f"{metric}_pass", pd.Series([False] * len(base))),
                "fallback_pass": fall.get(f"{metric}_pass", pd.Series([False] * len(fall))),
            }
        )

        regressions = diff_df[diff_df["delta"] < -0.05].sort_values(by="delta")
        improvements = diff_df[diff_df["delta"] > 0.05].sort_values(by="delta", ascending=False)

        print(f"\n--- METRIC: {metric} ---")
        print("Averages by variant and sample_type:")
        grouped = _group_means(
            pd.concat(
                [base.assign(variant="baseline"), fall.assign(variant="fallback")],
                ignore_index=True,
            )
        )
        if not grouped.empty:
            print(grouped.to_string())

        print(f"Regressions ({len(regressions)}):")
        print(regressions.to_string(index=False) if not regressions.empty else "  None")
        print(f"\nImprovements ({len(improvements)}):")
        print(improvements.to_string(index=False) if not improvements.empty else "  None")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Ragas evaluation CSVs.")
    parser.add_argument("--baseline", default="ragas_baseline.csv", help="Baseline CSV path.")
    parser.add_argument("--fallback", default="ragas_fallback.csv", help="Fallback CSV path.")
    args = parser.parse_args()
    print_regression_report(args.baseline, args.fallback)


if __name__ == "__main__":
    main()
