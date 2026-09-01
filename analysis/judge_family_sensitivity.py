#!/usr/bin/env python3
"""Compare same-family judge scores with leave-same-family-out scores."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("detailed_results", [])
    if not isinstance(rows, list):
        raise ValueError("Expected a result list or detailed_results")
    return rows


def row_comparisons(row: dict[str, Any]) -> list[dict[str, Any]]:
    same_values: dict[str, list[float]] = defaultdict(list)
    other_values: dict[str, list[float]] = defaultdict(list)
    for judge in row.get("judge_results") or []:
        if judge.get("error") or not judge.get("scores"):
            continue
        bucket = same_values if judge.get("same_family_as_target", False) else other_values
        for metric, value in judge["scores"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                bucket[metric].append(float(value))
    output = []
    for metric in sorted(set(same_values) & set(other_values)):
        same_mean = mean(same_values[metric])
        other_mean = mean(other_values[metric])
        output.append({
            "response_id": row.get("response_id"),
            "scenario_id": row.get("scenario_id"),
            "target_model": row.get("model_name") or row.get("target_model"),
            "metric": metric,
            "same_family_mean": same_mean,
            "nonoverlap_mean": other_mean,
            "same_minus_nonoverlap": same_mean - other_mean,
            "same_family_judge_count": len(same_values[metric]),
            "nonoverlap_judge_count": len(other_values[metric]),
        })
    return output


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons = [comparison for row in rows for comparison in row_comparisons(row)]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for comparison in comparisons:
        grouped[(comparison["target_model"], comparison["metric"])].append(
            comparison["same_minus_nonoverlap"]
        )
    summary = [
        {
            "target_model": target_model,
            "metric": metric,
            "paired_response_count": len(deltas),
            "mean_same_minus_nonoverlap": mean(deltas),
            "mean_absolute_difference": mean(abs(delta) for delta in deltas),
        }
        for (target_model, metric), deltas in sorted(grouped.items())
    ]
    return {
        "schema_version": "1.0",
        "interpretation": (
            "Positive deltas mean the same-family judge assigned a higher raw score. "
            "Interpret direction using each metric's scale; no causal bias claim is implied."
        ),
        "comparison_count": len(comparisons),
        "summary": summary,
        "paired_comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
