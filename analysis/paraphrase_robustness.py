#!/usr/bin/env python3
"""Quantify within-frame prompt variance for the rephrasing robustness sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("detailed_results", [])
    return [row for row in rows if not row.get("error") and row.get("judge_scores")]


def family(row: dict[str, Any]) -> str:
    if row.get("frame_family"):
        return str(row["frame_family"])
    label = str(row.get("context_label") or "neutral")
    if label in {"none", "neutral", "neutral_none"}:
        return "neutral"
    return label.rsplit("_v", 1)[0]


def build_report(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        value = (row.get("judge_scores") or {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            grouped[
                (
                    str(row.get("model_name") or row.get("target_model")),
                    str(row.get("scenario_id")),
                    family(row),
                )
            ][str(row.get("prompt_variant_id") or row.get("context_label"))] = float(
                value
            )

    within: list[dict[str, Any]] = []
    variant_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, scenario, frame), variants in sorted(grouped.items()):
        for variant, value in variants.items():
            variant_values[(frame, variant)].append(value)
        if frame == "neutral" or len(variants) < 2:
            continue
        values = list(variants.values())
        within.append(
            {
                "model_name": model,
                "scenario_id": scenario,
                "frame_family": frame,
                "variant_count": len(values),
                "within_family_mean": mean(values),
                "within_family_sd": pstdev(values),
                "within_family_range": max(values) - min(values),
            }
        )

    variant_summary: list[dict[str, Any]] = [
        {
            "frame_family": frame,
            "prompt_variant_id": variant,
            "n": len(values),
            "mean": mean(values),
            "sd": pstdev(values) if len(values) > 1 else 0.0,
        }
        for (frame, variant), values in sorted(variant_values.items())
    ]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in within:
        by_family[str(row["frame_family"])].append(row)
    family_summary: list[dict[str, Any]] = []
    for frame, family_rows in sorted(by_family.items()):
        variant_means: list[float] = [
            float(item["mean"])
            for item in variant_summary
            if item["frame_family"] == frame
        ]
        family_summary.append(
            {
                "frame_family": frame,
                "paired_model_scenario_count": len(family_rows),
                "mean_within_family_sd": mean(
                    float(item["within_family_sd"]) for item in family_rows
                ),
                "mean_within_family_range": mean(
                    float(item["within_family_range"]) for item in family_rows
                ),
                "range_of_variant_means": max(variant_means) - min(variant_means),
            }
        )
    return {
        "schema_version": "1.0",
        "metric": metric,
        "interpretation": (
            "Small within-family spread supports robustness to wording; report it beside, "
            "not instead of, paired frame effects."
        ),
        "family_summary": family_summary,
        "variant_summary": variant_summary,
        "within_pair_diagnostics": within,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--metric", default="belief_amplification")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load_rows(args.input), args.metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
