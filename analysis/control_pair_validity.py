#!/usr/bin/env python3
"""Evaluate rubric discriminant validity on matched conspiratorial/control scenarios."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

DEFAULT_METRICS = (
    "belief_amplification",
    "epistemic_grounding",
    "uncertainty_calibration",
    "supportive_engagement",
    "response_appropriateness",
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("detailed_results"), list):
        return payload["detailed_results"]
    raise ValueError("Expected a result list or an object with detailed_results")


def _pair_identity(row: dict[str, Any]) -> tuple[Any, ...] | None:
    meta = row.get("scenario_metadata") or {}
    pair_id = meta.get("control_pair_id") or row.get("control_pair_id")
    if not pair_id:
        return None
    return (
        pair_id,
        row.get("model_name"),
        row.get("prompt_variant_id") or row.get("context_label"),
        row.get("replicate_id", 0),
        row.get("generation_seed"),
    )


def matched_pairs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("error"):
            continue
        identity = _pair_identity(row)
        if identity is None:
            continue
        meta = row.get("scenario_metadata") or {}
        is_control = bool(meta.get("is_control", row.get("is_control", False)))
        buckets[identity][is_control] = row

    pairs = []
    for identity, sides in sorted(buckets.items(), key=lambda item: str(item[0])):
        if True not in sides or False not in sides:
            continue
        pairs.append(
            {
                "pair_id": identity[0],
                "model_name": identity[1],
                "prompt_variant_id": identity[2],
                "replicate_id": identity[3],
                "generation_seed": identity[4],
                "conspiracy": sides[False],
                "control": sides[True],
            }
        )
    return pairs


def _bootstrap_ci(
    values: list[float], seed: int, draws: int = 2000
) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return [means[int(0.025 * draws)], means[min(draws - 1, int(0.975 * draws))]]


def _metric_summary(values: list[float], seed: int) -> dict[str, Any]:
    greater = sum(value > 0 for value in values)
    equal = sum(value == 0 for value in values)
    lower = sum(value < 0 for value in values)
    return {
        "pair_count": len(values),
        "mean_conspiracy_minus_control": statistics.fmean(values) if values else None,
        "median_conspiracy_minus_control": statistics.median(values)
        if values
        else None,
        "bootstrap_95_ci": _bootstrap_ci(values, seed),
        "conspiracy_higher_rate": greater / len(values) if values else None,
        "tie_rate": equal / len(values) if values else None,
        "conspiracy_lower_rate": lower / len(values) if values else None,
    }


def build_report(
    rows: Iterable[dict[str, Any]],
    metrics: Iterable[str] = DEFAULT_METRICS,
    seed: int = 24,
) -> dict[str, Any]:
    rows = list(rows)
    pairs = matched_pairs(rows)
    metric_differences: dict[str, list[float]] = defaultdict(list)
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_frame: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pair_rows = []

    for pair in pairs:
        conspiracy_scores = pair["conspiracy"].get("judge_scores") or {}
        control_scores = pair["control"].get("judge_scores") or {}
        differences = {}
        for metric in metrics:
            left = conspiracy_scores.get(metric)
            right = control_scores.get(metric)
            if (
                isinstance(left, (int, float))
                and not isinstance(left, bool)
                and isinstance(right, (int, float))
                and not isinstance(right, bool)
            ):
                difference = float(left) - float(right)
                differences[metric] = difference
                metric_differences[metric].append(difference)
                by_model[str(pair["model_name"])][metric].append(difference)
                by_frame[str(pair["prompt_variant_id"])][metric].append(difference)
        pair_rows.append(
            {
                key: pair[key]
                for key in (
                    "pair_id",
                    "model_name",
                    "prompt_variant_id",
                    "replicate_id",
                    "generation_seed",
                )
            }
            | {
                "conspiracy_scenario_id": pair["conspiracy"].get("scenario_id"),
                "control_scenario_id": pair["control"].get("scenario_id"),
                "differences": differences,
            }
        )

    possible_pair_buckets = {
        identity
        for row in rows
        if not row.get("error")
        for identity in [_pair_identity(row)]
        if identity is not None
    }
    return {
        "schema_version": "1.0",
        "interpretation": (
            "Positive differences mean the conspiratorial item received a higher raw score. "
            "For belief_amplification and response_appropriateness this is the expected discriminant direction; "
            "other dimensions must be interpreted according to their scale direction."
        ),
        "matched_pair_count": len(pairs),
        "possible_pair_bucket_count": len(possible_pair_buckets),
        "pair_coverage": len(pairs) / len(possible_pair_buckets)
        if possible_pair_buckets
        else None,
        "metrics": {
            metric: _metric_summary(metric_differences.get(metric, []), seed + index)
            for index, metric in enumerate(metrics)
        },
        "by_model": {
            group: {
                metric: _metric_summary(values, seed + index)
                for index, (metric, values) in enumerate(sorted(metric_map.items()))
            }
            for group, metric_map in sorted(by_model.items())
        },
        "by_prompt_variant": {
            group: {
                metric: _metric_summary(values, seed + index)
                for index, (metric, values) in enumerate(sorted(metric_map.items()))
            }
            for group, metric_map in sorted(by_frame.items())
        },
        "pairs": pair_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(load_rows(args.results), metrics=args.metrics, seed=args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
