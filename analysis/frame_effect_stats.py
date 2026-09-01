#!/usr/bin/env python3
"""Prespecified paired frame-effect statistics for Conspire-Bench v2."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable


FRAME_DESIGNS = {
    "v2": ("brainstorming", "critical_review"),
    "v3": (
        "brainstorming",
        "critical_review",
        "sensemaking",
        "supportive_listening",
    ),
}
FRAMES = FRAME_DESIGNS["v2"]
STRUCTURES = (
    "single_turn_complete_logic",
    "multi_turn_progression",
    "complete_logic_then_resistance",
)
CANONICAL_VARIANTS = {
    "neutral": "neutral_none",
    "brainstorming": "brainstorming_v1",
    "critical_review": "critical_review_v1",
    "sensemaking": "sensemaking_v1",
    "supportive_listening": "supportive_listening_v1",
}
PairKey = tuple[str, str, str, int]


def exact_binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probabilities = [math.comb(n, i) / (2**n) for i in range(n + 1)]
    observed = probabilities[k]
    return min(1.0, sum(p for p in probabilities if p <= observed + 1e-15))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in their original order."""
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    size = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * value))
        adjusted[index] = running
    return adjusted


def bootstrap_cluster_ci(
    values_by_scenario: dict[str, list[float]], draws: int, seed: int
) -> tuple[float, float]:
    """Bootstrap scenario-level means so target models are not pseudo-replicates."""
    cluster_values = [mean(values) for values in values_by_scenario.values() if values]
    if not cluster_values:
        raise ValueError("No complete scenario clusters are available")
    rng = random.Random(seed)
    size = len(cluster_values)
    estimates = [
        mean(cluster_values[rng.randrange(size)] for _ in range(size))
        for _ in range(draws)
    ]
    estimates.sort()
    low = estimates[max(0, int(0.025 * draws))]
    high = estimates[min(draws - 1, int(0.975 * draws))]
    return low, high


def numeric_metric(name: str) -> Callable[[dict[str, Any]], float]:
    return lambda row: float(row["judge_scores"][name])


def harm(row: dict[str, Any]) -> float:
    return float(bool(row["judge_scores"]["harmful_elaboration"]))


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("detailed_results", [])
    return [row for row in rows if not row.get("error") and row.get("judge_scores")]


def frame_family(row: dict[str, Any]) -> str:
    if row.get("frame_family"):
        return str(row["frame_family"])
    label = str(row.get("context_label") or "none")
    if label in {"none", "neutral", "neutral_none"}:
        return "neutral"
    if label.startswith("brainstorming"):
        return "brainstorming"
    if label.startswith("critical_review"):
        return "critical_review"
    if label.startswith("sensemaking"):
        return "sensemaking"
    if label.startswith("supportive_listening"):
        return "supportive_listening"
    return label


def is_control(row: dict[str, Any]) -> bool:
    metadata = row.get("scenario_metadata") or {}
    return bool(row.get("is_control") or metadata.get("is_control"))


def build_pairs(
    rows: list[dict[str, Any]],
    *,
    canonical_only: bool = True,
    include_controls: bool = False,
    frames: tuple[str, ...] = FRAMES,
) -> dict[PairKey, dict[str, dict[str, Any]]]:
    pairs: dict[PairKey, dict[str, dict[str, Any]]] = defaultdict(dict)
    allowed = {CANONICAL_VARIANTS[frame] for frame in ("neutral", *frames)}
    for row in rows:
        if is_control(row) and not include_controls:
            continue
        frame = frame_family(row)
        if frame not in {"neutral", *frames}:
            continue
        variant = str(row.get("prompt_variant_id") or row.get("context_label") or "")
        if canonical_only and variant not in allowed:
            continue
        key: PairKey = (
            str(row.get("model_name") or row.get("target_model")),
            str(row.get("scenario_id")),
            str(row.get("generation_seed")),
            int(row.get("replicate_id", 0) or 0),
        )
        if frame in pairs[key]:
            raise ValueError(
                "Duplicate row for pairing unit and frame: "
                f"model={key[0]} scenario={key[1]} seed={key[2]} "
                f"replicate={key[3]} frame={frame}. Use canonical-only mode or "
                "analyze prompt variants separately."
            )
        pairs[key][frame] = row
    return dict(pairs)


def paired_differences(
    pairs: dict[PairKey, dict[str, dict[str, Any]]],
    frame: str,
    metric: Callable[[dict[str, Any]], float],
    structure: str | None = None,
) -> list[tuple[str, float]]:
    values: list[tuple[str, float]] = []
    for key, frames in pairs.items():
        if "neutral" not in frames or frame not in frames:
            continue
        if structure and frames["neutral"].get("scenario_type") != structure:
            continue
        try:
            difference = metric(frames[frame]) - metric(frames["neutral"])
        except (KeyError, TypeError, ValueError):
            continue
        values.append((key[1], difference))
    return values


def by_scenario(values: list[tuple[str, float]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for scenario_id, value in values:
        grouped[scenario_id].append(value)
    return dict(grouped)


def sign_test(values_by_scenario: dict[str, list[float]]) -> dict[str, Any]:
    values = [mean(group) for group in values_by_scenario.values()]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    return {
        "positive_scenarios": positive,
        "negative_scenarios": negative,
        "tie_scenarios": ties,
        "p_value": exact_binom_two_sided(
            min(positive, negative), positive + negative
        ),
    }


def mcnemar(
    pairs: dict[PairKey, dict[str, dict[str, Any]]],
    frame: str,
    structure: str | None = None,
) -> dict[str, Any]:
    neutral_to_harm = 0
    harm_to_neutral = 0
    for frames in pairs.values():
        if "neutral" not in frames or frame not in frames:
            continue
        if structure and frames["neutral"].get("scenario_type") != structure:
            continue
        try:
            neutral = bool(harm(frames["neutral"]))
            framed = bool(harm(frames[frame]))
        except KeyError:
            continue
        if not neutral and framed:
            neutral_to_harm += 1
        elif neutral and not framed:
            harm_to_neutral += 1
    return {
        "neutral_to_harm": neutral_to_harm,
        "harm_to_neutral": harm_to_neutral,
        "p_value_descriptive": exact_binom_two_sided(
            min(neutral_to_harm, harm_to_neutral),
            neutral_to_harm + harm_to_neutral,
        ),
        "note": "Dyad-level McNemar p-value is descriptive; the CI is scenario-clustered.",
    }


def summarize(
    values: list[tuple[str, float]], draws: int, seed: int
) -> dict[str, Any]:
    grouped = by_scenario(values)
    scenario_means = [mean(group) for group in grouped.values()]
    if not scenario_means:
        return {
            "matched_dyads": 0,
            "scenario_clusters": 0,
            "mean_paired_difference": None,
            "ci_95": [None, None],
        }
    low, high = bootstrap_cluster_ci(grouped, draws=draws, seed=seed)
    return {
        "matched_dyads": len(values),
        "scenario_clusters": len(grouped),
        "mean_paired_difference": mean(scenario_means),
        "ci_95": [low, high],
    }


def build_report(
    rows: list[dict[str, Any]],
    metric_name: str = "belief_amplification",
    *,
    draws: int = 10000,
    seed: int = 7,
    canonical_only: bool = True,
    include_controls: bool = False,
    frames: tuple[str, ...] = FRAMES,
) -> dict[str, Any]:
    pairs = build_pairs(
        rows,
        canonical_only=canonical_only,
        include_controls=include_controls,
        frames=frames,
    )
    score = numeric_metric(metric_name)
    contrasts = []
    raw_sign_p_values = []
    for index, frame in enumerate(frames):
        numeric_values = paired_differences(pairs, frame, score)
        harm_values = paired_differences(pairs, frame, harm)
        numeric_summary = summarize(numeric_values, draws, seed + index * 10)
        sign = sign_test(by_scenario(numeric_values))
        raw_sign_p_values.append(float(sign["p_value"]))
        contrasts.append({
            "frame": frame,
            "reference": "neutral",
            "metric": metric_name,
            "effect": numeric_summary,
            "scenario_level_sign_test": sign,
            "harmful_elaboration": {
                "effect": summarize(harm_values, draws, seed + index * 10 + 1),
                "mcnemar": mcnemar(pairs, frame),
            },
        })
    for contrast, adjusted in zip(contrasts, holm_adjust(raw_sign_p_values)):
        contrast["scenario_level_sign_test"]["holm_adjusted_p_value"] = adjusted

    by_structure = []
    for frame_index, frame in enumerate(frames):
        for structure_index, structure in enumerate(STRUCTURES):
            numeric_values = paired_differences(pairs, frame, score, structure)
            harm_values = paired_differences(pairs, frame, harm, structure)
            by_structure.append({
                "frame": frame,
                "reference": "neutral",
                "scenario_type": structure,
                "metric_effect": summarize(
                    numeric_values,
                    draws,
                    seed + 100 + frame_index * 10 + structure_index,
                ),
                "harmful_elaboration_effect": summarize(
                    harm_values,
                    draws,
                    seed + 200 + frame_index * 10 + structure_index,
                ),
                "mcnemar": mcnemar(pairs, frame, structure),
            })
    return {
        "schema_version": "2.0",
        "metric": metric_name,
        "frames": list(frames),
        "canonical_only": canonical_only,
        "include_controls": include_controls,
        "pairing_unit": [
            "target_model",
            "scenario_id",
            "generation_seed",
            "replicate_id",
        ],
        "estimand": (
            "Mean paired frame-minus-neutral difference with equal scenario weighting; "
            "95% CIs resample scenario clusters."
        ),
        "pairing_units_observed": len(pairs),
        "contrasts": contrasts,
        "by_conversation_structure": by_structure,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metric", default="belief_amplification")
    parser.add_argument(
        "--frame-design",
        choices=sorted(FRAME_DESIGNS),
        default="v2",
        help="Use v3 for the five-family main design; v2 preserves replication output.",
    )
    parser.add_argument(
        "--include-noncanonical",
        action="store_true",
        help="Permit noncanonical variants; duplicate frame rows are still rejected.",
    )
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(
        load_rows(args.results),
        args.metric,
        draws=args.draws,
        seed=args.seed,
        canonical_only=not args.include_noncanonical,
        include_controls=args.include_controls,
        frames=FRAME_DESIGNS[args.frame_design],
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
