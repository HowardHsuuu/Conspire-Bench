#!/usr/bin/env python3
"""Prespecified paired frame-effect statistics for Conspire-Bench V2 and V3."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from rubric_v2 import RUBRIC_DIMENSIONS

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
DEFAULT_ANALYSIS_PLAN = (
    Path(__file__).resolve().parents[1] / "configs" / "analysis_plan_v3.json"
)
PairKey = tuple[str, str, str, str, int]


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


def benjamini_hochberg_adjust(p_values: list[float]) -> list[float]:
    """Return Benjamini-Hochberg FDR-adjusted p-values in original order."""

    ordered = sorted(enumerate(p_values), key=lambda item: item[1], reverse=True)
    adjusted = [1.0] * len(p_values)
    running = 1.0
    size = len(p_values)
    for reverse_rank, (index, value) in enumerate(ordered, start=1):
        rank = size - reverse_rank + 1
        running = min(running, min(1.0, value * size / rank))
        adjusted[index] = running
    return adjusted


def bootstrap_cluster_ci(
    values_by_motif: dict[str, list[float]], draws: int, seed: int
) -> tuple[float, float]:
    """Bootstrap motif-level means so structures and models are not pseudo-replicates."""

    cluster_values = [mean(values) for values in values_by_motif.values() if values]
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
    value = row["judge_scores"]["harmful_elaboration"]
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError("harmful_elaboration must be boolean or a family-average rate")


def redirection_applicable(row: dict[str, Any]) -> float:
    if row.get("appropriate_redirection_applicability_rate") is not None:
        return float(row["appropriate_redirection_applicability_rate"])
    return float(
        (row.get("judge_scores") or {}).get("appropriate_redirection") is not None
    )


def _subtype_values(row: dict[str, Any]) -> list[str]:
    value = (row.get("judge_scores") or {}).get("harmful_elaboration_subtype")
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item) != "none"]
    return [] if str(value) == "none" else [str(value)]


def _update_subtype_counter(counter: Counter[str], row: dict[str, Any]) -> None:
    wording_counts = row.get("wording_subtype_counts")
    if isinstance(wording_counts, dict):
        counter.update(
            {
                str(subtype): int(count)
                for subtype, count in wording_counts.items()
                if isinstance(count, int)
            }
        )
        return
    counter.update(_subtype_values(row))


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data if isinstance(data, list) else data.get("detailed_results", [])
    return [row for row in rows if not row.get("error") and row.get("judge_scores")]


def load_overlap_cluster_map(
    analysis_plan_path: Path = DEFAULT_ANALYSIS_PLAN,
) -> dict[str, str]:
    plan = json.loads(analysis_plan_path.read_text(encoding="utf-8"))
    groups = (
        (plan.get("uncertainty_and_multiplicity") or {})
        .get("overlap_sensitivity", {})
        .get("cluster_groups", {})
    )
    cluster_map: dict[str, str] = {}
    for cluster_id, motif_ids in groups.items():
        for motif_id in motif_ids:
            if motif_id in cluster_map:
                raise ValueError(
                    f"Motif appears in multiple overlap clusters: {motif_id}"
                )
            cluster_map[str(motif_id)] = str(cluster_id)
    return cluster_map


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
    average_wording_variants: bool = False,
) -> dict[PairKey, dict[str, dict[str, Any]]]:
    grouped: dict[PairKey, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
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
            str(
                row.get("motif_id")
                or (row.get("scenario_metadata") or {}).get("motif_id")
                or row.get("scenario_id")
            ),
            str(row.get("scenario_type")),
            str(row.get("generation_seed")),
            int(row.get("replicate_id", 0) or 0),
        )
        grouped[key][frame].append(row)

    pairs: dict[PairKey, dict[str, dict[str, Any]]] = defaultdict(dict)
    for key, frame_rows in grouped.items():
        for frame, variant_rows in frame_rows.items():
            if len(variant_rows) > 1 and not average_wording_variants:
                raise ValueError(
                    "Duplicate row for pairing unit and frame: "
                    f"model={key[0]} motif={key[1]} structure={key[2]} "
                    f"seed={key[3]} replicate={key[4]} frame={frame}. "
                    "Use canonical-only mode or "
                    "enable family-level wording averaging."
                )
            if average_wording_variants and frame != "neutral":
                expected_ids = {f"{frame}_v{index}" for index in range(1, 5)}
                observed_ids = {
                    str(row.get("prompt_variant_id") or row.get("context_label") or "")
                    for row in variant_rows
                }
                if observed_ids != expected_ids:
                    continue
            pairs[key][frame] = _average_variant_rows(variant_rows)
    return dict(pairs)


def _average_variant_rows(
    variant_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Represent a frame family by equal-weighting its available wording rows."""

    if not variant_rows:
        raise ValueError("Cannot average an empty wording family")
    if len(variant_rows) == 1:
        return variant_rows[0]
    averaged = deepcopy(variant_rows[0])
    score_maps = [row.get("judge_scores") or {} for row in variant_rows]
    averaged_scores: dict[str, Any] = {}
    for metric in sorted({key for scores in score_maps for key in scores}):
        values = [scores.get(metric) for scores in score_maps]
        usable = [float(value) for value in values if isinstance(value, (int, float))]
        if usable:
            averaged_scores[metric] = mean(usable)
    averaged["judge_scores"] = averaged_scores
    averaged["prompt_variant_id"] = f"{frame_family(averaged)}_family_average"
    averaged["wording_variant_count"] = len(variant_rows)
    averaged["wording_variant_ids"] = sorted(
        str(row.get("prompt_variant_id") or row.get("context_label") or "")
        for row in variant_rows
    )
    subtype_counts: Counter[str] = Counter()
    for row in variant_rows:
        subtype_counts.update(_subtype_values(row))
    averaged["wording_subtype_counts"] = dict(sorted(subtype_counts.items()))
    averaged["appropriate_redirection_applicability_rate"] = mean(
        redirection_applicable(row) for row in variant_rows
    )
    return averaged


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


def by_motif(values: list[tuple[str, float]]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for scenario_id, value in values:
        grouped[scenario_id].append(value)
    return dict(grouped)


def sign_test(values_by_motif: dict[str, list[float]]) -> dict[str, Any]:
    values = [mean(group) for group in values_by_motif.values()]
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    ties = sum(value == 0 for value in values)
    return {
        "positive_motifs": positive,
        "negative_motifs": negative,
        "tie_motifs": ties,
        "p_value": exact_binom_two_sided(min(positive, negative), positive + negative),
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


def subtype_frequencies(
    pairs: dict[PairKey, dict[str, dict[str, Any]]], frame: str
) -> dict[str, Any]:
    neutral: Counter[str] = Counter()
    framed: Counter[str] = Counter()
    matched = 0
    for frame_rows in pairs.values():
        if "neutral" not in frame_rows or frame not in frame_rows:
            continue
        matched += 1
        _update_subtype_counter(neutral, frame_rows["neutral"])
        _update_subtype_counter(framed, frame_rows[frame])
    return {
        "matched_dyads": matched,
        "neutral_counts": dict(sorted(neutral.items())),
        "framed_counts": dict(sorted(framed.items())),
    }


def summarize(values: list[tuple[str, float]], draws: int, seed: int) -> dict[str, Any]:
    grouped = by_motif(values)
    motif_means = [mean(group) for group in grouped.values()]
    if not motif_means:
        return {
            "matched_dyads": 0,
            "motif_clusters": 0,
            "mean_paired_difference": None,
            "ci_95": [None, None],
        }
    low, high = bootstrap_cluster_ci(grouped, draws=draws, seed=seed)
    return {
        "matched_dyads": len(values),
        "motif_clusters": len(grouped),
        "mean_paired_difference": mean(motif_means),
        "ci_95": [low, high],
    }


def equal_weighted_model_summary(
    pairs: dict[PairKey, dict[str, dict[str, Any]]],
    frame: str,
    metric: Callable[[dict[str, Any]], float],
) -> dict[str, Any]:
    """Average each target model's motif-weighted effect with equal model weight."""

    model_effects: dict[str, float] = {}
    for target_model in sorted({key[0] for key in pairs}):
        model_pairs = {
            key: value for key, value in pairs.items() if key[0] == target_model
        }
        motif_groups = by_motif(paired_differences(model_pairs, frame, metric))
        motif_means = [mean(values) for values in motif_groups.values() if values]
        if motif_means:
            model_effects[target_model] = mean(motif_means)
    return {
        "target_model_count": len(model_effects),
        "mean_paired_difference": (
            mean(model_effects.values()) if model_effects else None
        ),
        "target_model_effects": model_effects,
    }


def build_overlap_sensitivity_report(
    rows: list[dict[str, Any]],
    metric_name: str,
    *,
    draws: int,
    seed: int,
    canonical_only: bool,
    include_controls: bool,
    cluster_map: dict[str, str],
) -> dict[str, Any]:
    pairs = build_pairs(
        rows,
        canonical_only=canonical_only,
        include_controls=include_controls,
        frames=FRAME_DESIGNS["v3"],
        average_wording_variants=not canonical_only,
    )
    score = numeric_metric(metric_name)
    contrasts: list[dict[str, Any]] = []
    for index, frame in enumerate(FRAME_DESIGNS["v3"]):
        values = [
            (cluster_map.get(motif_id, motif_id), difference)
            for motif_id, difference in paired_differences(pairs, frame, score)
        ]
        contrasts.append(
            {
                "frame": frame,
                "reference": "neutral",
                "metric": metric_name,
                "effect": summarize(values, draws, seed + index * 10),
                "motif_family_level_sign_test": sign_test(by_motif(values)),
            }
        )
    return {
        "metric": metric_name,
        "cluster_unit": "prespecified_overlap_motif_family",
        "cluster_map": cluster_map,
        "contrasts": contrasts,
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
    apply_within_outcome_holm: bool = True,
) -> dict[str, Any]:
    pairs = build_pairs(
        rows,
        canonical_only=canonical_only,
        include_controls=include_controls,
        frames=frames,
        average_wording_variants=not canonical_only,
    )
    score = numeric_metric(metric_name)
    contrasts: list[dict[str, Any]] = []
    raw_sign_p_values: list[float] = []
    for index, frame in enumerate(frames):
        numeric_values = paired_differences(pairs, frame, score)
        harm_values = paired_differences(pairs, frame, harm)
        numeric_summary = summarize(numeric_values, draws, seed + index * 10)
        sign = sign_test(by_motif(numeric_values))
        raw_sign_p_values.append(float(sign["p_value"]))
        contrasts.append(
            {
                "frame": frame,
                "reference": "neutral",
                "metric": metric_name,
                "effect": numeric_summary,
                "equal_weighted_target_model_summary": equal_weighted_model_summary(
                    pairs, frame, score
                ),
                "motif_level_sign_test": sign,
                "harmful_elaboration": {
                    "effect": summarize(harm_values, draws, seed + index * 10 + 1),
                    "mcnemar": mcnemar(pairs, frame),
                    "subtype_frequencies": subtype_frequencies(pairs, frame),
                },
                **(
                    {
                        "appropriate_redirection_applicability": {
                            "effect": summarize(
                                paired_differences(
                                    pairs, frame, redirection_applicable
                                ),
                                draws,
                                seed + index * 10 + 2,
                            ),
                            "interpretation": (
                                "Paired difference in the rate at which redirection was "
                                "applicable; the outcome effect above is conditional on "
                                "both rows having a non-null score."
                            ),
                        }
                    }
                    if metric_name == "appropriate_redirection"
                    else {}
                ),
            }
        )
    if apply_within_outcome_holm:
        adjusted_p_values = holm_adjust(raw_sign_p_values)
        if len(contrasts) != len(adjusted_p_values):
            raise RuntimeError("Contrast and adjusted p-value counts diverged")
        for contrast, adjusted in zip(contrasts, adjusted_p_values):  # noqa: B905
            contrast["motif_level_sign_test"]["holm_adjusted_p_value"] = adjusted

    by_structure = []
    for frame_index, frame in enumerate(frames):
        for structure_index, structure in enumerate(STRUCTURES):
            numeric_values = paired_differences(pairs, frame, score, structure)
            harm_values = paired_differences(pairs, frame, harm, structure)
            by_structure.append(
                {
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
                }
            )
    by_target_model: list[dict[str, Any]] = []
    target_models = sorted({key[0] for key in pairs})
    for model_index, target_model in enumerate(target_models):
        model_pairs = {
            key: value for key, value in pairs.items() if key[0] == target_model
        }
        for frame_index, frame in enumerate(frames):
            by_target_model.append(
                {
                    "target_model": target_model,
                    "frame": frame,
                    "reference": "neutral",
                    "metric": metric_name,
                    "effect": summarize(
                        paired_differences(model_pairs, frame, score),
                        draws,
                        seed + 500 + model_index * 100 + frame_index,
                    ),
                    "harmful_elaboration_effect": summarize(
                        paired_differences(model_pairs, frame, harm),
                        draws,
                        seed + 700 + model_index * 100 + frame_index,
                    ),
                }
            )
    return {
        "schema_version": "2.0",
        "metric": metric_name,
        "frames": list(frames),
        "canonical_only": canonical_only,
        "wording_aggregation": (
            "canonical_variant_only"
            if canonical_only
            else "equal_mean_of_four_variants_nested_within_frame_family"
        ),
        "include_controls": include_controls,
        "pairing_unit": [
            "target_model",
            "motif_id",
            "scenario_type",
            "generation_seed",
            "replicate_id",
        ],
        "estimand": (
            "Mean paired frame-minus-neutral difference with equal motif weighting; "
            "interaction structures and target models are averaged within motif and 95% CIs "
            "resample motif clusters."
        ),
        "pairing_units_observed": len(pairs),
        "contrasts": contrasts,
        "by_conversation_structure": by_structure,
        "by_target_model": by_target_model,
    }


def build_v3_coequal_report(
    rows: list[dict[str, Any]],
    *,
    draws: int = 10000,
    seed: int = 7,
    canonical_only: bool = True,
    include_controls: bool = False,
    overlap_cluster_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build all 28 V3 estimands and apply one prespecified BH-FDR family."""

    outcome_reports: list[dict[str, Any]] = []
    overlap_reports: list[dict[str, Any]] = []
    contrast_refs: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    effective_overlap_map = (
        overlap_cluster_map
        if overlap_cluster_map is not None
        else load_overlap_cluster_map()
    )
    for outcome_index, outcome in enumerate(RUBRIC_DIMENSIONS):
        report = build_report(
            rows,
            outcome,
            draws=draws,
            seed=seed + outcome_index * 1000,
            canonical_only=canonical_only,
            include_controls=include_controls,
            frames=FRAME_DESIGNS["v3"],
            apply_within_outcome_holm=False,
        )
        outcome_reports.append(report)
        overlap_reports.append(
            build_overlap_sensitivity_report(
                rows,
                outcome,
                draws=draws,
                seed=seed + outcome_index * 1000 + 500,
                canonical_only=canonical_only,
                include_controls=include_controls,
                cluster_map=effective_overlap_map,
            )
        )
        for contrast in report["contrasts"]:
            raw_p_values.append(float(contrast["motif_level_sign_test"]["p_value"]))
            contrast_refs.append(contrast)

    adjusted = benjamini_hochberg_adjust(raw_p_values)
    if len(contrast_refs) != 28 or len(adjusted) != 28:
        raise RuntimeError("V3 coequal analysis must contain exactly 28 estimands")
    for contrast, adjusted_p in zip(contrast_refs, adjusted):  # noqa: B905
        contrast["motif_level_sign_test"]["fdr_bh_adjusted_p_value"] = adjusted_p

    return {
        "schema_version": "3.0",
        "design_version": "v3",
        "outcome_status": "coequal",
        "outcomes": list(RUBRIC_DIMENSIONS),
        "frames": list(FRAME_DESIGNS["v3"]),
        "estimand_count": len(contrast_refs),
        "multiplicity": (
            "Benjamini-Hochberg FDR across all 28 frame-by-outcome sign tests"
        ),
        "canonical_only": canonical_only,
        "outcome_reports": outcome_reports,
        "overlap_cluster_sensitivity": {
            "status": "sensitivity_only",
            "policy": (
                "Repeat all 28 effect summaries after grouping the three audited "
                "high-overlap motifs with their documented overlap families."
            ),
            "outcome_reports": overlap_reports,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--metric",
        help=(
            "Analyze one outcome only. When omitted with --frame-design v3, all seven "
            "coequal outcomes and the 28-test BH-FDR family are produced."
        ),
    )
    parser.add_argument(
        "--frame-design",
        choices=sorted(FRAME_DESIGNS),
        default="v2",
        help="Use v3 for the five-family main design; v2 preserves replication output.",
    )
    parser.add_argument(
        "--include-noncanonical",
        action="store_true",
        help=(
            "Use the full V3 wording sweep by equal-averaging the four variants nested "
            "within each non-neutral frame family."
        ),
    )
    parser.add_argument("--include-controls", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.results)
    if args.frame_design == "v3" and args.metric is None:
        report = build_v3_coequal_report(
            rows,
            draws=args.draws,
            seed=args.seed,
            canonical_only=not args.include_noncanonical,
            include_controls=args.include_controls,
        )
    else:
        report = build_report(
            rows,
            args.metric or "belief_amplification",
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
