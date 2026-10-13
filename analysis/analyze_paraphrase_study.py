#!/usr/bin/env python3
"""Formal analysis for the frozen Conspire-Bench paraphrase robustness study.

The analysis follows ``configs/analysis_plan_v3.json``: non-neutral wording
variants are nested within their frame family, the four variants are equally
weighted before the family-minus-neutral contrast, and motifs are the bootstrap
and sign-test clusters.  Target models, judges, structures, and wording variants
are treated as fixed design factors rather than independent observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_full_matrix import (  # noqa: E402
    FAMILIES,
    Inference,
    aggregate,
    agreement,
    family_affinity,
    finite_mean,
)
from analysis.frame_effect_stats import (  # noqa: E402
    benjamini_hochberg_adjust,
    build_v3_coequal_report,
    exact_binom_two_sided,
)
from rubric_v2 import RUBRIC_DIMENSIONS  # noqa: E402

METRICS = list(RUBRIC_DIMENSIONS)
FRAMES = [
    "neutral",
    "brainstorming",
    "critical_review",
    "sensemaking",
    "supportive_listening",
]
STRUCTURES = [
    "single_turn_complete_logic",
    "multi_turn_progression",
    "complete_logic_then_resistance",
]
VARIANTS = [
    "neutral_none",
    *[f"{frame}_v{index}" for frame in FRAMES[1:] for index in range(1, 5)],
]
VARIANTS_BY_FRAME = {
    frame: [f"{frame}_v{index}" for index in range(1, 5)] for frame in FRAMES[1:]
}
HARM = METRICS.index("harmful_elaboration")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return math.nan


def _motif(row: dict[str, Any]) -> str:
    return str(
        row.get("motif_id")
        or (row.get("scenario_metadata") or {}).get("motif_id")
        or row.get("scenario_id")
    )


def _model_name(role: dict[str, Any]) -> str:
    return f"{role['provider']}/{role['model']}"


def build_score_arrays(
    rows: list[dict[str, Any]], config: dict[str, Any], design: dict[str, Any]
) -> dict[str, Any]:
    """Validate the full Cartesian product and reconstruct judge aggregates."""

    models = [_model_name(role) for role in config["models"]]
    model_families = [str(role["model_family"]) for role in config["models"]]
    motifs = list(design["selection"]["motif_ids"])
    expected = design["expected_counts"]
    shape = (
        len(motifs),
        len(models),
        len(STRUCTURES),
        len(VARIANTS),
        len(FAMILIES),
        len(METRICS),
    )
    scores = np.full(shape, np.nan)
    stored = np.full(shape[:-2] + (len(METRICS),), np.nan)
    occupied = np.zeros(shape[:-2], dtype=bool)
    response_ids: set[str] = set()
    errors: list[str] = []

    for row in rows:
        response_id = str(row.get("response_id") or "")
        if not response_id or response_id in response_ids:
            errors.append(f"missing or duplicate response_id: {response_id}")
            continue
        response_ids.add(response_id)
        if row.get("error") or row.get("evaluation_error"):
            errors.append(f"{response_id}: row error")
        if not row.get("generation_complete") or not row.get("evaluation_complete"):
            errors.append(f"{response_id}: incomplete row")
        try:
            index = (
                motifs.index(_motif(row)),
                models.index(str(row["model_name"])),
                STRUCTURES.index(str(row["scenario_type"])),
                VARIANTS.index(str(row["prompt_variant_id"])),
            )
        except (KeyError, ValueError) as error:
            errors.append(f"{response_id}: unexpected design cell ({error})")
            continue
        if occupied[index]:
            errors.append(f"{response_id}: duplicate design cell")
            continue
        occupied[index] = True
        stored[index] = [
            _score((row.get("judge_scores") or {}).get(metric)) for metric in METRICS
        ]
        judge_results = row.get("judge_results") or []
        if len(judge_results) != len(FAMILIES):
            errors.append(f"{response_id}: expected four judge results")
            continue
        observed_judges: set[str] = set()
        for result in judge_results:
            judge = str(result.get("judge_model_family"))
            if judge not in FAMILIES or judge in observed_judges:
                errors.append(f"{response_id}: invalid or duplicate judge {judge}")
                continue
            observed_judges.add(judge)
            if result.get("error") or not result.get("scores"):
                errors.append(f"{response_id}: failed judge {judge}")
                continue
            scores[index + (FAMILIES.index(judge),)] = [
                _score(result["scores"].get(metric)) for metric in METRICS
            ]

    if len(rows) != int(expected["responses"]):
        errors.append(f"expected {expected['responses']} rows, found {len(rows)}")
    if not occupied.all():
        errors.append(f"missing {int((~occupied).sum())} design cells")
    if errors:
        raise ValueError("\n".join(errors[:50]))

    primary = aggregate(scores, model_families)
    if not np.allclose(primary, stored, equal_nan=True):
        mismatch = int((~np.isclose(primary, stored, equal_nan=True)).sum())
        raise ValueError(
            f"reconstructed leave-same-family-out aggregate has {mismatch} mismatches"
        )
    return {
        "scores": scores,
        "primary": primary,
        "models": models,
        "model_families": model_families,
        "motifs": motifs,
    }


def summarize_scalar(differences: np.ndarray, inference: Inference) -> dict[str, Any]:
    """Equal-weight motif estimate and motif-cluster bootstrap CI."""

    values = np.asarray(differences, dtype=float).reshape(inference.clusters, -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cluster_means = np.nanmean(values, axis=1)
    present = np.isfinite(cluster_means)
    observed = cluster_means[present]
    denominator = inference.weights @ present.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bootstrap = (inference.weights @ np.nan_to_num(cluster_means)) / denominator
    bootstrap = bootstrap[np.isfinite(bootstrap)]
    positive = int((observed > 1e-12).sum())
    negative = int((observed < -1e-12).sum())
    return {
        "matched_observations": int(np.isfinite(values).sum()),
        "motif_clusters": len(observed),
        "mean_difference": float(observed.mean()) if len(observed) else None,
        "ci95": (
            np.quantile(bootstrap, [0.025, 0.975]).tolist()
            if len(bootstrap)
            else [None, None]
        ),
        "positive_motifs": positive,
        "negative_motifs": negative,
        "tie_motifs": len(observed) - positive - negative,
        "p_motif_sign": exact_binom_two_sided(
            min(positive, negative), positive + negative
        ),
    }


def strict_flip_summary(deltas: np.ndarray) -> dict[str, Any]:
    """Summarize sign instability across the four wording-specific contrasts."""

    values = np.asarray(deltas, dtype=float)
    if values.shape[-1] != 4:
        raise ValueError("wording delta axis must contain exactly four variants")
    valid = np.isfinite(values).all(axis=-1)
    positive = (values > 1e-12).any(axis=-1)
    negative = (values < -1e-12).any(axis=-1)
    flip = valid & positive & negative
    denominator = int(valid.sum())
    return {
        "complete_cells": denominator,
        "strict_polarity_flip_count": int(flip.sum()),
        "strict_polarity_flip_rate": (
            float(flip.sum() / denominator) if denominator else None
        ),
    }


def _breakdown(
    deltas: np.ndarray,
    *,
    models: list[str],
    motifs: list[str],
) -> dict[str, list[dict[str, Any]]]:
    return {
        "by_target_model": [
            {"target_model": model, **strict_flip_summary(deltas[:, index])}
            for index, model in enumerate(models)
        ],
        "by_structure": [
            {"scenario_type": structure, **strict_flip_summary(deltas[:, :, index])}
            for index, structure in enumerate(STRUCTURES)
        ],
        "by_motif": [
            {"motif_id": motif, **strict_flip_summary(deltas[index])}
            for index, motif in enumerate(motifs)
        ],
    }


def _canonical_family_flip(deltas: np.ndarray) -> dict[str, Any]:
    family_mean = finite_mean(deltas, axis=-1)
    canonical = deltas[..., 0]
    eligible = (
        np.isfinite(family_mean)
        & np.isfinite(canonical)
        & (np.abs(family_mean) > 1e-12)
        & (np.abs(canonical) > 1e-12)
    )
    flip = eligible & (family_mean * canonical < 0)
    count = int(eligible.sum())
    return {
        "nonzero_comparable_cells": count,
        "strict_sign_flip_count": int(flip.sum()),
        "strict_sign_flip_rate": float(flip.sum() / count) if count else None,
    }


def _variance_ratio(primary: np.ndarray, metric_index: int) -> dict[str, Any]:
    within_variances = []
    family_means = [primary[..., 0, metric_index]]
    for frame in FRAMES[1:]:
        indices = [VARIANTS.index(variant) for variant in VARIANTS_BY_FRAME[frame]]
        values = primary[..., indices, metric_index]
        count = np.isfinite(values).sum(axis=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            variance = np.nanvar(values, axis=-1)
        variance[count < 2] = np.nan
        within_variances.append(variance)
        family_means.append(finite_mean(values, axis=-1))
    within = np.stack(within_variances, axis=-1)
    between_values = np.stack(family_means, axis=-1)
    between_count = np.isfinite(between_values).sum(axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        between = np.nanvar(between_values, axis=-1)
    between[between_count < 2] = np.nan
    mean_within = float(np.nanmean(within)) if np.isfinite(within).any() else None
    mean_between = float(np.nanmean(between)) if np.isfinite(between).any() else None
    return {
        "within_wording_family_cell_variances": int(np.isfinite(within).sum()),
        "between_frame_cell_variances": int(np.isfinite(between).sum()),
        "mean_within_wording_variance": mean_within,
        "mean_between_frame_variance": mean_between,
        "within_to_between_variance_ratio": (
            mean_within / mean_between
            if mean_within is not None and mean_between not in {None, 0.0}
            else None
        ),
        "definition": (
            "Mean population variance across four wordings within each non-neutral "
            "family divided by mean population variance across neutral plus the four "
            "wording-averaged frame families, evaluated at matched motif-model-structure cells."
        ),
    }


def paraphrase_diagnostics(
    primary: np.ndarray,
    inference: Inference,
    *,
    models: list[str],
    motifs: list[str],
) -> dict[str, Any]:
    metric_reports = []
    wording_records: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRICS):
        family_reports = []
        neutral = primary[..., 0, metric_index]
        for frame in FRAMES[1:]:
            variants = VARIANTS_BY_FRAME[frame]
            indices = [VARIANTS.index(variant) for variant in variants]
            values = primary[..., indices, metric_index]
            deltas = values - neutral[..., None]
            finite_count = np.isfinite(values).sum(axis=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                within_sd = np.nanstd(values, axis=-1)
                within_range = np.nanmax(values, axis=-1) - np.nanmin(values, axis=-1)
            within_sd[finite_count < 2] = np.nan
            within_range[finite_count < 2] = np.nan
            variant_effects = []
            for variant_index, variant in enumerate(variants):
                record = {
                    "metric": metric,
                    "frame": frame,
                    "prompt_variant_id": variant,
                    **summarize_scalar(deltas[..., variant_index], inference),
                }
                variant_effects.append(record)
                wording_records.append(record)
            family_mean_effect = summarize_scalar(
                finite_mean(values, axis=-1) - neutral, inference
            )
            population_signs = [
                int(np.sign(record["mean_difference"])) for record in variant_effects
            ]
            family_sign = int(np.sign(family_mean_effect["mean_difference"]))
            family_reports.append(
                {
                    "frame": frame,
                    "eligible_within_family_cells": int(np.isfinite(within_sd).sum()),
                    "mean_within_family_sd": (
                        float(np.nanmean(within_sd))
                        if np.isfinite(within_sd).any()
                        else None
                    ),
                    "mean_within_family_range": (
                        float(np.nanmean(within_range))
                        if np.isfinite(within_range).any()
                        else None
                    ),
                    "wording_specific_effects": variant_effects,
                    "wording_averaged_family_effect": family_mean_effect,
                    "population_effect_signs": population_signs,
                    "all_wording_population_effects_match_family_direction": bool(
                        family_sign
                        and all(sign == family_sign for sign in population_signs)
                    ),
                    "cell_level_strict_polarity_flip": {
                        **strict_flip_summary(deltas),
                        **_breakdown(deltas, models=models, motifs=motifs),
                    },
                    "canonical_v1_vs_family_average": _canonical_family_flip(deltas),
                }
            )
        metric_reports.append(
            {
                "metric": metric,
                "variance_decomposition": _variance_ratio(primary, metric_index),
                "families": family_reports,
            }
        )

    adjusted = benjamini_hochberg_adjust(
        [float(record["p_motif_sign"]) for record in wording_records]
    )
    for record, q_value in zip(wording_records, adjusted, strict=True):
        record["q_bh_across_112_wording_tests"] = q_value
    return {
        "schema_version": "1.0",
        "strict_flip_definition": (
            "At a matched motif-target-structure cell, at least one wording-specific "
            "frame-minus-neutral contrast is positive and at least one is negative; "
            "ties alone do not count as polarity flips."
        ),
        "canonical_flip_definition": (
            "The canonical v1 contrast and the equal-weight four-wording family contrast "
            "are both nonzero and have opposite signs."
        ),
        "multiplicity": (
            "BH-FDR across the 112 exploratory wording-specific metric contrasts; the "
            "prespecified 28 family-level tests retain their separate primary FDR family."
        ),
        "metrics": metric_reports,
    }


def family_effects_from_array(
    values: np.ndarray, inference: Inference
) -> list[dict[str, Any]]:
    records = []
    neutral = values[..., 0, :]
    for frame in FRAMES[1:]:
        indices = [VARIANTS.index(variant) for variant in VARIANTS_BY_FRAME[frame]]
        family_mean = finite_mean(values[..., indices, :], axis=-2)
        for record in inference.summarize(family_mean - neutral):
            record.update(frame=frame, reference="neutral")
            records.append(record)
    adjusted = benjamini_hochberg_adjust(
        [float(record["p_motif_sign"]) for record in records]
    )
    for record, q_value in zip(records, adjusted, strict=True):
        record["q_bh"] = q_value
    return records


def judge_quality(scores: np.ndarray) -> list[dict[str, Any]]:
    output = []
    for judge_index, judge in enumerate(FAMILIES):
        judge_scores = scores[..., judge_index, :]
        distributions = {}
        for metric_index, metric in enumerate(METRICS):
            values = judge_scores[..., metric_index]
            finite = values[np.isfinite(values)]
            distributions[metric] = {
                "rated": len(finite),
                "missing_or_not_applicable": int(values.size - len(finite)),
                "counts": dict(
                    sorted(Counter(str(float(value)) for value in finite).items())
                ),
            }
        grounding = judge_scores[..., METRICS.index("epistemic_grounding")]
        calibration = judge_scores[..., METRICS.index("uncertainty_calibration")]
        comparable = np.isfinite(grounding) & np.isfinite(calibration)
        output.append(
            {
                "judge": judge,
                "rows": int(np.isfinite(judge_scores[..., 0]).sum()),
                "metric_distributions": distributions,
                "grounding_calibration_exact_match_rate": (
                    float((grounding[comparable] == calibration[comparable]).mean())
                    if comparable.any()
                    else None
                ),
            }
        )
    return output


def judge_sensitivity(
    scores: np.ndarray,
    primary: np.ndarray,
    model_families: list[str],
    inference: Inference,
) -> dict[str, Any]:
    primary_effects = family_effects_from_array(primary, inference)
    primary_lookup = {
        (record["frame"], record["metric"]): record for record in primary_effects
    }
    panels: dict[str, list[dict[str, Any]]] = {
        f"judge_only_{judge}": family_effects_from_array(
            scores[..., index, :], inference
        )
        for index, judge in enumerate(FAMILIES)
    }
    panels["all_four_judges_mean"] = family_effects_from_array(
        aggregate(scores, model_families, exclude_self=False, harm_rule="mean"),
        inference,
    )
    for judge in FAMILIES:
        panels[f"leave_out_{judge}"] = family_effects_from_array(
            aggregate(
                scores,
                model_families,
                panel=set(FAMILIES) - {judge},
                harm_rule="mean",
            ),
            inference,
        )
    summaries = []
    for panel, records in panels.items():
        comparisons = []
        for record in records:
            reference = primary_lookup[(record["frame"], record["metric"])]
            estimate = record["mean_difference"]
            reference_estimate = reference["mean_difference"]
            comparable = estimate is not None and reference_estimate is not None
            comparisons.append(
                {
                    "frame": record["frame"],
                    "metric": record["metric"],
                    "effect": estimate,
                    "primary_effect": reference_estimate,
                    "absolute_effect_difference": (
                        abs(float(estimate) - float(reference_estimate))
                        if comparable
                        else None
                    ),
                    "same_direction_as_primary": (
                        bool(
                            np.sign(float(estimate))
                            == np.sign(float(reference_estimate))
                        )
                        if comparable
                        else None
                    ),
                }
            )
        comparable = [
            item
            for item in comparisons
            if item["same_direction_as_primary"] is not None
        ]
        summaries.append(
            {
                "panel": panel,
                "same_direction_count": sum(
                    item["same_direction_as_primary"] for item in comparable
                ),
                "estimand_count": len(comparisons),
                "comparable_estimand_count": len(comparable),
                "mean_absolute_effect_difference": float(
                    np.mean([item["absolute_effect_difference"] for item in comparable])
                )
                if comparable
                else None,
                "comparisons": comparisons,
                "effects": records,
            }
        )
    affinity = family_affinity(scores, model_families, inference)
    for record in affinity["adjusted"]:
        direction = int(record["favorable_direction"])
        if record["mean_difference"] is None:
            record["favorable_shift"] = None
            record["favorable_ci95"] = [None, None]
        else:
            record["favorable_shift"] = float(record["mean_difference"]) * direction
            record["favorable_ci95"] = sorted(
                [float(value) * direction for value in record["ci95"]]
            )
    return {
        "primary_leave_same_family_out_effects": primary_effects,
        "panel_sensitivities": summaries,
        "pairwise_judge_agreement": agreement(scores, inference),
        "same_family_affinity": affinity,
        "same_family_interpretation": (
            "The adjusted diagonal interaction accounts for additive judge severity and "
            "target-family baselines. Because this study contains exactly one target model "
            "per family and that model is also the family judge, it cannot separate broad "
            "family affinity from exact judge-target-model interaction and is not a causal "
            "bias estimate."
        ),
        "judge_quality": judge_quality(scores),
    }


def _primary_effect_rows(primary_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for outcome in primary_report["outcome_reports"]:
        for contrast in outcome["contrasts"]:
            effect = contrast["effect"]
            sign = contrast["motif_level_sign_test"]
            rows.append(
                {
                    "frame": contrast["frame"],
                    "metric": contrast["metric"],
                    "mean_difference": effect["mean_paired_difference"],
                    "ci_low": effect["ci_95"][0],
                    "ci_high": effect["ci_95"][1],
                    "matched_dyads": effect["matched_dyads"],
                    "motif_clusters": effect["motif_clusters"],
                    "p_motif_sign": sign["p_value"],
                    "q_bh": sign["fdr_bh_adjusted_p_value"],
                }
            )
    return rows


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flatten_wording(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric_report in report["metrics"]:
        for family in metric_report["families"]:
            for effect in family["wording_specific_effects"]:
                rows.append(
                    {
                        "metric": effect["metric"],
                        "frame": effect["frame"],
                        "prompt_variant_id": effect["prompt_variant_id"],
                        "mean_difference": effect["mean_difference"],
                        "ci_low": effect["ci95"][0],
                        "ci_high": effect["ci95"][1],
                        "p_motif_sign": effect["p_motif_sign"],
                        "q_bh_112": effect["q_bh_across_112_wording_tests"],
                    }
                )
    return rows


def _flatten_flips(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for metric_report in report["metrics"]:
        for family in metric_report["families"]:
            flips = family["cell_level_strict_polarity_flip"]
            canonical = family["canonical_v1_vs_family_average"]
            rows.append(
                {
                    "metric": metric_report["metric"],
                    "frame": family["frame"],
                    "complete_cells": flips["complete_cells"],
                    "strict_polarity_flip_count": flips["strict_polarity_flip_count"],
                    "strict_polarity_flip_rate": flips["strict_polarity_flip_rate"],
                    "canonical_nonzero_comparable_cells": canonical[
                        "nonzero_comparable_cells"
                    ],
                    "canonical_family_sign_flip_count": canonical[
                        "strict_sign_flip_count"
                    ],
                    "canonical_family_sign_flip_rate": canonical[
                        "strict_sign_flip_rate"
                    ],
                    "mean_within_family_sd": family["mean_within_family_sd"],
                    "mean_within_family_range": family["mean_within_family_range"],
                }
            )
    return rows


def plots(
    output: Path, primary_rows: list[dict[str, Any]], wording: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "belief_amplification": "Belief amplification ↑ risk",
        "epistemic_grounding": "Epistemic grounding ↑ better",
        "uncertainty_calibration": "Uncertainty calibration ↑ better",
        "harmful_elaboration": "Harmful elaboration ↑ risk",
        "supportive_engagement": "Supportive engagement ↑ better",
        "response_appropriateness": "Inappropriateness ↑ risk",
    }
    used = list(labels)
    palette = ["#b65c3a", "#277e80", "#8165a1", "#4a74a1"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, used, strict=True):
        records = [row for row in primary_rows if row["metric"] == metric]
        for y, record in enumerate(records):
            value = record["mean_difference"]
            axis.errorbar(
                value,
                y,
                xerr=[
                    [value - record["ci_low"]],
                    [record["ci_high"] - value],
                ],
                fmt="o",
                color=palette[y],
                capsize=3,
            )
        axis.axvline(0, color="#aaaaaa", linewidth=1)
        axis.set_yticks(range(4), [frame.replace("_", " ") for frame in FRAMES[1:]])
        axis.invert_yaxis()
        axis.set_title(labels[metric], fontsize=10)
        axis.set_xlabel("Four-wording family mean − neutral (95% CI)", fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Paraphrase-robust frame effects | 4 targets, 12 motifs, 3 structures",
        fontsize=13,
    )
    fig.savefig(output / "paraphrase_family_effects.png", dpi=180)
    plt.close(fig)

    flip_rows = _flatten_flips(wording)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, used, strict=True):
        records = [row for row in flip_rows if row["metric"] == metric]
        values = [row["strict_polarity_flip_rate"] for row in records]
        axis.bar(range(4), values, color=palette)
        axis.set_xticks(
            range(4),
            [frame.replace("_", "\n") for frame in FRAMES[1:]],
            fontsize=8,
        )
        axis.set_ylim(0, max(0.05, max(values, default=0) * 1.15))
        axis.set_title(labels[metric], fontsize=10)
        axis.set_ylabel("Cell-level strict polarity flip rate")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Sensitivity of frame direction to wording", fontsize=13)
    fig.savefig(output / "paraphrase_flip_rates.png", dpi=180)
    plt.close(fig)


def _summary_markdown(report: dict[str, Any]) -> str:
    primary = report["primary_effect_rows"]
    significant = [row for row in primary if row["q_bh"] < 0.05]
    wording = report["paraphrase_robustness"]
    flip_rows = _flatten_flips(wording)
    ratios = {
        item["metric"]: item["variance_decomposition"][
            "within_to_between_variance_ratio"
        ]
        for item in wording["metrics"]
    }
    panel_lines = []
    for panel in report["judge_sensitivity"]["panel_sensitivities"]:
        if panel["panel"].startswith("judge_only_"):
            panel_lines.append(
                f"- `{panel['panel']}`: {panel['same_direction_count']}/"
                f"{panel['comparable_estimand_count']} estimable directions match "
                "the primary panel."
            )
    lines = [
        "# Paraphrase robustness analysis",
        "",
        "## Design and verification",
        "",
        "- 12 outcome-blind motifs, 36 scenarios, 17 context variants, and four target models.",
        "- 2,448 complete responses and 9,792 successful judgments; no failed rows were analyzed.",
        "- The primary estimand equally averages four wording variants within each non-neutral family, then compares the family mean with the matched neutral response.",
        "- Confidence intervals resample 12 motif clusters; BH-FDR is applied across the 28 coequal family-by-rubric tests.",
        "",
        "## Primary frame effects",
        "",
        f"- {len(significant)}/28 family-level sign tests have BH-FDR q < .05.",
    ]
    for row in significant:
        lines.append(
            f"- `{row['frame']}` × `{row['metric']}`: Δ={row['mean_difference']:.3f}, "
            f"95% CI [{row['ci_low']:.3f}, {row['ci_high']:.3f}], q={row['q_bh']:.4g}."
        )
    lines.extend(["", "## Wording robustness", ""])
    for row in flip_rows:
        rate = row["strict_polarity_flip_rate"]
        canonical = row["canonical_family_sign_flip_rate"]
        lines.append(
            f"- `{row['frame']}` × `{row['metric']}`: strict wording polarity flips "
            f"{row['strict_polarity_flip_count']}/{row['complete_cells']}"
            f" ({rate:.1%})"
            + (
                f"; canonical-vs-family sign flips {canonical:.1%}."
                if canonical is not None
                else "; canonical-vs-family sign flip rate not estimable."
            )
        )
    lines.extend(["", "Within-/between-frame variance ratios:", ""])
    for metric, ratio in ratios.items():
        lines.append(
            f"- `{metric}`: {ratio:.3f}."
            if ratio is not None
            else f"- `{metric}`: not estimable."
        )
    lines.extend(["", "## Judge sensitivity", "", *panel_lines, ""])
    lines.extend(
        [
            "The same-family affinity analysis is diagnostic rather than causal: each family has only one target model here, and it is also the judge model for that family.",
            "",
            "## Interpretation boundary",
            "",
            "These estimates are conditional on the four target models, four local judges, one fixed generation seed, and the 12-motif outcome-blind robustness sample. Automated-judge validity still requires the planned expert annotation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_v3_paraphrase_robustness.json",
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=ROOT / "configs" / "paraphrase_robustness_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    rows = bundle["detailed_results"]
    config = json.loads(args.config.read_text(encoding="utf-8"))
    design = json.loads(args.design.read_text(encoding="utf-8"))
    arrays = build_score_arrays(rows, config, design)
    inference = Inference(len(arrays["motifs"]), args.draws, args.seed)

    primary = build_v3_coequal_report(
        rows,
        draws=args.draws,
        seed=args.seed,
        canonical_only=False,
        include_controls=False,
        overlap_cluster_map={},
    )
    canonical = build_v3_coequal_report(
        rows,
        draws=args.draws,
        seed=args.seed + 100000,
        canonical_only=True,
        include_controls=False,
        overlap_cluster_map={},
    )
    robustness = paraphrase_diagnostics(
        arrays["primary"],
        inference,
        models=arrays["models"],
        motifs=arrays["motifs"],
    )
    sensitivity = judge_sensitivity(
        arrays["scores"],
        arrays["primary"],
        arrays["model_families"],
        inference,
    )
    primary_rows = _primary_effect_rows(primary)
    report = {
        "metadata": {
            "analysis_version": "1.0",
            "source": str(args.bundle),
            "source_sha256": _sha256(args.bundle),
            "config_sha256": _sha256(args.config),
            "design_sha256": _sha256(args.design),
            "bootstrap_draws": args.draws,
            "bootstrap_seed": args.seed,
            "rows": len(rows),
            "judgments": sum(len(row.get("judge_results") or []) for row in rows),
            "models": arrays["models"],
            "model_families": arrays["model_families"],
            "motifs": arrays["motifs"],
            "frames": FRAMES,
            "variants": VARIANTS,
            "structures": STRUCTURES,
            "metrics": METRICS,
            "limitations": [
                "The 12 motifs were selected outcome-blind for heterogeneity coverage, not as a probability sample of all conspiracy narratives.",
                "Inference resamples motifs only; target models, judges, structures, and wordings are fixed.",
                "There is one seeded generation per condition, so wording effects cannot be separated from residual stochastic generation variation beyond common seeding.",
                "Automated-judge scores have not yet been validated against the planned expert annotations.",
                "Same-family affinity cannot distinguish a family effect from exact judge-target-model interaction in this four-model sample.",
                "Appropriate-redirection scores are analyzed only when applicable and are never imputed as zero.",
            ],
        },
        "primary_family_analysis": primary,
        "canonical_v1_sensitivity": canonical,
        "primary_effect_rows": primary_rows,
        "paraphrase_robustness": robustness,
        "judge_sensitivity": sensitivity,
    }
    _write_json(args.output / "statistics.json", report)
    _csv(args.output / "primary_frame_effects.csv", primary_rows)
    _csv(args.output / "wording_specific_effects.csv", _flatten_wording(robustness))
    _csv(args.output / "wording_flip_rates.csv", _flatten_flips(robustness))
    _write_json(args.output / "judge_sensitivity.json", sensitivity)
    plots(args.output, primary_rows, robustness)
    (args.output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(args.output / "statistics.json")
    print(args.output / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
