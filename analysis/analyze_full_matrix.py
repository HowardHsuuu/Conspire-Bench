#!/usr/bin/env python3
"""Audit a completed canonical-frame matrix without changing source judgments.

Run with a Python environment containing numpy and matplotlib. All inference is
conditional on these target models, judges, and one wording/seed per condition.
Motifs, not conversations or judge votes, are the resampling units.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.frame_effect_stats import (  # noqa: E402
    benjamini_hochberg_adjust,
    exact_binom_two_sided,
    load_overlap_cluster_map,
)
from analysis.rubric_consistency_audit import rule_hits  # noqa: E402
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
FAMILIES = ["qwen", "llama", "gemma", "gpt_oss"]
HARM = METRICS.index("harmful_elaboration")
REDIR = METRICS.index("appropriate_redirection")
FAVORABLE = np.array([-1, 1, 1, -1, 1, 1, -1])
TEMPLATE = [1, 1, 1, False, 1, None, 0]
IDEAL = [1, 5, 5, False, 5, None, 0]
TEMPORAL = [
    "first_amplification_turn",
    "amplification_turn_count",
    "stance_drift",
    "recovery_after_pressure",
    "escalation_pattern",
    "over_refusal",
    "dependency_or_isolation_reinforcement",
]


def finite_mean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


class Inference:
    def __init__(self, clusters, draws, seed):
        self.clusters = clusters
        self.draws = draws
        self.seed = seed
        rng = np.random.default_rng(seed)
        samples = rng.integers(0, clusters, size=(draws, clusters))
        self.weights = np.zeros((draws, clusters), dtype=float)
        np.add.at(self.weights, (np.arange(draws)[:, None], samples), 1 / clusters)

    def summarize(self, differences):
        """Equal-weight cluster means, conditional complete pairs for missing scores."""
        d = np.asarray(differences, dtype=float).reshape(
            self.clusters, -1, len(METRICS)
        )
        cluster = finite_mean(d, axis=1)
        present = np.isfinite(cluster)
        denom = self.weights @ present.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.weights @ np.nan_to_num(cluster)) / denom
        result = []
        for k, metric in enumerate(METRICS):
            vals = cluster[present[:, k], k]
            positive = int((vals > 1e-12).sum())
            negative = int((vals < -1e-12).sum())
            valid_boot = boot[np.isfinite(boot[:, k]), k]
            result.append(
                {
                    "metric": metric,
                    "matched_observations": int(np.isfinite(d[..., k]).sum()),
                    "motif_clusters": len(vals),
                    "mean_difference": float(vals.mean()) if len(vals) else None,
                    "ci95": np.quantile(valid_boot, [0.025, 0.975]).tolist()
                    if len(valid_boot)
                    else [None, None],
                    "positive_motifs": positive,
                    "negative_motifs": negative,
                    "tie_motifs": len(vals) - positive - negative,
                    "p_motif_sign": exact_binom_two_sided(
                        min(positive, negative), positive + negative
                    ),
                }
            )
        return result


def bh(records, p="p_motif_sign", output="q_bh"):
    for row, q in zip(
        records,
        benjamini_hochberg_adjust([r[p] for r in records]),
        strict=True,
    ):
        row[output] = q


def aggregate(x, model_families, *, exclude_self=True, panel=None, harm_rule="any"):
    a = x.copy()
    panel = set(FAMILIES if panel is None else panel)
    for m, family in enumerate(model_families):
        for j, judge in enumerate(FAMILIES):
            if judge not in panel or (exclude_self and judge == family):
                a[:, m, :, :, j, :] = np.nan
    result = finite_mean(a, axis=4)
    if harm_rule == "any":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result[..., HARM] = np.nanmax(a[..., HARM], axis=4)
    return result


def frame_effects(a, inference):
    records = []
    for f, frame in enumerate(FRAMES[1:], 1):
        for r in inference.summarize(a[:, :, :, f, :] - a[:, :, :, 0, :]):
            r.update(frame=frame, reference="neutral")
            records.append(r)
    bh(records)
    return records


def rankdata(a):
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    ranks = np.cumsum(counts) - (counts - 1) / 2
    return ranks[inverse]


def kappa_from_confusion(c, levels, weighted=True):
    c = np.asarray(c, dtype=float)
    n = c.sum(axis=(-1, -2))
    distance = (
        np.abs(levels[:, None] - levels[None, :])
        if weighted
        else (levels[:, None] != levels[None, :]).astype(float)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = (c * distance).sum(axis=(-1, -2)) / n
        expected = (
            c.sum(axis=-1)[..., :, None] * c.sum(axis=-2)[..., None, :] * distance
        ).sum(axis=(-1, -2)) / n**2
        return np.where(expected > 0, 1 - observed / expected, np.nan)


def agreement(x, inference):
    result = []
    for k, metric in enumerate(METRICS):
        levels = (
            np.arange(2)
            if k == HARM
            else (
                np.arange(4)
                if k == REDIR
                else (
                    np.arange(3)
                    if metric == "response_appropriateness"
                    else np.arange(1, 6)
                )
            )
        )
        for ja, jb in itertools.combinations(range(4), 2):
            a = x[..., ja, k].reshape(len(x), -1)
            b = x[..., jb, k].reshape(len(x), -1)
            conf = np.zeros((len(x), len(levels), len(levels)))
            valid = np.isfinite(a) & np.isfinite(b)
            for c in range(len(x)):
                for ia, va in enumerate(levels):
                    for ib, vb in enumerate(levels):
                        conf[c, ia, ib] = ((a[c] == va) & (b[c] == vb)).sum()
            av, bv = a[valid], b[valid]
            if len(av) and np.std(av) > 0 and np.std(bv) > 0:
                rho = float(np.corrcoef(rankdata(av), rankdata(bv))[0, 1])
            else:
                rho = None
            estimate = kappa_from_confusion(conf.sum(axis=0), levels, k != HARM)
            boot_conf = (inference.weights @ conf.reshape(len(x), -1)).reshape(
                -1, len(levels), len(levels)
            )
            boot = kappa_from_confusion(boot_conf, levels, k != HARM)
            finite = boot[np.isfinite(boot)]
            rec = {
                "metric": metric,
                "judge_a": FAMILIES[ja],
                "judge_b": FAMILIES[jb],
                "paired_ratings": len(av),
                "exact_agreement": float((av == bv).mean()) if len(av) else None,
                "mean_absolute_difference": float(np.abs(av - bv).mean())
                if len(av)
                else None,
                "spearman_rho": rho,
                "kappa": float(estimate),
                "kappa_type": "unweighted" if k == HARM else "linear_weighted",
                "kappa_ci95": np.quantile(finite, [0.025, 0.975]).tolist()
                if len(finite)
                else [None, None],
                "confusion_levels": levels.tolist(),
                "confusion": conf.sum(axis=0).astype(int).tolist(),
            }
            if k == HARM:
                both = int(((av == 1) & (bv == 1)).sum())
                either = int(((av == 1) | (bv == 1)).sum())
                rec.update(
                    both_positive=both,
                    either_positive=either,
                    positive_agreement=2 * both / (int(av.sum() + bv.sum()))
                    if (av.sum() + bv.sum())
                    else None,
                )
            result.append(rec)
    return result


def family_affinity(x, model_families, inference):
    """Judge + target-family additive baselines and four own-family interactions.

    Each target family receives equal weight; within a family models are equally
    weighted. This adjusts generic judge severity but is not a causal bias test.
    """
    cells = np.stack(
        [
            finite_mean(x[:, np.array(model_families) == family], axis=(1, 2, 3))
            for family in FAMILIES
        ],
        axis=1,
    )  # motif,target_family,judge,metric
    design = []
    for t in range(4):
        for j in range(4):
            design.append(
                [1]
                + [int(j == a) for a in range(1, 4)]
                + [int(t == a) for a in range(1, 4)]
                + [int(j == a and t == a) for a in range(4)]
            )
    design = np.asarray(design, dtype=float)
    assert np.linalg.matrix_rank(design) == 11
    inverse = np.linalg.pinv(design)
    coefficients = np.einsum(
        "pq,cqk->cpk", inverse, cells.reshape(len(x), 16, len(METRICS))
    )
    adjusted, raw = [], []
    for j, judge in enumerate(FAMILIES):
        for r in inference.summarize(coefficients[:, 7 + j, :]):
            r.update(
                judge=judge,
                coefficient="own_family_interaction_adjusted_for_judge_and_target_family",
                favorable_direction=int(FAVORABLE[METRICS.index(r["metric"])]),
            )
            adjusted.append(r)
        gap = cells[:, j, j, :] - finite_mean(
            np.delete(cells[:, j, :, :], j, axis=1), axis=1
        )
        for r in inference.summarize(gap):
            r.update(
                judge=judge, coefficient="raw_same_minus_other_judges_on_own_family"
            )
            raw.append(r)
    bh(adjusted)
    return {
        "method": family_affinity.__doc__,
        "cell_means": finite_mean(cells, axis=0),
        "cell_axis_order": ["target_family", "judge", "metric"],
        "family_order": FAMILIES,
        "adjusted": adjusted,
        "raw": raw,
    }


def model_scaling(primary, models, config, inference):
    lookup = {m["provider"] + "/" + m["model"]: m for m in config["models"]}
    series = {
        "qwen_2_5": [m for m in models if "/Qwen2.5-" in m],
        "gemma_3": [m for m in models if "/gemma-3-" in m],
        "llama_3_mixed_versions": [m for m in models if "/Llama-3." in m],
        "gpt_oss": [m for m in models if "/gpt-oss-" in m],
    }
    records = []
    for series_name, names in series.items():
        if len(names) < 2:
            continue
        indices = [models.index(n) for n in names]
        sizes = np.array([lookup[n]["parameter_scale_b"] for n in names])
        z = np.log2(sizes)
        z = (z - z.mean()) / ((z - z.mean()) ** 2).sum()
        subset = primary[:, indices]
        levels = finite_mean(subset, axis=(2, 3))
        slopes = np.einsum("m,cmk->ck", z, levels)
        for r in inference.summarize(slopes):
            r.update(
                series=series_name,
                models=names,
                scales_b=sizes.tolist(),
                estimand="score_change_per_parameter_doubling",
            )
            records.append(r)
        delta = finite_mean(subset[:, :, :, 1, :] - subset[:, :, :, 0, :], axis=2)
        for r in inference.summarize(np.einsum("m,cmk->ck", z, delta)):
            r.update(
                series=series_name,
                models=names,
                scales_b=sizes.tolist(),
                estimand="brainstorm_effect_change_per_parameter_doubling",
            )
            records.append(r)
    bh(records)
    return records


def plots(out, report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "belief_amplification": "Belief amplification ↑ risk",
        "epistemic_grounding": "Epistemic grounding ↑ better",
        "uncertainty_calibration": "Uncertainty calibration ↑ better",
        "harmful_elaboration": "Any-judge elaboration flag ↑ risk",
        "supportive_engagement": "Supportive engagement ↑ better",
        "response_appropriateness": "Inappropriateness ↑ risk",
    }
    used = [m for m in METRICS if m != "appropriate_redirection"]
    palette = ["#b65c3a", "#277e80", "#8165a1", "#4a74a1"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, metric in zip(axes.flat, used, strict=True):
        recs = [r for r in report["frame_effects"] if r["metric"] == metric]
        for y, r in enumerate(recs):
            v, (lo, hi) = r["mean_difference"], r["ci95"]
            ax.errorbar(
                v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=palette[y], capsize=3
            )
        ax.axvline(0, color="#aaaaaa", lw=1)
        ax.set_yticks(range(4), [f.replace("_", " ") for f in FRAMES[1:]])
        ax.invert_yaxis()
        ax.set_title(labels[metric], fontsize=10)
        ax.set_xlabel("Frame − neutral; motif-bootstrap 95% CI", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Canonical framing effects | 17 targets, 51 motifs, 3 structures\nOther-family judges; current scores are not human validated",
        fontsize=13,
    )
    fig.savefig(out / "frame_effects.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    means = np.array(report["judge_means_by_target_family"], dtype=float)
    for ax, metric in zip(axes.flat, used, strict=True):
        k = METRICS.index(metric)
        mat = means[:, :, k]
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        for y in range(4):
            for j in range(4):
                ax.text(
                    j,
                    y,
                    f"{mat[y, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mat[y, j] < np.mean(mat) else "black",
                    fontsize=9,
                )
        ax.set_xticks(range(4), FAMILIES)
        ax.set_yticks(range(4), FAMILIES)
        ax.set_title(
            "Elaboration flag rate (individual judge)"
            if metric == "harmful_elaboration"
            else labels[metric],
            fontsize=10,
        )
        ax.set_xlabel("Judge")
        ax.set_ylabel("Target family (equal model weights)")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle(
        "Judge severity and target-family differences | raw scores, not accuracy",
        fontsize=13,
    )
    fig.savefig(out / "judge_family_matrix.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260917)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading immutable source bundle", flush=True)
    bundle = json.loads(args.bundle.read_text())
    rows = bundle["detailed_results"]
    config = json.loads(args.config.read_text())
    models = [m["provider"] + "/" + m["model"] for m in config["models"]]
    model_families = [m["model_family"] for m in config["models"]]
    motifs = sorted({r["scenario_metadata"]["motif_id"] for r in rows})
    assert len(rows) == 13005 and len(models) == 17 and len(motifs) == 51
    x = np.full((len(motifs), len(models), 3, 5, 4, len(METRICS)), np.nan)
    stored = np.full(x.shape[:4] + (len(METRICS),), np.nan)
    occupied = np.zeros(x.shape[:4], dtype=bool)
    truncated = np.zeros(x.shape[:4], dtype=bool)
    word_counts = np.zeros(x.shape[:4])
    audit = {
        j: {
            "rating_count": 0,
            "template_vector_count": 0,
            "ideal_vector_count": 0,
            "placeholder_rationale_count": 0,
            "all_rationales_placeholder_count": 0,
            "raw_score_mismatches": 0,
            "raw_parse_failures": 0,
            "rule_hits": Counter(),
            "vectors": Counter(),
            "rationales": Counter(),
            "metric_distributions": {m: Counter() for m in METRICS},
            "temporal_field_counts": Counter(),
            "grounding_1_positive_rationale_heuristic": 0,
            "finish_reasons": Counter(),
        }
        for j in FAMILIES
    }
    examples, example_counts = [], Counter()
    variant_counts, generation_configs = Counter(), Counter()
    response_ids = set()
    for r in rows:
        assert r["response_id"] not in response_ids
        response_ids.add(r["response_id"])
        assert (
            r.get("generation_complete")
            and r.get("evaluation_complete")
            and not r.get("error")
        )
        index = (
            motifs.index(r["scenario_metadata"]["motif_id"]),
            models.index(r["model_name"]),
            STRUCTURES.index(r["scenario_type"]),
            FRAMES.index(r["frame_family"]),
        )
        assert not occupied[index]
        occupied[index] = True
        variant_counts[r["prompt_variant_id"]] += 1
        generation_configs[
            (
                r["model_name"],
                json.dumps(r.get("generation_config"), sort_keys=True),
                str(r.get("generation_seed")),
            )
        ] += 1
        assistant = [m for m in r["conversation_log"] if m["role"] == "assistant"]
        truncated[index] = any(
            (m.get("response_metadata") or {}).get("finish_reason") == "length"
            for m in assistant
        )
        word_counts[index] = sum(len(m["content"].split()) for m in assistant)
        stored[index] = [r["judge_scores"].get(k, np.nan) for k in METRICS]
        assert len(r["judge_results"]) == 4
        assert {j["judge_model_family"] for j in r["judge_results"]} == set(FAMILIES)
        for j in r["judge_results"]:
            assert not j.get("error") and j.get("scores")
            judge, s = j["judge_model_family"], j["scores"]
            x[index + (FAMILIES.index(judge),)] = [s.get(k, np.nan) for k in METRICS]
            a = audit[judge]
            a["rating_count"] += 1
            values = [s.get(k) for k in METRICS]
            a["template_vector_count"] += values == TEMPLATE
            a["ideal_vector_count"] += values == IDEAL
            a["vectors"][json.dumps(values)] += 1
            reasoning = j.get("reasoning") or {}
            norm = {
                k: re.sub(r"\s+", " ", str(v).strip().lower())
                for k, v in reasoning.items()
            }
            placeholder = [
                v
                in {
                    "brief evidence",
                    "brief evidence or not applicable",
                    "brief global assessment",
                }
                for v in norm.values()
            ]
            a["placeholder_rationale_count"] += any(placeholder)
            a["all_rationales_placeholder_count"] += bool(placeholder) and all(
                placeholder
            )
            a["rationales"][json.dumps(norm, sort_keys=True)] += 1
            positive = (
                bool(
                    re.search(
                        r"\b(clearly separates|clearly distinguishes|provides clear|strong grounding)\b",
                        norm.get("epistemic_grounding", ""),
                    )
                )
                and s.get("epistemic_grounding") == 1
            )
            a["grounding_1_positive_rationale_heuristic"] += positive
            a["rule_hits"].update(rule_hits(s))
            a["finish_reasons"][
                str((j.get("response_metadata") or {}).get("finish_reason"))
            ] += 1
            for k in METRICS:
                a["metric_distributions"][k][str(s.get(k))] += 1
            for k in TEMPORAL:
                a["temporal_field_counts"][k] += k in j or k in s
            try:
                raw = j["raw_response"]
                parsed = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                raw_values = [
                    parsed[k]["present" if k == "harmful_elaboration" else "score"]
                    for k in METRICS
                ]
                a["raw_score_mismatches"] += raw_values != values
            except (KeyError, AttributeError, ValueError, TypeError):
                a["raw_parse_failures"] += 1
            reason = (
                "grounding_score_rationale_tension"
                if positive
                else ("literal_placeholder_rationales" if any(placeholder) else None)
            )
            if reason and example_counts[(judge, reason)] < 3:
                example_counts[(judge, reason)] += 1
                examples.append(
                    {
                        "selection": "first three matching source rows per judge/flag; purposive, not prevalence validation",
                        "flag": reason,
                        "judge": judge,
                        "response_id": r["response_id"],
                        "target": r["model_name"],
                        "frame": r["frame_family"],
                        "scenario_id": r["scenario_id"],
                        "scores": s,
                        "reasoning": reasoning,
                        "conversation": r["conversation_log"],
                    }
                )
    assert occupied.all()
    expected_variants = {"neutral_none", *(f + "_v1" for f in FRAMES[1:])}
    assert set(variant_counts) == expected_variants
    assert all(n == 2601 for n in variant_counts.values())
    primary = aggregate(x, model_families)
    assert np.allclose(primary, stored, equal_nan=True), (
        "Reconstruction does not match frozen aggregate"
    )
    inference = Inference(len(motifs), args.draws, args.seed)
    quality = []
    for j, a in audit.items():
        vectors = a.pop("vectors")
        rationales = a.pop("rationales")
        a.update(
            judge=j,
            unique_score_vectors=len(vectors),
            unique_rationale_sets=len(rationales),
            most_common_vectors=[
                {"scores": json.loads(k), "count": n} for k, n in vectors.most_common(5)
            ],
            largest_duplicate_rationale_set=max(rationales.values()),
        )
        ji = FAMILIES.index(j)
        a["grounding_calibration_exact_match_rate"] = float(
            (x[..., ji, 1] == x[..., ji, 2]).mean()
        )
        quality.append(a)
    print("Computing motif-clustered frame effects and sensitivities", flush=True)
    report = {
        "metadata": {
            "source": str(args.bundle),
            "source_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
            "analysis_version": "1.0",
            "seed": args.seed,
            "bootstrap_draws": args.draws,
            "rows": len(rows),
            "models": models,
            "motifs": motifs,
            "model_families": model_families,
            "frames": FRAMES,
            "structures": STRUCTURES,
            "metrics": METRICS,
            "judge_order": FAMILIES,
            "variant_counts": variant_counts,
            "truncated_conversations": int(truncated.sum()),
            "generation_configs": [
                {"model": k[0], "config": json.loads(k[1]), "seed": k[2], "rows": n}
                for k, n in generation_configs.items()
            ],
            "limitations": [
                "One canonical wording and one response per condition; no paraphrase or stochastic-repeat inference.",
                "Rubric v2 differs from original paper; no overall_safety score or direct scale-equivalence claim.",
                "Judge validity has not been established against humans. All individual scores are preserved.",
                "Stored harmful_elaboration is an OR over other-family judges, not a consensus or mean judge rate.",
                "Temporal rubric fields were not requested by the automated judge prompt and cannot be analyzed.",
                "Missing redirection scores mean judge-declared not applicable; do not impute zero.",
                "Sign-test p-values test motif effect direction; bootstrap CIs estimate mean effects.",
                "Model, judge, and dataset populations are fixed; bootstrap uncertainty is across motifs only.",
                "Family affinity is an adjusted association, not causal proof of self-preference.",
            ],
        },
        "frame_effects": frame_effects(primary, inference),
        "frame_means": [
            {
                "frame": f,
                "scores": dict(
                    zip(
                        METRICS,
                        finite_mean(primary[:, :, :, fi, :], axis=(0, 1, 2)),
                        strict=True,
                    )
                ),
            }
            for fi, f in enumerate(FRAMES)
        ],
        "judge_quality": quality,
        "sensitivities": {},
    }
    for j, judge in enumerate(FAMILIES):
        report["sensitivities"]["judge_only_" + judge] = frame_effects(
            x[..., j, :], inference
        )
        report["sensitivities"]["leave_out_" + judge] = frame_effects(
            aggregate(x, model_families, panel=set(FAMILIES) - {judge}), inference
        )
    report["judge_specific_structure_effects"] = []
    for j, judge in enumerate(FAMILIES):
        for s, structure in enumerate(STRUCTURES):
            for r in frame_effects(x[:, :, s : s + 1, :, j, :], inference):
                r.update(judge=judge, structure=structure)
                report["judge_specific_structure_effects"].append(r)
    report["sensitivities"]["all_four_judges"] = frame_effects(
        aggregate(x, model_families, exclude_self=False), inference
    )
    report["sensitivities"]["mean_judge_harm_instead_of_or"] = frame_effects(
        aggregate(x, model_families, harm_rule="mean"), inference
    )
    report["sensitivities"]["gemma_gptoss_fixed_panel_includes_own_family"] = (
        frame_effects(
            aggregate(
                x, model_families, exclude_self=False, panel={"gemma", "gpt_oss"}
            ),
            inference,
        )
    )
    no_truncation = primary.copy()
    no_truncation[truncated] = np.nan
    report["sensitivities"]["exclude_truncated_conversations"] = frame_effects(
        no_truncation, inference
    )
    overlap = load_overlap_cluster_map()
    cluster_names = sorted({overlap.get(m, m) for m in motifs})
    grouped = np.stack(
        [
            finite_mean(primary[[overlap.get(m, m) == group for m in motifs]], axis=0)
            for group in cluster_names
        ]
    )
    report["sensitivities"]["overlap_cluster_grouping"] = frame_effects(
        grouped, Inference(len(grouped), args.draws, args.seed + 1)
    )
    report["overlap_cluster_count"] = len(grouped)
    by_model, by_structure, interactions = [], [], []
    for m, model in enumerate(models):
        for r in frame_effects(primary[:, m : m + 1], inference):
            r["model"] = model
            by_model.append(r)
    for s, structure in enumerate(STRUCTURES):
        for r in frame_effects(primary[:, :, s : s + 1], inference):
            r["structure"] = structure
            by_structure.append(r)
        if s:
            for fi, frame in enumerate(FRAMES):
                difference = primary[:, :, s, fi, :] - primary[:, :, 0, fi, :]
                for r in inference.summarize(difference):
                    r.update(
                        structure=structure,
                        reference_structure=STRUCTURES[0],
                        frame=frame,
                        type="structure_contrast",
                    )
                    interactions.append(r)
                if fi:
                    did = difference - (primary[:, :, s, 0, :] - primary[:, :, 0, 0, :])
                    for r in inference.summarize(did):
                        r.update(
                            structure=structure,
                            reference_structure=STRUCTURES[0],
                            frame=frame,
                            type="frame_by_structure_interaction",
                        )
                        interactions.append(r)
    bh(interactions)
    report.update(
        by_model=by_model,
        by_structure=by_structure,
        structure_interactions=interactions,
        model_frame_means=[
            {
                "model": model,
                "family": model_families[m],
                "frame": frame,
                "scores": dict(
                    zip(
                        METRICS,
                        finite_mean(primary[:, m, :, fi, :], axis=(0, 1)),
                        strict=True,
                    )
                ),
                "mean_assistant_words": float(word_counts[:, m, :, fi].mean()),
            }
            for m, model in enumerate(models)
            for fi, frame in enumerate(FRAMES)
        ],
    )
    report["critical_review_paired_direction"] = []
    d = primary[:, :, :, 2, :] - primary[:, :, :, 0, :]
    for k, metric in enumerate(METRICS):
        v = d[..., k].ravel()
        v = v[np.isfinite(v)]
        report["critical_review_paired_direction"].append(
            {
                "metric": metric,
                "n": len(v),
                "increase": int((v > 1e-12).sum()),
                "decrease": int((v < -1e-12).sum()),
                "tie": int((np.abs(v) <= 1e-12).sum()),
                "note": "Numeric direction only, not the original paper's qualitative epistemic-posture labels.",
            }
        )
    print("Computing judge agreement and severity-adjusted family affinity", flush=True)
    report["judge_agreement"] = agreement(x, inference)
    report["family_affinity"] = family_affinity(x, model_families, inference)
    judge_models = {j["provider"] + "/" + j["model"] for j in config["judges"]}
    other_models = [m for m, name in enumerate(models) if name not in judge_models]
    report["family_affinity_excluding_exact_judge_targets"] = family_affinity(
        x[:, other_models], [model_families[m] for m in other_models], inference
    )
    report["family_affinity_excluding_exact_judge_targets"]["retained_models"] = [
        models[m] for m in other_models
    ]
    report["judge_means_by_target_family"] = report["family_affinity"]["cell_means"]
    report["scaling"] = model_scaling(primary, models, config, inference)
    report["redirection"] = {
        "applicable_per_judge": {
            j: int(np.isfinite(x[..., ji, REDIR]).sum())
            for ji, j in enumerate(FAMILIES)
        },
        "applicable_per_frame_per_judge": {
            frame: {
                j: int(np.isfinite(x[:, :, :, fi, ji, REDIR]).sum())
                for ji, j in enumerate(FAMILIES)
            }
            for fi, frame in enumerate(FRAMES)
        },
        "primary_applicable_conversations": int(np.isfinite(primary[..., REDIR]).sum()),
    }
    save(args.output / "statistics.json", report)
    save(args.output / "judge_audit_examples.json", examples)
    np.savez_compressed(
        args.output / "analysis_arrays.npz",
        scores=x,
        primary=primary,
        truncated=truncated,
        assistant_words=word_counts,
    )
    plots(args.output, report)
    print(
        json.dumps(
            clean(
                {
                    "frame_effects": report["frame_effects"],
                    "redirection": report["redirection"],
                }
            )
        ),
        flush=True,
    )
    print("Saved analysis to", args.output, flush=True)


if __name__ == "__main__":
    main()
