#!/usr/bin/env python3
"""Render V3 coequal frame-effect figures from the frozen analysis report."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_v2 import RUBRIC_DIMENSIONS

FRAME_ORDER = (
    "brainstorming",
    "critical_review",
    "sensemaking",
    "supportive_listening",
)
FRAME_LABELS = {
    "brainstorming": "Brainstorming",
    "critical_review": "Critical review",
    "sensemaking": "Sensemaking",
    "supportive_listening": "Supportive listening",
}
OUTCOME_LABELS = {
    "belief_amplification": "Belief amplification",
    "epistemic_grounding": "Epistemic grounding",
    "uncertainty_calibration": "Uncertainty calibration",
    "harmful_elaboration": "Harmful elaboration",
    "supportive_engagement": "Supportive engagement",
    "appropriate_redirection": "Appropriate redirection\n(when applicable)",
    "response_appropriateness": "Response inappropriateness",
}
OUTCOME_DIRECTIONS = {
    "belief_amplification": "Positive = more amplification",
    "epistemic_grounding": "Positive = better grounding",
    "uncertainty_calibration": "Positive = better calibration",
    "harmful_elaboration": "Positive = higher harmful-elaboration rate",
    "supportive_engagement": "Positive = better support",
    "appropriate_redirection": "Positive = better redirection",
    "response_appropriateness": "Positive = more inappropriate",
}


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Expected one V3 frame-effect report object")
    validate_report(payload)
    return payload


def validate_report(report: dict[str, Any]) -> None:
    if report.get("design_version") != "v3":
        raise ValueError("Figure input must be a V3 frame-effect report")
    if report.get("outcome_status") != "coequal":
        raise ValueError("V3 figure input must declare coequal outcomes")
    if tuple(report.get("outcomes") or ()) != RUBRIC_DIMENSIONS:
        raise ValueError("Figure input does not contain the seven frozen outcomes")
    if tuple(report.get("frames") or ()) != FRAME_ORDER:
        raise ValueError("Figure input does not contain the four frame contrasts")
    if report.get("estimand_count") != 28:
        raise ValueError("Figure input must contain exactly 28 estimands")


def effect_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract plotting rows without constructing a cross-outcome composite."""

    validate_report(report)
    rows: list[dict[str, Any]] = []
    for outcome_report in report.get("outcome_reports") or []:
        outcome = str(outcome_report.get("metric") or "")
        if outcome not in RUBRIC_DIMENSIONS:
            raise ValueError(f"Unexpected outcome report: {outcome or '<missing>'}")
        contrasts = outcome_report.get("contrasts") or []
        if len(contrasts) != len(FRAME_ORDER):
            raise ValueError(f"Outcome {outcome} must contain four contrasts")
        by_frame = {str(item.get("frame")): item for item in contrasts}
        if set(by_frame) != set(FRAME_ORDER):
            raise ValueError(f"Outcome {outcome} has an invalid frame set")
        for frame in FRAME_ORDER:
            contrast = by_frame[frame]
            effect = contrast.get("effect") or {}
            interval = effect.get("ci_95")
            estimate = effect.get("mean_paired_difference")
            if estimate is not None and (
                not isinstance(interval, list)
                or len(interval) != 2
                or any(value is None for value in interval)
            ):
                raise ValueError(
                    f"Outcome {outcome}/{frame} has an estimate without a 95% CI"
                )
            sign_test = contrast.get("motif_level_sign_test") or {}
            rows.append(
                {
                    "outcome": outcome,
                    "frame": frame,
                    "estimate": estimate,
                    "ci_95": interval,
                    "matched_dyads": effect.get("matched_dyads", 0),
                    "motif_clusters": effect.get("motif_clusters", 0),
                    "fdr_bh_adjusted_p_value": sign_test.get("fdr_bh_adjusted_p_value"),
                }
            )
    if len(rows) != 28:
        raise ValueError("Figure extraction must produce exactly 28 rows")
    return rows


def render_frame_effect_figure(
    report: dict[str, Any], out_dir: Path, *, stem: str = "frame_effects_v3"
) -> list[Path]:
    """Render one seven-panel forest plot and return its PDF/PNG paths."""

    matplotlib = import_module("matplotlib")
    matplotlib.use("Agg")
    plt = import_module("matplotlib.pyplot")

    rows = effect_rows(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(12.0, 6.5), constrained_layout=True)
    flat_axes = list(axes.flat)
    colors = {
        "brainstorming": "#7c3aed",
        "critical_review": "#2563eb",
        "sensemaking": "#059669",
        "supportive_listening": "#d97706",
    }

    for axis, outcome in zip(flat_axes, RUBRIC_DIMENSIONS):  # noqa: B905
        outcome_rows = [row for row in rows if row["outcome"] == outcome]
        axis.axvline(0.0, color="#6b7280", linewidth=0.9, linestyle="--")
        for position, row in enumerate(outcome_rows):
            estimate = row["estimate"]
            interval = row["ci_95"]
            if estimate is None or interval is None:
                continue
            low, high = (float(interval[0]), float(interval[1]))
            estimate = float(estimate)
            q_value = row["fdr_bh_adjusted_p_value"]
            significant = isinstance(q_value, (int, float)) and q_value <= 0.05
            axis.errorbar(
                estimate,
                position,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=colors[str(row["frame"])],
                markeredgecolor="black" if significant else "none",
                markeredgewidth=0.8,
                capsize=2.5,
                linewidth=1.2,
            )
        axis.set_yticks(
            range(len(FRAME_ORDER)), [FRAME_LABELS[frame] for frame in FRAME_ORDER]
        )
        axis.set_title(OUTCOME_LABELS[outcome], fontsize=10, weight="bold")
        axis.set_xlabel(OUTCOME_DIRECTIONS[outcome], fontsize=7.8)
        axis.grid(axis="x", color="#e5e7eb", linewidth=0.7)
        axis.tick_params(labelsize=8)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)

    flat_axes[-1].axis("off")
    fig.suptitle(
        "Conspire-Bench V3: paired frame-minus-neutral effects\n"
        "Points are motif-cluster estimates; bars are 95% CIs; outlined points have BH-FDR q ≤ .05",
        fontsize=12,
        weight="bold",
    )
    outputs = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=240, bbox_inches="tight")
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", type=Path, help="V3 JSON report from frame_effect_stats.py"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory; defaults to <report-directory>/figures",
    )
    parser.add_argument("--stem", default="frame_effects_v3")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir or args.report.parent / "figures"
    for path in render_frame_effect_figure(
        load_report(args.report), out_dir, stem=args.stem
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
