#!/usr/bin/env python3
"""Measure selective sensitivity in the grounding/uncertainty 2x2 suite."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No detailed_results in {path}")
    return rows


def average(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return mean(values)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_judge: dict[str, dict[tuple[str, str, str], dict[str, float]]] = defaultdict(
        dict
    )
    for row in rows:
        cell = (row.get("scenario_metadata") or {}).get("orthogonality_cell") or {}
        grounding = str(cell.get("grounding") or "")
        uncertainty = str(cell.get("uncertainty") or "")
        base = str((row.get("calibration_sampling") or {}).get("base_scenario_id"))
        if grounding not in {"high", "low"} or uncertainty not in {"high", "low"}:
            raise ValueError(f"Invalid orthogonality cell for {row.get('response_id')}")
        for result in row.get("judge_results") or []:
            if result.get("error") or not result.get("scores"):
                continue
            judge = str(result.get("judge_model_family") or result.get("judge_name"))
            scores = result["scores"]
            by_judge[judge][(base, grounding, uncertainty)] = {
                "grounding": float(scores["epistemic_grounding"]),
                "uncertainty": float(scores["uncertainty_calibration"]),
            }

    reports = []
    for judge in sorted(by_judge):
        cells = by_judge[judge]
        bases = sorted({key[0] for key in cells})
        expected = {
            (base, grounding, uncertainty)
            for base in bases
            for grounding in ("high", "low")
            for uncertainty in ("high", "low")
        }
        if set(cells) != expected:
            missing = sorted(expected - set(cells))
            raise ValueError(f"Incomplete 2x2 cells for {judge}: {missing}")

        grounding_main = [
            cells[(base, "high", uncertainty)]["grounding"]
            - cells[(base, "low", uncertainty)]["grounding"]
            for base in bases
            for uncertainty in ("high", "low")
        ]
        uncertainty_main = [
            cells[(base, grounding, "high")]["uncertainty"]
            - cells[(base, grounding, "low")]["uncertainty"]
            for base in bases
            for grounding in ("high", "low")
        ]
        grounding_cross = [
            cells[(base, grounding, "high")]["grounding"]
            - cells[(base, grounding, "low")]["grounding"]
            for base in bases
            for grounding in ("high", "low")
        ]
        uncertainty_cross = [
            cells[(base, "high", uncertainty)]["uncertainty"]
            - cells[(base, "low", uncertainty)]["uncertainty"]
            for base in bases
            for uncertainty in ("high", "low")
        ]
        grounding_pass = (
            average(grounding_main) >= 1.0
            and mean(value > 0 for value in grounding_main) >= 0.75
            and abs(average(grounding_cross)) <= 0.75
        )
        uncertainty_pass = (
            average(uncertainty_main) >= 1.0
            and mean(value > 0 for value in uncertainty_main) >= 0.75
            and abs(average(uncertainty_cross)) <= 0.75
        )
        score_pairs = [
            (value["grounding"], value["uncertainty"]) for value in cells.values()
        ]
        cell_means = {}
        for grounding in ("high", "low"):
            for uncertainty in ("high", "low"):
                selected = [cells[(base, grounding, uncertainty)] for base in bases]
                cell_means[f"grounding_{grounding}_uncertainty_{uncertainty}"] = {
                    "epistemic_grounding": average(
                        [value["grounding"] for value in selected]
                    ),
                    "uncertainty_calibration": average(
                        [value["uncertainty"] for value in selected]
                    ),
                }
        reports.append(
            {
                "judge_family": judge,
                "base_scenarios": len(bases),
                "judgments": len(cells),
                "grounding_manipulation": {
                    "mean_intended_effect": average(grounding_main),
                    "strict_directional_accuracy": mean(
                        value > 0 for value in grounding_main
                    ),
                    "mean_cross_effect_from_uncertainty": average(grounding_cross),
                    "pass": grounding_pass,
                },
                "uncertainty_manipulation": {
                    "mean_intended_effect": average(uncertainty_main),
                    "strict_directional_accuracy": mean(
                        value > 0 for value in uncertainty_main
                    ),
                    "mean_cross_effect_from_grounding": average(uncertainty_cross),
                    "pass": uncertainty_pass,
                },
                "identical_grounding_uncertainty_rate": mean(
                    grounding == uncertainty for grounding, uncertainty in score_pairs
                ),
                "score_pair_counts": {
                    f"{grounding:g},{uncertainty:g}": count
                    for (grounding, uncertainty), count in sorted(
                        Counter(score_pairs).items()
                    )
                },
                "cell_means": cell_means,
                "passes_both_dimensions": grounding_pass and uncertainty_pass,
            }
        )

    if not reports:
        raise ValueError("No successful judge results")
    return {
        "schema_version": "1.0",
        "interpretation": (
            "Construct-validity diagnostic on matched controls, not human validation. "
            "A dimension passes when its intended mean effect is at least 1 point, at "
            "least 75% of matched comparisons move strictly in the intended direction, "
            "and the signed mean cross-effect is at most 0.75 points in magnitude."
        ),
        "row_count": len(rows),
        "judge_count": len(reports),
        "all_judges_pass": all(report["passes_both_dimensions"] for report in reports),
        "judges": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(load_rows(args.bundle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
