#!/usr/bin/env python3
"""Validate and summarize returned expert rubric content-validity forms."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.export_annotations import RUBRIC_DIMENSIONS, RUBRIC_VERSION

RATING_FIELDS = [
    "definition_clarity_1_5",
    "construct_relevance_1_5",
    "distinctness_1_5",
    "observable_from_transcript_1_5",
]
RECOMMENDATIONS = {"retain", "revise", "remove"}
GLOBAL_TEXT_FIELDS = [
    "missing_constructs",
    "redundant_constructs",
    "difficult_boundary_cases",
    "clinically_inappropriate_assumptions",
    "risk_of_overclaiming_deployment_or_clinical_safety",
    "recommended_scope_statement",
    "overall_comments",
]


def _integer_1_5(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer 1-5")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer 1-5") from error
    if str(value).strip() not in {str(parsed), f"{parsed}.0"} or not 1 <= parsed <= 5:
        raise ValueError(f"{label} must be an integer 1-5")
    return parsed


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    dimension_rows = [row for row in rows if row.get("dimension") != "__global__"]
    global_rows = [row for row in rows if row.get("dimension") == "__global__"]
    if len(global_rows) != 1:
        raise ValueError(f"{path}: CSV must contain exactly one __global__ row")
    expert_ids = {str(row.get("expert_id") or "").strip() for row in rows}
    expert_ids.discard("")
    if len(expert_ids) != 1:
        raise ValueError(f"{path}: provide one consistent non-empty expert_id")
    expertise_values = {
        str(row.get("expertise_description_nonidentifying") or "").strip()
        for row in rows
    }
    expertise_values.discard("")
    if len(expertise_values) > 1:
        raise ValueError(
            f"{path}: expertise description must be consistent across rows"
        )
    global_row = global_rows[0]
    return {
        "form_type": "expert_rubric_content_validity_feedback",
        "rubric_version": global_row.get("rubric_version"),
        "expert_id": next(iter(expert_ids)),
        "expertise_description_nonidentifying": next(iter(expertise_values), ""),
        "dimensions": dimension_rows,
        "global_feedback": {
            "overall_content_validity_1_5": global_row.get(
                "overall_content_validity_1_5"
            ),
            **{field: global_row.get(field, "") for field in GLOBAL_TEXT_FIELDS},
        },
    }


def read_feedback(path: Path) -> dict[str, Any]:
    return _read_csv(path) if path.suffix.lower() == ".csv" else _read_json(path)


def validate_feedback(value: dict[str, Any], path: Path) -> dict[str, Any]:
    if value.get("form_type") != "expert_rubric_content_validity_feedback":
        raise ValueError(f"{path}: wrong form_type")
    if value.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError(f"{path}: expected rubric_version {RUBRIC_VERSION}")
    expert_id = str(value.get("expert_id") or "").strip()
    if not expert_id:
        raise ValueError(f"{path}: expert_id is required")
    dimensions = value.get("dimensions")
    if not isinstance(dimensions, list):
        raise ValueError(f"{path}: dimensions must be a list")
    by_dimension = {str(row.get("dimension")): row for row in dimensions}
    if len(by_dimension) != len(dimensions) or set(by_dimension) != set(
        RUBRIC_DIMENSIONS
    ):
        raise ValueError(f"{path}: must rate every rubric dimension exactly once")
    normalized_dimensions = []
    for dimension in RUBRIC_DIMENSIONS:
        row = by_dimension[dimension]
        recommendation = str(row.get("recommendation") or "").strip().lower()
        if recommendation not in RECOMMENDATIONS:
            raise ValueError(
                f"{path}: {dimension} recommendation must be retain/revise/remove"
            )
        normalized_dimensions.append(
            {
                "dimension": dimension,
                **{
                    field: _integer_1_5(
                        row.get(field), label=f"{path}: {dimension} {field}"
                    )
                    for field in RATING_FIELDS
                },
                "recommendation": recommendation,
                "ambiguity_or_overlap_notes": str(
                    row.get("ambiguity_or_overlap_notes") or ""
                ),
                "suggested_revision": str(row.get("suggested_revision") or ""),
                "anchor_examples_needed": str(row.get("anchor_examples_needed") or ""),
            }
        )
    global_feedback = value.get("global_feedback")
    if not isinstance(global_feedback, dict):
        raise ValueError(f"{path}: global_feedback must be an object")
    normalized_global = {
        "overall_content_validity_1_5": _integer_1_5(
            global_feedback.get("overall_content_validity_1_5"),
            label=f"{path}: overall_content_validity_1_5",
        ),
        **{
            field: str(global_feedback.get(field) or "") for field in GLOBAL_TEXT_FIELDS
        },
    }
    return {
        "expert_id": expert_id,
        "expertise_description_nonidentifying": str(
            value.get("expertise_description_nonidentifying") or ""
        ),
        "rubric_version": RUBRIC_VERSION,
        "dimensions": normalized_dimensions,
        "global_feedback": normalized_global,
        "source_file": str(path),
    }


def import_feedback(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = [validate_feedback(read_feedback(path), path) for path in paths]
    expert_ids = [row["expert_id"] for row in rows]
    if len(expert_ids) != len(set(expert_ids)):
        raise ValueError(
            "Each expert_id may occur in only one returned rubric-validity file"
        )
    return rows


def _mean_sd(values: list[int]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "share_4_or_5": sum(value >= 4 for value in values) / len(values)
        if values
        else None,
    }


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dimension_report: dict[str, Any] = {}
    for dimension in RUBRIC_DIMENSIONS:
        ratings = [
            next(item for item in row["dimensions"] if item["dimension"] == dimension)
            for row in rows
        ]
        if len(rows) != len(ratings):
            raise RuntimeError("Rubric rows and ratings diverged")
        dimension_report[dimension] = {
            "ratings": {
                field: _mean_sd([rating[field] for rating in ratings])
                for field in RATING_FIELDS
            },
            "recommendations": dict(
                Counter(rating["recommendation"] for rating in ratings)
            ),
            "qualitative_feedback_by_expert": [
                {
                    "expert_id": row["expert_id"],
                    "ambiguity_or_overlap_notes": rating["ambiguity_or_overlap_notes"],
                    "suggested_revision": rating["suggested_revision"],
                    "anchor_examples_needed": rating["anchor_examples_needed"],
                }
                for row, rating in zip(rows, ratings)  # noqa: B905 - checked above
            ],
        }
    scale_cvi_average = {
        field: statistics.mean(
            dimension_report[dimension]["ratings"][field]["share_4_or_5"]
            for dimension in RUBRIC_DIMENSIONS
        )
        if rows
        else None
        for field in RATING_FIELDS
    }
    relevance_universal = (
        sum(
            all(
                next(
                    item for item in row["dimensions"] if item["dimension"] == dimension
                )["construct_relevance_1_5"]
                >= 4
                for row in rows
            )
            for dimension in RUBRIC_DIMENSIONS
        )
        / len(RUBRIC_DIMENSIONS)
        if rows
        else None
    )
    return {
        "schema_version": "1.0",
        "rubric_version": RUBRIC_VERSION,
        "expert_count": len(rows),
        "expert_ids": [row["expert_id"] for row in rows],
        "cvi_threshold": "A rating of 4 or 5 counts as content-valid for the reported CVI summaries.",
        "small_panel_caution": (
            "With only two or three experts, CVI values are descriptive evidence and should be "
            "reported with the raw counts and qualitative feedback, not treated as definitive validation."
        ),
        "dimensions": dimension_report,
        "scale_cvi_average_by_criterion": scale_cvi_average,
        "scale_cvi_universal_agreement_relevance": relevance_universal,
        "overall_content_validity": _mean_sd(
            [row["global_feedback"]["overall_content_validity_1_5"] for row in rows]
        ),
        "global_feedback_by_expert": [
            {
                "expert_id": row["expert_id"],
                **row["global_feedback"],
            }
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = import_feedback(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined_path = args.out_dir / "rubric_feedback_joined.private.json"
    report_path = args.out_dir / "rubric_content_validity_report.private.json"
    joined_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(build_report(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(joined_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
