#!/usr/bin/env python3
"""Validate a pseudonymous annotator roster and freeze eligibility by study stage."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_conditions import stable_digest
from rubric_v2 import RUBRIC_VERSION


FORBIDDEN_IDENTITY_FIELDS = {
    "name",
    "full_name",
    "email",
    "phone",
    "student_id",
    "employee_id",
}


def _parse_bool(value: Any, field: str, annotator_id: str) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"{annotator_id}: {field} must be true or false")


def _parse_number(value: Any, field: str, annotator_id: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{annotator_id}: {field} must be numeric") from error


def read_roster(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("annotators") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON roster must be a list or contain an annotators list")
    else:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {str(name).strip().lower() for name in (reader.fieldnames or [])}
            forbidden = sorted(fieldnames & FORBIDDEN_IDENTITY_FIELDS)
            if forbidden:
                raise ValueError(
                    "Roster must use pseudonymous IDs; remove identity fields: "
                    + ", ".join(forbidden)
                )
            rows = list(reader)
    if not rows:
        raise ValueError("Annotator roster is empty")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every annotator roster row must be an object")
    return rows


def normalize_roster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for raw in rows:
        annotator_id = str(raw.get("annotator_id") or "").strip()
        role = str(raw.get("role") or "").strip().lower()
        if not annotator_id or annotator_id in seen:
            raise ValueError("annotator_id values must be non-empty and unique")
        if role not in {"expert", "student"}:
            raise ValueError(f"{annotator_id}: role must be expert or student")
        seen.add(annotator_id)
        rubric_version = str(raw.get("rubric_version") or "").strip()
        version_matches = rubric_version == RUBRIC_VERSION
        expertise_verified = _parse_bool(
            raw.get("expertise_verified"), "expertise_verified", annotator_id
        )
        content_validity_complete = _parse_bool(
            raw.get("content_validity_complete"),
            "content_validity_complete",
            annotator_id,
        )
        calibration_complete = _parse_bool(
            raw.get("calibration_complete"), "calibration_complete", annotator_id
        )
        training_complete = _parse_bool(
            raw.get("training_complete"), "training_complete", annotator_id
        )
        score = _parse_number(raw.get("qualification_score"), "qualification_score", annotator_id)
        maximum = _parse_number(raw.get("qualification_max"), "qualification_max", annotator_id)
        threshold = _parse_number(
            raw.get("qualification_pass_threshold"),
            "qualification_pass_threshold",
            annotator_id,
        )
        if role == "student":
            if maximum is None or maximum <= 0 or score is None or threshold is None:
                student_quiz_passed = False
            else:
                if not 0 <= score <= maximum:
                    raise ValueError(f"{annotator_id}: qualification_score must be within 0..max")
                if not 0 <= threshold <= 1:
                    raise ValueError(
                        f"{annotator_id}: qualification_pass_threshold must be in 0..1"
                    )
                student_quiz_passed = score / maximum >= threshold
        else:
            student_quiz_passed = False

        expert_calibration_ready = (
            role == "expert"
            and version_matches
            and expertise_verified
            and content_validity_complete
        )
        expert_formal_ready = expert_calibration_ready and calibration_complete
        student_formal_ready = (
            role == "student"
            and version_matches
            and training_complete
            and student_quiz_passed
        )
        normalized.append({
            "annotator_id": annotator_id,
            "role": role,
            "rubric_version": rubric_version,
            "expertise_verified": expertise_verified,
            "content_validity_complete": content_validity_complete,
            "calibration_complete": calibration_complete,
            "training_complete": training_complete,
            "qualification_score": score,
            "qualification_max": maximum,
            "qualification_pass_threshold": threshold,
            "student_quiz_passed": student_quiz_passed,
            "eligible_for_expert_calibration": expert_calibration_ready,
            "eligible_for_expert_formal": expert_formal_ready,
            "eligible_for_student_formal": student_formal_ready,
            "exclusion_reason": str(raw.get("exclusion_reason") or "").strip(),
            "eligibility_note": str(raw.get("eligibility_note") or "").strip(),
        })
    return sorted(normalized, key=lambda row: row["annotator_id"])


def build_roster_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = normalize_roster(rows)
    manifest = {
        "schema_version": "1.0",
        "roster_status": "frozen",
        "rubric_version": RUBRIC_VERSION,
        "annotator_count": len(normalized),
        "eligible_expert_calibration_ids": [
            row["annotator_id"] for row in normalized
            if row["eligible_for_expert_calibration"]
        ],
        "eligible_expert_formal_ids": [
            row["annotator_id"] for row in normalized
            if row["eligible_for_expert_formal"]
        ],
        "eligible_student_formal_ids": [
            row["annotator_id"] for row in normalized
            if row["eligible_for_student_formal"]
        ],
        "annotators": normalized,
    }
    manifest["roster_digest"] = stable_digest(normalized, length=64)
    return manifest


def freeze_roster(input_path: Path, output_path: Path) -> Path:
    manifest = build_roster_manifest(read_roster(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    path = freeze_roster(args.input, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
