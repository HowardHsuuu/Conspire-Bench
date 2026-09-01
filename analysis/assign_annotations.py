#!/usr/bin/env python3
"""Create deterministic per-annotator blinded packages and an assignment ledger."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.export_annotations import (
    validate_public_items,
    write_annotation_csv,
    write_jsonl,
)
from analysis.human_annotation_plan import (
    human_annotation_plan_digest,
    load_human_annotation_plan,
)
from experiment_conditions import stable_digest
from rubric_v2 import RUBRIC_VERSION


def read_public_items(path: Path, expected_type: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no items")
    validate_public_items(rows)
    wrong = [
        row.get("annotation_item_id")
        for row in rows
        if row.get("item_type") != expected_type
    ]
    if wrong:
        raise ValueError(f"{path}: expected only {expected_type} items")
    return rows


def _validate_ids(ids: list[str], label: str) -> list[str]:
    cleaned = [str(value).strip() for value in ids]
    if any(not value for value in cleaned) or len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{label} IDs must be non-empty and unique")
    return cleaned


def load_roster_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("roster_status") != "frozen":
        raise ValueError("Annotator roster manifest must be a frozen JSON object")
    rows = manifest.get("annotators")
    if not isinstance(rows, list):
        raise ValueError("Annotator roster manifest has no annotators list")
    if manifest.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError(
            "Annotator roster rubric_version does not match the current rubric"
        )
    if manifest.get("roster_digest") != stable_digest(rows, length=64):
        raise ValueError("Annotator roster digest does not match its annotators")
    derived = {
        "eligible_expert_calibration_ids": sorted(
            str(row["annotator_id"])
            for row in rows
            if row.get("eligible_for_expert_calibration") is True
        ),
        "eligible_expert_formal_ids": sorted(
            str(row["annotator_id"])
            for row in rows
            if row.get("eligible_for_expert_formal") is True
        ),
        "eligible_student_formal_ids": sorted(
            str(row["annotator_id"])
            for row in rows
            if row.get("eligible_for_student_formal") is True
        ),
    }
    for field, expected in derived.items():
        if sorted(manifest.get(field) or []) != expected:
            raise ValueError(
                f"Annotator roster {field} does not match its annotator records"
            )
    return manifest


def validate_roster_assignments(
    manifest: dict[str, Any],
    expert_ids: list[str],
    student_ids: list[str],
    *,
    release_mode: str,
) -> None:
    if release_mode == "formal":
        expert_allowed = set(manifest.get("eligible_expert_formal_ids") or [])
        student_allowed = set(manifest.get("eligible_student_formal_ids") or [])
    elif release_mode == "calibration":
        expert_allowed = set(manifest.get("eligible_expert_calibration_ids") or [])
        student_allowed = set()
        if student_ids:
            raise ValueError("Calibration assignment mode is for expert items only")
    else:
        return
    invalid_experts = sorted(set(expert_ids) - expert_allowed)
    invalid_students = sorted(set(student_ids) - student_allowed)
    if invalid_experts or invalid_students:
        details = []
        if invalid_experts:
            details.append("ineligible experts: " + ", ".join(invalid_experts))
        if invalid_students:
            details.append("ineligible students: " + ", ".join(invalid_students))
        raise ValueError("Roster eligibility check failed (" + "; ".join(details) + ")")


def balanced_assignments(
    items: list[dict[str, Any]],
    annotator_ids: list[str],
    ratings_per_item: int,
    *,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    annotator_ids = _validate_ids(annotator_ids, "Annotator")
    if not 1 <= ratings_per_item <= len(annotator_ids):
        raise ValueError(f"ratings_per_item must be between 1 and {len(annotator_ids)}")
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_index, item in enumerate(shuffled):
        for offset in range(ratings_per_item):
            annotator = annotator_ids[(item_index + offset) % len(annotator_ids)]
            assigned = json.loads(json.dumps(item))
            assigned["annotator_id"] = annotator
            output[annotator].append(assigned)
    for index, annotator in enumerate(annotator_ids):
        random.Random(seed + 1009 + index).shuffle(output[annotator])
    return dict(output)


def _write_packages(
    output_dir: Path,
    assignments: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for annotator, items in sorted(assignments.items()):
        jsonl_path = output_dir / f"{annotator}.jsonl"
        csv_path = output_dir / f"{annotator}.csv"
        write_jsonl(jsonl_path, items)
        write_annotation_csv(csv_path, items)
        written.extend([jsonl_path, csv_path])
    return written


def build_assignment_manifest(
    expert_items: list[dict[str, Any]],
    student_items: list[dict[str, Any]],
    expert_assignments: dict[str, list[dict[str, Any]]],
    student_assignments: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    expert_ratings_per_item: int,
    student_ratings_per_item: int,
    release_mode: str = "pilot",
    roster_manifest: dict[str, Any] | None = None,
    annotation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assignment_rows: list[dict[str, Any]] = []
    for study, assignments in (
        ("expert_conversation", expert_assignments),
        ("paired_conversation", student_assignments),
    ):
        for annotator, items in sorted(assignments.items()):
            assignment_rows.extend(
                {
                    "item_type": study,
                    "annotation_item_id": item["annotation_item_id"],
                    "annotator_id": annotator,
                }
                for item in items
            )
    assignment_rows.sort(
        key=lambda row: (
            row["item_type"],
            row["annotation_item_id"],
            row["annotator_id"],
        )
    )
    type_counts = Counter(row["item_type"] for row in assignment_rows)
    annotator_counts = Counter(row["annotator_id"] for row in assignment_rows)
    all_annotators = sorted({*expert_assignments, *student_assignments})
    rubric_versions = sorted(
        {str(item.get("rubric_version")) for item in [*expert_items, *student_items]}
    )
    manifest = {
        "schema_version": "1.1",
        "assignment_status": "frozen",
        "release_mode": release_mode,
        "roster_digest": roster_manifest.get("roster_digest")
        if roster_manifest
        else None,
        "annotation_plan_version": (
            annotation_plan.get("plan_version") if annotation_plan else None
        ),
        "annotation_plan_digest": (
            human_annotation_plan_digest(annotation_plan) if annotation_plan else None
        ),
        "seed": seed,
        "rubric_versions": rubric_versions,
        "expert_item_digest": stable_digest(expert_items, length=64)
        if expert_items
        else None,
        "student_item_digest": stable_digest(student_items, length=64)
        if student_items
        else None,
        "expert_item_count": len(expert_items),
        "student_item_count": len(student_items),
        "expert_annotator_ids": sorted(expert_assignments),
        "student_annotator_ids": sorted(student_assignments),
        "expert_ratings_per_item": expert_ratings_per_item if expert_items else 0,
        "student_ratings_per_item": student_ratings_per_item if student_items else 0,
        "assignment_counts_by_type": dict(sorted(type_counts.items())),
        "assignment_counts_by_annotator": {
            annotator: annotator_counts.get(annotator, 0)
            for annotator in all_annotators
        },
        "assignments": assignment_rows,
    }
    manifest["assignment_digest"] = stable_digest(assignment_rows, length=64)
    manifest["manifest_digest"] = stable_digest(manifest, length=64)
    return manifest


def assign_package(
    *,
    expert_path: Path | None,
    student_path: Path | None,
    output_dir: Path,
    expert_ids: list[str],
    student_ids: list[str],
    expert_ratings_per_item: int,
    student_ratings_per_item: int,
    seed: int,
    release_mode: str = "pilot",
    roster_manifest_path: Path | None = None,
    annotation_plan_path: Path | None = None,
) -> list[Path]:
    expert_items = (
        read_public_items(expert_path, "expert_conversation") if expert_path else []
    )
    student_items = (
        read_public_items(student_path, "paired_conversation") if student_path else []
    )
    if not expert_items and not student_items:
        raise ValueError("Provide --expert-items and/or --student-items")
    if expert_items and not expert_ids:
        raise ValueError("--expert-ids is required with --expert-items")
    if student_items and not student_ids:
        raise ValueError("--student-ids is required with --student-items")
    expert_ids = _validate_ids(expert_ids, "Expert") if expert_ids else []
    student_ids = _validate_ids(student_ids, "Student") if student_ids else []
    if set(expert_ids) & set(student_ids):
        raise ValueError("Expert and student pseudonymous ID sets must not overlap")
    if release_mode not in {"pilot", "calibration", "formal"}:
        raise ValueError("release_mode must be pilot, calibration, or formal")
    if release_mode in {"calibration", "formal"} and roster_manifest_path is None:
        raise ValueError(f"{release_mode} assignments require --roster-manifest")
    if release_mode in {"calibration", "formal"} and annotation_plan_path is None:
        raise ValueError(f"{release_mode} assignments require --annotation-plan")
    annotation_plan = (
        load_human_annotation_plan(
            annotation_plan_path, require_frozen=release_mode == "formal"
        )
        if annotation_plan_path
        else None
    )
    if annotation_plan:
        plan_digest = human_annotation_plan_digest(annotation_plan)
        item_digests = {
            str(item.get("annotation_plan_digest") or "")
            for item in [*expert_items, *student_items]
        }
        if release_mode == "formal" and item_digests != {plan_digest}:
            raise ValueError(
                "Formal public items are not bound to the supplied human annotation plan"
            )
        expected_seed = annotation_plan["randomization"]["assignment_seed"]
        if release_mode in {"calibration", "formal"} and seed != expected_seed:
            raise ValueError(
                f"Assignment seed {seed} does not match annotation plan seed {expected_seed}"
            )
        if release_mode == "formal":
            expert_formal = annotation_plan["expert_panel"]["formal_sample"]
            student_plan = annotation_plan["student_panel"]
            if expert_items and len(expert_items) != expert_formal["total_item_count"]:
                raise ValueError(
                    "Expert item count does not match the frozen annotation plan"
                )
            if (
                student_items
                and len(student_items) != student_plan["paired_item_count"]
            ):
                raise ValueError(
                    "Student pair count does not match the frozen annotation plan"
                )
            if (
                expert_items
                and expert_ratings_per_item != expert_formal["ratings_per_item"]
            ):
                raise ValueError(
                    "Expert ratings per item do not match the frozen annotation plan"
                )
            if (
                student_items
                and student_ratings_per_item != student_plan["ratings_per_item"]
            ):
                raise ValueError(
                    "Student ratings per item do not match the frozen annotation plan"
                )
            if expert_items and not 2 <= len(expert_ids) <= 3:
                raise ValueError(
                    "Formal expert roster must contain two or three experts"
                )
            if (
                student_items
                and len(student_ids) < student_plan["minimum_eligible_annotator_count"]
            ):
                raise ValueError(
                    "Formal student roster is smaller than the prespecified minimum"
                )
        elif release_mode == "calibration":
            expert_plan = annotation_plan["expert_panel"]
            if student_items:
                raise ValueError("Calibration mode cannot assign student items")
            if len(expert_items) != expert_plan["calibration_item_count"]:
                raise ValueError(
                    "Calibration item count does not match the annotation plan"
                )
            if not 2 <= len(expert_ids) <= 3:
                raise ValueError("Calibration requires two or three eligible experts")
            if expert_ratings_per_item != len(expert_ids):
                raise ValueError("Every calibration item must be rated by every expert")
    roster_manifest = (
        load_roster_manifest(roster_manifest_path) if roster_manifest_path else None
    )
    if roster_manifest:
        validate_roster_assignments(
            roster_manifest, expert_ids, student_ids, release_mode=release_mode
        )

    expert_assignments = (
        balanced_assignments(
            expert_items, expert_ids, expert_ratings_per_item, seed=seed
        )
        if expert_items
        else {}
    )
    student_assignments = (
        balanced_assignments(
            student_items, student_ids, student_ratings_per_item, seed=seed + 1
        )
        if student_items
        else {}
    )
    written = []
    if expert_assignments:
        written.extend(_write_packages(output_dir / "expert", expert_assignments))
    if student_assignments:
        written.extend(_write_packages(output_dir / "student", student_assignments))
    manifest = build_assignment_manifest(
        expert_items,
        student_items,
        expert_assignments,
        student_assignments,
        seed=seed,
        expert_ratings_per_item=expert_ratings_per_item,
        student_ratings_per_item=student_ratings_per_item,
        release_mode=release_mode,
        roster_manifest=roster_manifest,
        annotation_plan=annotation_plan,
    )
    manifest_path = output_dir / "assignment_manifest.private.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-items", type=Path)
    parser.add_argument("--student-items", type=Path)
    parser.add_argument("--expert-ids", nargs="*", default=[])
    parser.add_argument("--student-ids", nargs="*", default=[])
    parser.add_argument("--expert-ratings-per-item", type=int, default=2)
    parser.add_argument("--student-ratings-per-item", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--release-mode", choices=("pilot", "calibration", "formal"), default="pilot"
    )
    parser.add_argument("--roster-manifest", type=Path)
    parser.add_argument("--annotation-plan", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    annotation_plan_path = args.annotation_plan
    if args.release_mode in {"calibration", "formal"} and annotation_plan_path is None:
        annotation_plan_path = ROOT / "configs" / "human_annotation_plan_v2.json"
    annotation_plan = (
        load_human_annotation_plan(
            annotation_plan_path, require_frozen=args.release_mode == "formal"
        )
        if annotation_plan_path
        else None
    )
    assignment_seed = (
        annotation_plan["randomization"]["assignment_seed"]
        if annotation_plan and args.seed == 20260831
        else args.seed
    )
    for path in assign_package(
        expert_path=args.expert_items,
        student_path=args.student_items,
        output_dir=args.out_dir,
        expert_ids=args.expert_ids,
        student_ids=args.student_ids,
        expert_ratings_per_item=args.expert_ratings_per_item,
        student_ratings_per_item=args.student_ratings_per_item,
        seed=assignment_seed,
        release_mode=args.release_mode,
        roster_manifest_path=args.roster_manifest,
        annotation_plan_path=annotation_plan_path,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
