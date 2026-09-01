#!/usr/bin/env python3
"""Load and validate the prespecified Conspire-Bench v2 human annotation plan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiment_conditions import stable_digest
from rubric_v2 import RUBRIC_VERSION


def load_human_annotation_plan(
    path: Path, *, require_frozen: bool = False
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Human annotation plan must be a JSON object")
    errors = validate_human_annotation_plan(plan, require_frozen=require_frozen)
    if errors:
        raise ValueError("Invalid human annotation plan:\n- " + "\n- ".join(errors))
    return plan


def human_annotation_plan_digest(plan: dict[str, Any]) -> str:
    return stable_digest(plan, length=64)


def validate_human_annotation_plan(
    plan: dict[str, Any], *, require_frozen: bool = False
) -> list[str]:
    errors: list[str] = []
    if plan.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    if not str(plan.get("plan_version") or "").strip():
        errors.append("plan_version is required")
    if plan.get("status") not in {
        "draft_to_freeze_after_timed_calibration",
        "frozen",
    }:
        errors.append("status must be draft_to_freeze_after_timed_calibration or frozen")
    if require_frozen and plan.get("status") != "frozen":
        errors.append("status must be frozen before formal annotation")
    if str(plan.get("rubric_version")) != RUBRIC_VERSION:
        errors.append(f"rubric_version must be {RUBRIC_VERSION}")
    rationale = plan.get("sample_size_rationale") or {}
    for field in ("expert", "student", "power_boundary"):
        if not str(rationale.get(field) or "").strip():
            errors.append(f"sample_size_rationale.{field} is required")

    experts = plan.get("expert_panel") or {}
    minimum = experts.get("eligible_expert_count_min")
    maximum = experts.get("eligible_expert_count_max")
    if minimum != 2 or maximum != 3:
        errors.append("expert panel must prespecify two to three eligible experts")
    if experts.get("same_panel_for_content_validity_calibration_and_formal_rating") is not True:
        errors.append("the same verified expert panel must cover validity, calibration, and formal rating")
    calibration_count = experts.get("calibration_item_count")
    if not isinstance(calibration_count, int) or not 12 <= calibration_count <= 20:
        errors.append("expert calibration_item_count must be between 12 and 20")
    if experts.get("calibration_ratings_per_item") != "all_eligible_experts":
        errors.append("every calibration item must be rated by all eligible experts")

    formal = experts.get("formal_sample") or {}
    representative = formal.get("representative_count")
    enriched = formal.get("judge_disagreement_enriched_count")
    total = formal.get("total_item_count")
    ratings = formal.get("ratings_per_item")
    assignments = formal.get("planned_assignment_count")
    if not all(isinstance(value, int) and value >= 0 for value in (representative, enriched)):
        errors.append("expert representative and enriched counts must be non-negative integers")
    elif total != representative + enriched:
        errors.append("expert total_item_count must equal representative plus enriched counts")
    if ratings != 2:
        errors.append("expert formal ratings_per_item must be 2")
    if isinstance(total, int) and isinstance(ratings, int) and assignments != total * ratings:
        errors.append("expert planned_assignment_count arithmetic does not match")
    if not str(formal.get("adjudication_policy") or "").strip():
        errors.append("expert adjudication_policy is required")

    students = plan.get("student_panel") or {}
    student_count = students.get("paired_item_count")
    student_ratings = students.get("ratings_per_item")
    student_assignments = students.get("planned_assignment_count")
    if students.get("minimum_eligible_annotator_count", 0) < student_ratings:
        errors.append("student minimum eligible count must be at least ratings_per_item")
    if not isinstance(student_count, int) or student_count <= 0:
        errors.append("student paired_item_count must be a positive integer")
    if student_ratings != 3:
        errors.append("student ratings_per_item must be 3")
    if (
        isinstance(student_count, int)
        and isinstance(student_ratings, int)
        and student_assignments != student_count * student_ratings
    ):
        errors.append("student planned_assignment_count arithmetic does not match")
    workload_examples = students.get("workload_examples") or {}
    if isinstance(student_assignments, int):
        for count, field in (
            (6, "six_students_items_each"),
            (9, "nine_students_items_each"),
            (12, "twelve_students_items_each"),
        ):
            if workload_examples.get(field) != student_assignments // count:
                errors.append(f"student workload example {field} does not match")

    randomization = plan.get("randomization") or {}
    for field in ("calibration_sampling_seed", "sampling_seed", "assignment_seed"):
        if not isinstance(randomization.get(field), int):
            errors.append(f"randomization.{field} must be an integer")
    reporting = plan.get("stratification_and_reporting") or {}
    expected_strata = {"target_model", "frame_family", "scenario_type", "category"}
    if set(reporting.get("representative_strata") or []) != expected_strata:
        errors.append(
            "representative_strata must bind target model, frame, scenario type, and category"
        )
    expected_pair_strata = {
        "comparison_frame_family",
        "scenario_type",
        "target_model",
        "category",
    }
    if set(reporting.get("student_pair_strata") or []) != expected_pair_strata:
        errors.append("student_pair_strata must bind frame, type, model, and category")
    if reporting.get("report_representative_and_disagreement_enriched_separately") is not True:
        errors.append("representative and disagreement-enriched results must be separate")

    change = plan.get("change_control") or {}
    if change.get("timed_calibration_required_before_final_freeze") is not True:
        errors.append("timed calibration must precede final plan freeze")
    if not change.get("forbidden") or not change.get("required_record"):
        errors.append("change control must state forbidden changes and the required record")
    if require_frozen:
        freeze = plan.get("freeze_record") or {}
        if not str(freeze.get("frozen_at") or "").strip():
            errors.append("freeze_record.frozen_at is required")
        approved = freeze.get("approved_by")
        if not isinstance(approved, list) or not approved or len(approved) != len(set(approved)):
            errors.append("freeze_record.approved_by must contain unique approver IDs")
        if freeze.get("timed_calibration_completed") is not True:
            errors.append("timed calibration must be completed before freeze")
        for field in (
            "expert_calibration_item_minutes_median",
            "student_pair_ui_item_minutes_median",
        ):
            value = freeze.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                errors.append(f"freeze_record.{field} must be a positive number")
        if freeze.get("workload_feasibility_confirmed") is not True:
            errors.append("workload feasibility must be confirmed before freeze")
        if not str(freeze.get("change_summary") or "").strip():
            errors.append("freeze_record.change_summary is required")
    return errors
