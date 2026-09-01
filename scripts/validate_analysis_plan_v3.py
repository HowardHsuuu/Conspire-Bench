#!/usr/bin/env python3
"""Validate the executable v3 design and its coequal outcome contract."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.human_annotation_plan import load_human_annotation_plan
from dataset_validation import validate_dataset
from experiment_conditions import load_context_set, stable_digest
from rubric_v2 import RUBRIC_DIMENSIONS, RUBRIC_VERSION
from scripts.validate_context_variants_v3 import FRAME_FAMILIES
from scripts.validate_interaction_catalog_v3 import validate as validate_interactions

DEFAULT_PLAN = ROOT / "configs" / "analysis_plan_v3.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str, root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def design_digest(plan: dict[str, Any]) -> str:
    design = copy.deepcopy(plan)
    design["status"] = "ready_for_local_generation_pending_human_rubric_freeze"
    design.pop("rubric_freeze_record", None)
    design.pop("scenario_review_record", None)
    design.pop("freeze_record", None)
    return stable_digest(design, length=64)


def _validate_frozen_evidence(
    plan: dict[str, Any],
    *,
    root: Path,
    manifest: dict[str, Any],
    catalog: dict[str, Any],
    dataset: dict[str, Any],
    quality: dict[str, Any],
    narratives: dict[str, Any],
    identity_policy: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    artifacts = plan.get("artifacts") or {}
    rubric_freeze = plan.get("rubric_freeze_record") or {}
    rubric_fields = (
        "content_validity_report",
        "content_validity_report_digest",
        "calibration_exclusion_manifest",
        "calibration_exclusion_manifest_digest",
        "calibration_decision_record",
        "calibration_decision_record_digest",
        "final_rubric_version",
    )
    for field in rubric_fields:
        if not rubric_freeze.get(field):
            errors.append(f"rubric_freeze_record.{field} is required")
    if all(rubric_freeze.get(field) for field in rubric_fields):
        try:
            validity = _read(
                _resolve(str(rubric_freeze["content_validity_report"]), root)
            )
            if validity.get("rubric_version") != str(artifacts["rubric_version"]):
                errors.append("content-validity report uses the wrong rubric version")
            expert_ids = validity.get("expert_ids") or []
            if validity.get("expert_count") not in {2, 3} or len(
                set(expert_ids)
            ) not in {2, 3}:
                errors.append(
                    "content-validity report must contain two or three unique experts"
                )
            if rubric_freeze["content_validity_report_digest"] != stable_digest(
                validity, length=64
            ):
                errors.append("content-validity report digest does not match")

            calibration = _read(
                _resolve(str(rubric_freeze["calibration_exclusion_manifest"]), root)
            )
            if calibration.get("status") != "frozen_calibration_exclusion":
                errors.append("calibration exclusion manifest is not frozen")
            if calibration.get("must_exclude_from_formal_annotation") is not True:
                errors.append("calibration responses must be excluded from formal data")
            if rubric_freeze["calibration_exclusion_manifest_digest"] != stable_digest(
                calibration, length=64
            ):
                errors.append("calibration exclusion manifest digest does not match")

            decision = _read(
                _resolve(str(rubric_freeze["calibration_decision_record"]), root)
            )
            if decision.get("status") != "approved_to_freeze":
                errors.append("calibration decision must be approved_to_freeze")
            if decision.get("rubric_version") != str(artifacts["rubric_version"]):
                errors.append("calibration decision uses the wrong rubric version")
            for field in ("independent_rating_complete", "amendments_applied"):
                if decision.get(field) is not True:
                    errors.append(f"calibration decision {field} must be true")
            if decision.get("unresolved_blocking_issues") is not False:
                errors.append("calibration decision has unresolved blocking issues")
            if set(decision.get("expert_ids") or []) != set(expert_ids):
                errors.append(
                    "calibration decision experts do not match content-validity experts"
                )
            if rubric_freeze["calibration_decision_record_digest"] != stable_digest(
                decision, length=64
            ):
                errors.append("calibration decision record digest does not match")
            if rubric_freeze["final_rubric_version"] != str(
                artifacts["rubric_version"]
            ):
                errors.append("final rubric version does not match V3 artifacts")
        except Exception as error:
            errors.append(f"rubric freeze evidence could not be validated: {error}")

    scenario_record = plan.get("scenario_review_record") or {}
    for field in ("approval_ledger", "approval_ledger_digest"):
        if not scenario_record.get(field):
            errors.append(f"scenario_review_record.{field} is required")
    if all(
        scenario_record.get(field)
        for field in ("approval_ledger", "approval_ledger_digest")
    ):
        try:
            ledger = _read(_resolve(str(scenario_record["approval_ledger"]), root))
            if (
                ledger.get("schema_version") != "3.0"
                or ledger.get("dataset_version") != "v3"
                or ledger.get("status") != "approved"
                or ledger.get("motif_count") != 51
                or ledger.get("blocking_review_count") != 0
            ):
                errors.append("scenario review ledger is not a complete V3 approval")
            if ledger.get("truth_adjudication") != "not_part_of_review":
                errors.append("scenario review must not adjudicate claim truth")
            if scenario_record["approval_ledger_digest"] != stable_digest(
                ledger, length=64
            ):
                errors.append("scenario review approval digest does not match")
            expected_digests = {
                "manifest": stable_digest(manifest, length=64),
                "narratives": stable_digest(narratives, length=64),
                "quality": stable_digest(quality, length=64),
                "catalog": stable_digest(catalog, length=64),
                "identity": stable_digest(identity_policy, length=64),
            }
            if ledger.get("artifact_digests") != expected_digests:
                errors.append(
                    "scenario review ledger does not bind current V3 artifacts"
                )
        except Exception as error:
            errors.append(f"scenario review evidence could not be validated: {error}")

    freeze = plan.get("freeze_record") or {}
    freeze_fields = (
        "frozen_at",
        "git_commit",
        "approved_by",
        "dataset_digest",
        "local_config_digest",
        "api_config_digest",
        "context_registry_digest",
        "human_annotation_plan_digest",
        "design_digest",
    )
    for field in freeze_fields:
        if not freeze.get(field):
            errors.append(f"freeze_record.{field} is required")
    approved_by = freeze.get("approved_by")
    if (
        not isinstance(approved_by, list)
        or not approved_by
        or len(approved_by) != len(set(approved_by))
    ):
        errors.append("freeze_record.approved_by must contain unique IDs")
    if freeze.get("git_commit") and not re.fullmatch(
        r"[0-9a-f]{40}", str(freeze["git_commit"])
    ):
        errors.append("freeze_record.git_commit must be a 40-character SHA")
    if freeze.get("frozen_at"):
        try:
            frozen_at = datetime.fromisoformat(
                str(freeze["frozen_at"]).replace("Z", "+00:00")
            )
            if frozen_at.tzinfo is None or frozen_at.utcoffset() is None:
                errors.append("freeze_record.frozen_at must include a timezone")
        except ValueError:
            errors.append("freeze_record.frozen_at must be ISO-8601")
    digest_expectations = {
        "dataset_digest": stable_digest(dataset, length=64),
        "local_config_digest": stable_digest(
            _read(_resolve(str(artifacts["local_config"]), root)), length=64
        ),
        "api_config_digest": stable_digest(
            _read(_resolve(str(artifacts["api_config"]), root)), length=64
        ),
        "context_registry_digest": stable_digest(
            _read(_resolve(str(artifacts["context_registry"]), root)), length=64
        ),
        "human_annotation_plan_digest": stable_digest(
            _read(_resolve(str(artifacts["human_annotation_plan"]), root)), length=64
        ),
        "design_digest": design_digest(plan),
    }
    for field, expected in digest_expectations.items():
        if freeze.get(field) and freeze[field] != expected:
            errors.append(f"freeze_record.{field} does not match current artifacts")
    return errors


def validate(
    plan: dict[str, Any], *, root: Path = ROOT, require_frozen: bool = False
) -> list[str]:
    errors: list[str] = []
    if plan.get("design_version") != "v3":
        errors.append("design_version must be v3")
    allowed_statuses = {
        "ready_for_local_generation_pending_human_rubric_freeze",
        "frozen",
    }
    if plan.get("status") not in allowed_statuses:
        errors.append("status must be a supported V3 draft or frozen state")
    if require_frozen and plan.get("status") != "frozen":
        errors.append("status must be frozen")
    artifacts = plan.get("artifacts") or {}
    required_artifacts = (
        "motif_manifest",
        "motif_narratives",
        "motif_quality_review",
        "interaction_catalog",
        "identity_policy",
        "dataset",
        "context_registry",
        "canonical_context_set",
        "full_context_set",
        "local_config",
        "api_config",
        "human_annotation_plan",
        "rubric_version",
    )
    artifact_errors: list[str] = []
    for field in required_artifacts:
        if not artifacts.get(field):
            artifact_errors.append(f"artifacts.{field} is required")
    errors.extend(artifact_errors)
    if artifact_errors:
        return errors

    manifest = _read(_resolve(artifacts["motif_manifest"], root))
    catalog = _read(_resolve(artifacts["interaction_catalog"], root))
    dataset = _read(_resolve(artifacts["dataset"], root))
    quality = _read(_resolve(artifacts["motif_quality_review"], root))
    narratives = _read(_resolve(artifacts["motif_narratives"], root))
    identity_policy = _read(_resolve(artifacts["identity_policy"], root))
    errors.extend(
        validate_interactions(
            catalog,
            dataset,
            manifest,
            quality,
            narratives,
            identity_policy,
        )
    )
    dataset_report = validate_dataset(dataset, strict_metadata=True)
    errors.extend(f"dataset: {error}" for error in dataset_report.errors)
    if len(dataset.get("scenarios", [])) != 153:
        errors.append("v3 dataset must contain 153 scenarios")

    registry = _resolve(artifacts["context_registry"], root)
    canonical = load_context_set(artifacts["canonical_context_set"], registry)
    full = load_context_set(artifacts["full_context_set"], registry)
    if tuple(item.frame for item in canonical) != FRAME_FAMILIES:
        errors.append("canonical context set must contain the five v3 frame families")
    if len(full) != 17:
        errors.append("full context set must contain 17 nested wording conditions")

    outcome_ids = [item.get("id") for item in plan.get("rubric_outcomes", [])]
    if tuple(outcome_ids) != RUBRIC_DIMENSIONS:
        errors.append("rubric_outcomes must exactly match the seven rubric dimensions")
    if artifacts.get("rubric_version") != RUBRIC_VERSION:
        errors.append("analysis-plan rubric version does not match code")
    grid = plan.get("estimand_grid") or {}
    expected_contrasts = [[frame, "neutral"] for frame in FRAME_FAMILIES[1:]]
    if grid.get("contrasts") != expected_contrasts:
        errors.append(
            "estimand grid must contain all four frame-minus-neutral contrasts"
        )
    if grid.get("outcomes") != list(RUBRIC_DIMENSIONS):
        errors.append("estimand grid must contain all seven outcomes")
    if (
        grid.get("status") != "coequal"
        or grid.get("total_frame_by_outcome_estimands") != 28
    ):
        errors.append("estimand grid must declare 28 coequal estimands")
    if (plan.get("wording_aggregation") or {}).get(
        "canonical_variant_privilege"
    ) is not False:
        errors.append("canonical wording must not be privileged in the full analysis")

    overlap_plan = (plan.get("uncertainty_and_multiplicity") or {}).get(
        "overlap_sensitivity"
    ) or {}
    overlap_groups = overlap_plan.get("cluster_groups") or {}
    manifest_ids = {item.get("motif_id") for item in manifest.get("motifs", [])}
    high_overlap_records = [
        item
        for item in quality.get("records", [])
        if item.get("distinctness") == "high_overlap"
    ]
    grouped_children: list[str] = []
    for record in high_overlap_records:
        motif_id = str(record.get("motif_id"))
        matching_groups = [
            members
            for members in overlap_groups.values()
            if isinstance(members, list) and motif_id in members
        ]
        if len(matching_groups) != 1:
            errors.append(
                f"high-overlap motif {motif_id} must occur in exactly one overlap cluster"
            )
            continue
        grouped_children.append(motif_id)
        expected_members = {motif_id, *(record.get("overlap_with") or [])}
        if set(matching_groups[0]) != expected_members:
            errors.append(
                f"overlap cluster for {motif_id} must equal its audited overlap_with family"
            )
    overlap_members = {
        member
        for members in overlap_groups.values()
        if isinstance(members, list)
        for member in members
    }
    if not overlap_members.issubset(manifest_ids):
        errors.append("overlap clusters may reference only frozen primary motifs")
    if len(grouped_children) != 3:
        errors.append("overlap sensitivity must cover all three high-overlap motifs")

    for field in ("local_config", "api_config", "human_annotation_plan"):
        if not _resolve(artifacts[field], root).exists():
            errors.append(f"missing artifact: {artifacts[field]}")
    try:
        load_human_annotation_plan(
            _resolve(str(artifacts["human_annotation_plan"]), root),
            require_frozen=require_frozen,
        )
    except Exception as error:
        errors.append(f"human annotation plan is invalid: {error}")
    if require_frozen:
        errors.extend(
            _validate_frozen_evidence(
                plan,
                root=root,
                manifest=manifest,
                catalog=catalog,
                dataset=dataset,
                quality=quality,
                narratives=narratives,
                identity_policy=identity_policy,
            )
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    plan = _read(args.plan)
    errors = validate(plan, require_frozen=args.require_frozen)
    print(
        json.dumps(
            {
                "ok": not errors,
                "plan": str(args.plan),
                "coequal_outcome_count": len(plan.get("rubric_outcomes", [])),
                "estimand_count": (plan.get("estimand_grid") or {}).get(
                    "total_frame_by_outcome_estimands"
                ),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
