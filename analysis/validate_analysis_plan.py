#!/usr/bin/env python3
"""Validate that the v2 analysis plan points to one internally consistent run."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation import validate_dataset
from experiment_conditions import load_context_conditions, load_context_set, stable_digest
from analysis.human_annotation_plan import load_human_annotation_plan
from scenario_expansion import load_benchmark_dataset, scenario_content_digest
from scenario_metadata import annotation_readiness_errors, enrich_dataset


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_plan(
    plan: dict[str, Any],
    *,
    root: Path = ROOT,
    require_frozen: bool = False,
) -> list[str]:
    errors: list[str] = []
    artifacts = plan.get("artifacts") or {}
    required = (
        "dataset",
        "main_config",
        "robustness_config",
        "context_registry",
        "main_context_set",
        "pilot_subset",
        "robustness_context_set",
        "exploratory_context_set",
        "robustness_subset",
        "human_annotation_plan",
        "rubric_version",
    )
    for field in required:
        if not artifacts.get(field):
            errors.append(f"artifacts.{field} is required")
    if errors:
        return errors

    try:
        human_plan = load_human_annotation_plan(
            _resolve(str(artifacts["human_annotation_plan"]), root),
            require_frozen=require_frozen,
        )
        if str(human_plan.get("rubric_version")) != str(artifacts["rubric_version"]):
            errors.append("human annotation plan uses the wrong rubric version")
    except Exception as exc:
        errors.append(f"human annotation plan could not be loaded: {exc}")

    try:
        dataset_path = _resolve(str(artifacts["dataset"]), root)
        dataset = enrich_dataset(load_benchmark_dataset(dataset_path))
        report = validate_dataset(dataset, strict_metadata=True)
        errors.extend(f"dataset: {error}" for error in report.errors)
        expected = int(artifacts.get("dataset_expected_scenarios", -1))
        if len(dataset.get("scenarios", [])) != expected:
            errors.append(
                f"dataset count is {len(dataset.get('scenarios', []))}, expected {expected}"
            )
    except Exception as exc:
        errors.append(f"dataset could not be loaded: {exc}")
        return errors

    try:
        registry_path = _resolve(str(artifacts["context_registry"]), root)
        conditions = load_context_conditions(registry_path)
        main_set = load_context_set(str(artifacts["main_context_set"]), registry_path)
        robust_set = load_context_set(
            str(artifacts["robustness_context_set"]), registry_path
        )
        exploratory_set = load_context_set(
            str(artifacts["exploratory_context_set"]), registry_path
        )
        main_ids = {condition.variant_id for condition in main_set}
        robust_ids = {condition.variant_id for condition in robust_set}
        canonical = (plan.get("confirmatory_population") or {}).get(
            "canonical_variants", {}
        )
        if set(canonical) != {"neutral", "brainstorming", "critical_review"}:
            errors.append("canonical_variants must define exactly three main frame families")
        for frame, variant_id in canonical.items():
            if variant_id not in main_ids:
                errors.append(f"canonical variant {variant_id} is not in the main set")
            elif conditions[variant_id].frame != frame:
                errors.append(f"canonical variant {variant_id} does not belong to {frame}")
        if not main_ids.issubset(robust_ids):
            errors.append("robustness context set must contain every main context variant")
        exploratory_ids = {condition.variant_id for condition in exploratory_set}
        if len(exploratory_ids) != 12:
            errors.append("exploratory context set must contain exactly 12 variants")
        if exploratory_ids & main_ids:
            errors.append("exploratory and main context sets must be disjoint")
        if any(condition.study_role != "exploratory" for condition in exploratory_set):
            errors.append("every exploratory context must have study_role=exploratory")
    except Exception as exc:
        errors.append(f"context registry could not be loaded: {exc}")

    scenario_by_id = {
        str(scenario["id"]): scenario for scenario in dataset.get("scenarios", [])
    }
    try:
        subset = _read(_resolve(str(artifacts["robustness_subset"]), root))
        ids = [str(value) for value in subset.get("scenario_ids", [])]
        if len(ids) != int(subset.get("selected_count", -1)):
            errors.append("robustness subset selected_count does not match scenario_ids")
        if len(ids) != len(set(ids)):
            errors.append("robustness subset contains duplicate scenario IDs")
        unknown = sorted(set(ids) - set(scenario_by_id))
        if unknown:
            errors.append(f"robustness subset has unknown scenarios: {unknown}")
        if not subset.get("include_controls", False):
            controls = [
                scenario_id
                for scenario_id in ids
                if scenario_by_id.get(scenario_id, {}).get("is_control")
            ]
            if controls:
                errors.append(f"robustness subset unexpectedly contains controls: {controls}")
        # Hash scenario content only: loader provenance paths can be relative or
        # absolute without changing the dataset that is actually analyzed.
        digest = scenario_content_digest(dataset)
        if subset.get("dataset_digest") != digest:
            errors.append("robustness subset dataset_digest does not match the dataset")
    except Exception as exc:
        errors.append(f"robustness subset could not be loaded: {exc}")

    try:
        pilot_subset = _read(_resolve(str(artifacts["pilot_subset"]), root))
        pilot_ids = [str(value) for value in pilot_subset.get("scenario_ids", [])]
        if len(pilot_ids) != 12 or int(pilot_subset.get("selected_count", -1)) != 12:
            errors.append("pilot subset must contain exactly 12 scenarios")
        if len(pilot_ids) != len(set(pilot_ids)):
            errors.append("pilot subset contains duplicate scenario IDs")
        if sorted(set(pilot_ids) - set(scenario_by_id)):
            errors.append("pilot subset contains unknown scenario IDs")
        if any(scenario_by_id.get(value, {}).get("is_control") for value in pilot_ids):
            errors.append("pilot subset must not contain matched controls")
        if pilot_subset.get("dataset_digest") != scenario_content_digest(dataset):
            errors.append("pilot subset dataset_digest does not match the dataset")
        if pilot_subset.get("purpose") != (
            "api_and_rubric_calibration_excluded_from_headline_estimates"
        ):
            errors.append("pilot subset purpose does not declare headline exclusion")
    except Exception as exc:
        errors.append(f"pilot subset could not be loaded: {exc}")

    try:
        config = _read(_resolve(str(artifacts["main_config"]), root))
        rubric_version = str(artifacts["rubric_version"])
        judges = config.get("judges") or []
        targets = config.get("models") or []
        if not judges or not targets:
            errors.append("main config must include target models and judges")
        for judge in judges:
            if str(judge.get("rubric_version")) != rubric_version:
                errors.append(
                    f"judge {judge.get('name') or judge.get('model')} uses the wrong rubric"
                )
        judge_providers = {str(judge.get("provider")) for judge in judges}
        expected_providers = {"openai", "anthropic", "gemini"}
        if len(judges) != 3 or judge_providers != expected_providers:
            errors.append("main config must include one strong judge from each API provider")
        target_provider_counts = {
            provider: sum(target.get("provider") == provider for target in targets)
            for provider in expected_providers
        }
        if len(targets) != 9 or set(target_provider_counts.values()) != {3}:
            errors.append("main config must include three targets from each API provider")
        for provider in expected_providers:
            tiers = {
                str(target.get("capacity_tier"))
                for target in targets
                if target.get("provider") == provider
            }
            if tiers != {"large", "medium", "efficient"}:
                errors.append(
                    f"main provider {provider} must include large, medium, and efficient tiers"
                )
        for target in targets:
            other = judge_providers - {str(target.get("provider"))}
            if not other:
                errors.append(
                    f"target {target.get('model')} has no non-overlapping judge provider"
                )
    except Exception as exc:
        errors.append(f"main config could not be loaded: {exc}")

    try:
        robustness_config = _read(
            _resolve(str(artifacts["robustness_config"]), root)
        )
        robustness_targets = robustness_config.get("models") or []
        providers = {
            str(target.get("provider")) for target in robustness_targets
        }
        if len(robustness_targets) != 6 or providers != {
            "openai",
            "anthropic",
            "gemini",
        }:
            errors.append(
                "robustness config must contain six targets spanning three providers"
            )
        for provider in providers:
            tiers = {
                str(target.get("capacity_tier"))
                for target in robustness_targets
                if target.get("provider") == provider
            }
            if tiers != {"large", "efficient"}:
                errors.append(
                    f"robustness provider {provider} must include large and efficient tiers"
                )
    except Exception as exc:
        errors.append(f"robustness config could not be loaded: {exc}")

    primary = plan.get("primary_estimands") or []
    expected_ids = {
        "brainstorming_minus_neutral_belief_amplification",
        "critical_review_minus_neutral_belief_amplification",
    }
    if {item.get("id") for item in primary} != expected_ids:
        errors.append("primary_estimands do not match the two prespecified contrasts")
    if any(item.get("metric") != "belief_amplification" for item in primary):
        errors.append("every primary estimand must use belief_amplification")

    if require_frozen:
        expansion_rows = [
            scenario
            for scenario in dataset.get("scenarios", [])
            if scenario.get("source_packet_id")
        ]
        readiness_failures = [
            scenario.get("id")
            for scenario in expansion_rows
            if annotation_readiness_errors(scenario)
        ]
        if readiness_failures:
            errors.append(
                f"{len(readiness_failures)} expansion scenarios lack accepted source QA"
            )
        if not (dataset.get("metadata") or {}).get("review_approval_applied"):
            errors.append("dataset review approval ledger has not been applied")
        rubric_freeze = plan.get("rubric_freeze_record") or {}
        required_rubric_freeze_fields = (
            "content_validity_report",
            "content_validity_report_digest",
            "calibration_exclusion_manifest",
            "calibration_exclusion_manifest_digest",
            "calibration_decision_record",
            "calibration_decision_record_digest",
            "final_rubric_version",
        )
        for field in required_rubric_freeze_fields:
            if not rubric_freeze.get(field):
                errors.append(f"rubric_freeze_record.{field} is required")
        if all(rubric_freeze.get(field) for field in required_rubric_freeze_fields):
            try:
                validity = _read(
                    _resolve(str(rubric_freeze["content_validity_report"]), root)
                )
                if validity.get("rubric_version") != str(artifacts["rubric_version"]):
                    errors.append("content-validity report uses the wrong rubric version")
                expert_count = validity.get("expert_count")
                if not isinstance(expert_count, int) or expert_count not in {2, 3}:
                    errors.append("content-validity report must contain two or three experts")
                if rubric_freeze["content_validity_report_digest"] != stable_digest(
                    validity, length=64
                ):
                    errors.append("content-validity report digest does not match")

                calibration = _read(
                    _resolve(
                        str(rubric_freeze["calibration_exclusion_manifest"]), root
                    )
                )
                if calibration.get("status") != "frozen_calibration_exclusion":
                    errors.append("calibration exclusion manifest is not frozen")
                if calibration.get("must_exclude_from_formal_annotation") is not True:
                    errors.append("calibration responses must be excluded from formal annotation")
                if rubric_freeze[
                    "calibration_exclusion_manifest_digest"
                ] != stable_digest(calibration, length=64):
                    errors.append("calibration exclusion manifest digest does not match")

                decision = _read(
                    _resolve(str(rubric_freeze["calibration_decision_record"]), root)
                )
                if decision.get("status") != "approved_to_freeze":
                    errors.append("calibration decision must be approved_to_freeze")
                if decision.get("rubric_version") != str(artifacts["rubric_version"]):
                    errors.append("calibration decision uses the wrong rubric version")
                if decision.get("independent_rating_complete") is not True:
                    errors.append("expert independent calibration ratings are incomplete")
                if decision.get("amendments_applied") is not True:
                    errors.append("calibration amendments are not recorded as applied")
                if decision.get("unresolved_blocking_issues") is not False:
                    errors.append("calibration decision has unresolved blocking issues")
                participants = decision.get("expert_ids") or []
                if len(set(participants)) not in {2, 3}:
                    errors.append("calibration decision must identify two or three experts")
                elif set(participants) != set(validity.get("expert_ids") or []):
                    errors.append(
                        "calibration decision expert IDs do not match content-validity experts"
                    )
                if rubric_freeze["calibration_decision_record_digest"] != stable_digest(
                    decision, length=64
                ):
                    errors.append("calibration decision record digest does not match")
                if rubric_freeze["final_rubric_version"] != str(
                    artifacts["rubric_version"]
                ):
                    errors.append("final rubric version does not match analysis artifacts")
            except Exception as exc:
                errors.append(f"rubric freeze evidence could not be validated: {exc}")
        freeze = plan.get("freeze_record") or {}
        for field in (
            "frozen_at",
            "git_commit",
            "dataset_digest",
            "config_digest",
            "context_registry_digest",
        ):
            if not freeze.get(field):
                errors.append(f"freeze_record.{field} is required for a frozen plan")
        if all(
            freeze.get(field)
            for field in (
                "frozen_at",
                "git_commit",
                "dataset_digest",
                "config_digest",
                "context_registry_digest",
            )
        ):
            try:
                datetime.fromisoformat(str(freeze["frozen_at"]).replace("Z", "+00:00"))
            except ValueError:
                errors.append("freeze_record.frozen_at must be an ISO-8601 timestamp")
            if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", str(freeze["git_commit"])):
                errors.append("freeze_record.git_commit must be a full Git object ID")
            if freeze["dataset_digest"] != scenario_content_digest(dataset):
                errors.append("freeze_record.dataset_digest does not match the dataset")
            try:
                expected_config_digest = stable_digest(
                    {
                        "main": _read(_resolve(str(artifacts["main_config"]), root)),
                        "robustness": _read(
                            _resolve(str(artifacts["robustness_config"]), root)
                        ),
                        "human_annotation": _read(
                            _resolve(str(artifacts["human_annotation_plan"]), root)
                        ),
                    },
                    length=64,
                )
                if freeze["config_digest"] != expected_config_digest:
                    errors.append("freeze_record.config_digest does not match")
                expected_context_digest = stable_digest(
                    _read(_resolve(str(artifacts["context_registry"]), root)),
                    length=64,
                )
                if freeze["context_registry_digest"] != expected_context_digest:
                    errors.append("freeze_record.context_registry_digest does not match")
            except Exception as exc:
                errors.append(f"freeze digests could not be validated: {exc}")
        approved_by = freeze.get("approved_by")
        if not isinstance(approved_by, list) or not approved_by or any(
            not isinstance(value, str) or not value.strip() for value in approved_by
        ):
            errors.append("freeze_record.approved_by must contain pseudonymous approver IDs")
        elif len(approved_by) != len(set(approved_by)):
            errors.append("freeze_record.approved_by contains duplicate IDs")
        if plan.get("status") != "frozen":
            errors.append("plan status must be frozen")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan", nargs="?", type=Path, default=ROOT / "configs" / "analysis_plan_v2.json"
    )
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    plan = _read(args.plan)
    errors = validate_plan(plan, root=ROOT, require_frozen=args.require_frozen)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Analysis plan is internally consistent: {args.plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
