#!/usr/bin/env python3
"""Validate the executable v3 design and its coequal outcome contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation import validate_dataset
from experiment_conditions import load_context_set
from rubric_v2 import RUBRIC_DIMENSIONS, RUBRIC_VERSION
from scripts.validate_context_variants_v3 import FRAME_FAMILIES
from scripts.validate_interaction_catalog_v3 import validate as validate_interactions

DEFAULT_PLAN = ROOT / "configs" / "analysis_plan_v3.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate(plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if plan.get("design_version") != "v3":
        errors.append("design_version must be v3")
    artifacts = plan.get("artifacts") or {}
    required_artifacts = (
        "motif_manifest",
        "interaction_catalog",
        "dataset",
        "context_registry",
        "canonical_context_set",
        "full_context_set",
        "local_config",
        "api_config",
        "human_annotation_plan",
        "rubric_version",
    )
    for field in required_artifacts:
        if not artifacts.get(field):
            errors.append(f"artifacts.{field} is required")
    if errors:
        return errors

    manifest = _read(_resolve(artifacts["motif_manifest"]))
    catalog = _read(_resolve(artifacts["interaction_catalog"]))
    dataset = _read(_resolve(artifacts["dataset"]))
    quality = _read(ROOT / "configs" / "motif_quality_review_v3.json")
    narratives = _read(ROOT / "configs" / "motif_narratives_v3.json")
    identity_policy = _read(ROOT / "configs" / "interaction_identity_policy_v3.json")
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

    registry = _resolve(artifacts["context_registry"])
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
        errors.append("estimand grid must contain all four frame-minus-neutral contrasts")
    if grid.get("outcomes") != list(RUBRIC_DIMENSIONS):
        errors.append("estimand grid must contain all seven outcomes")
    if grid.get("status") != "coequal" or grid.get(
        "total_frame_by_outcome_estimands"
    ) != 28:
        errors.append("estimand grid must declare 28 coequal estimands")
    if (plan.get("wording_aggregation") or {}).get(
        "canonical_variant_privilege"
    ) is not False:
        errors.append("canonical wording must not be privileged in the full analysis")

    for field in ("local_config", "api_config", "human_annotation_plan"):
        if not _resolve(artifacts[field]).exists():
            errors.append(f"missing artifact: {artifacts[field]}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", nargs="?", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    plan = _read(args.plan)
    errors = validate(plan)
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
