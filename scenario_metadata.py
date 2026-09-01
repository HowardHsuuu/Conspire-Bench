from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional


DEFAULT_METADATA_PATH = Path("configs/scenario_metadata_v2.json")


def enrich_dataset(
    dataset: dict[str, Any],
    metadata_path: Optional[str | Path] = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    """Apply the versioned metadata overlay without mutating the source dataset."""
    enriched = deepcopy(dataset)
    if metadata_path is None or not Path(metadata_path).exists():
        return enriched
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    profiles = payload.get("profiles_by_subcategory") or {}
    pressure = payload.get("pressure_by_scenario_type") or {}
    overrides = payload.get("scenario_overrides") or {}
    for scenario in enriched.get("scenarios", []):
        profile = profiles.get(scenario.get("subcategory"), {})
        type_profile = pressure.get(scenario.get("type"), {})
        override = overrides.get(scenario.get("id"), {})
        merged = {
            **profile,
            **type_profile,
            **override,
            "scenario_version": str(
                override.get("scenario_version", payload.get("scenario_version", "2.0"))
            ),
            "fact_check_status": override.get(
                "fact_check_status", payload.get("fact_check_status", "requires_source_audit")
            ),
            "review_status": override.get(
                "review_status", payload.get("review_status", "draft_pending_review")
            ),
        }
        for key, value in merged.items():
            scenario.setdefault(key, value)
    enriched.setdefault("metadata", {})["scenario_metadata_overlay"] = {
        "path": str(metadata_path),
        "schema_version": str(payload.get("schema_version", "1.0")),
    }
    return enriched


def annotation_readiness_errors(scenario: dict[str, Any]) -> list[str]:
    errors = []
    if scenario.get("fact_check_status") not in {
        "source_reviewed",
        "verified",
        "not_applicable",
    }:
        errors.append(
            "fact_check_status must be source_reviewed, verified, or not_applicable"
        )
    if scenario.get("review_status") not in {
        "ai_author_reviewed",
        "approved",
        "expert_approved",
    }:
        errors.append(
            "review_status must be ai_author_reviewed, approved, or expert_approved"
        )
    if scenario.get("source_packet_id") and scenario.get("review_status") in {
        "ai_author_reviewed",
        "approved",
        "expert_approved",
    }:
        if not scenario.get("review_approval_id"):
            errors.append("approved expansion scenario must record review_approval_id")
        if not scenario.get("reviewed_at_utc"):
            errors.append("approved expansion scenario must record reviewed_at_utc")
    return errors
