#!/usr/bin/env python3
"""Validate that the v3 recommendation tiers partition the audited candidate pool."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOMMENDATION = ROOT / "configs" / "motif_selection_recommendation_v3.json"
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_QUALITY = ROOT / "configs" / "motif_quality_review_v3.json"
TIER_FIELDS = (
    "recommended_primary_ids",
    "viable_alternate_ids",
    "auxiliary_variant_ids",
    "high_sensitivity_optional_ids",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_ids(manifest: dict) -> set[str]:
    return {
        item["motif_id"]
        for key in ("motifs", "additional_eligible_candidates")
        for item in manifest.get(key, [])
        if isinstance(item, dict) and isinstance(item.get("motif_id"), str)
    }


def validate_recommendation(
    recommendation: dict, manifest: dict, quality: dict
) -> list[str]:
    errors: list[str] = []
    if recommendation.get("eligibility_gate") != "documented_public_discourse_only":
        errors.append("eligibility_gate must be documented_public_discourse_only")
    assessment = recommendation.get("candidate_pool_assessment")
    if assessment not in {
        "coverage_sufficient_for_primary_selection",
        "all_eligible_motifs_included",
    }:
        errors.append(
            "candidate_pool_assessment must be coverage_sufficient_for_primary_selection"
        )
    if not isinstance(recommendation.get("expansion_stop_rule"), str) or not (
        recommendation["expansion_stop_rule"].strip()
    ):
        errors.append("expansion_stop_rule must be a non-empty string")
    expected_ids = _candidate_ids(manifest)
    quality_by_id = {
        item.get("motif_id"): item
        for item in quality.get("records", [])
        if isinstance(item, dict) and item.get("motif_id")
    }
    tier_values: dict[str, list[str]] = {}
    for field in TIER_FIELDS:
        values = recommendation.get(field)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            errors.append(f"{field} must be a list of motif IDs")
            values = []
        if len(values) != len(set(values)):
            errors.append(f"{field} contains duplicate motif IDs")
        tier_values[field] = values

    primary = tier_values["recommended_primary_ids"]
    target = recommendation.get("target_motif_count")
    if not isinstance(target, int) or target <= 0:
        errors.append("target_motif_count must be a positive integer")
    elif len(primary) != target:
        errors.append(f"recommended primary count must equal target: {len(primary)} != {target}")

    all_tier_ids = [motif_id for field in TIER_FIELDS for motif_id in tier_values[field]]
    duplicates = sorted(key for key, count in Counter(all_tier_ids).items() if count > 1)
    if duplicates:
        errors.append(f"motif IDs appear in multiple recommendation tiers: {duplicates}")
    actual_ids = set(all_tier_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        errors.append(f"candidate motifs absent from recommendation tiers: {missing}")
    if extra:
        errors.append(f"recommendation IDs absent from candidate manifest: {extra}")

    high_overlap_ids = {
        motif_id
        for motif_id, record in quality_by_id.items()
        if record.get("distinctness") == "high_overlap"
    }
    auxiliary = set(tier_values["auxiliary_variant_ids"])
    frozen_all = recommendation.get("selection_state") == "all_eligible_motifs_frozen"
    if frozen_all:
        if set(primary) != expected_ids:
            errors.append("frozen all-eligible primary IDs must equal the audited pool")
        if any(tier_values[field] for field in TIER_FIELDS[1:]):
            errors.append("frozen all-eligible design must not retain exclusion tiers")
    else:
        if auxiliary != high_overlap_ids:
            errors.append(
                "auxiliary_variant_ids must exactly match high-overlap quality records: "
                f"expected {sorted(high_overlap_ids)}, found {sorted(auxiliary)}"
            )
        if set(primary) & high_overlap_ids:
            errors.append("recommended primary IDs must not include high-overlap variants")

    swap_guidance = recommendation.get("swap_guidance")
    alternates = set(tier_values["viable_alternate_ids"])
    if not isinstance(swap_guidance, dict):
        errors.append("swap_guidance must be an object")
    else:
        if set(swap_guidance) != alternates:
            errors.append("swap_guidance keys must exactly match viable_alternate_ids")
        for motif_id, note in swap_guidance.items():
            if not isinstance(note, str) or not note.strip():
                errors.append(f"swap_guidance.{motif_id} must be a non-empty string")

    if set(quality_by_id) != expected_ids:
        errors.append("quality review and manifest IDs must match before selection")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recommendation", nargs="?", type=Path, default=DEFAULT_RECOMMENDATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    args = parser.parse_args()
    recommendation = _read_json(args.recommendation)
    manifest = _read_json(args.manifest)
    quality = _read_json(args.quality)
    errors = validate_recommendation(recommendation, manifest, quality)
    report = {
        "ok": not errors,
        "recommendation": str(args.recommendation),
        "recommended_primary_count": len(recommendation.get("recommended_primary_ids") or []),
        "candidate_partition_count": sum(
            len(recommendation.get(field) or []) for field in TIER_FIELDS
        ),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
