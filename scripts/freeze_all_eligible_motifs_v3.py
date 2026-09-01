#!/usr/bin/env python3
"""Freeze every audited, circulation-grounded v3 motif into the main pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_RECOMMENDATION = ROOT / "configs" / "motif_selection_recommendation_v3.json"

TIER_FIELDS = (
    "recommended_primary_ids",
    "viable_alternate_ids",
    "auxiliary_variant_ids",
    "high_sensitivity_optional_ids",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(
    manifest: dict[str, Any], recommendation: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    tier_by_id = {
        motif_id: tier
        for tier in TIER_FIELDS
        for motif_id in recommendation.get(tier, [])
    }
    candidates = [
        *manifest.get("motifs", []),
        *manifest.get("additional_eligible_candidates", []),
    ]
    ids = [item["motif_id"] for item in candidates]
    if len(ids) != 51 or len(set(ids)) != 51:
        raise ValueError("The audited v3 pool must contain exactly 51 unique motifs")

    frozen_motifs = []
    for source in candidates:
        item = dict(source)
        item.pop("selection_status", None)
        item["authoring_status"] = item.get("authoring_status", "not_authored")
        item["historical_selection_tier"] = tier_by_id.get(
            item["motif_id"], "unrecorded"
        )
        frozen_motifs.append(item)

    frozen_manifest = dict(manifest)
    frozen_manifest.update(
        {
            "selection_state": "frozen_primary",
            "candidate_motif_count": 51,
            "target_motif_count": 51,
            "motifs": frozen_motifs,
            "additional_eligible_candidates": [],
            "selection_decision": {
                "decided_at": "2026-09-01",
                "decision": "include_all_eligible_nonduplicate_motifs",
                "rationale": (
                    "All 51 records pass the same circulation and complete-story gate. "
                    "Overlap and sensitivity remain analysis or authoring metadata rather "
                    "than exclusion criteria."
                ),
            },
        }
    )

    frozen_recommendation = dict(recommendation)
    frozen_recommendation.update(
        {
            "selection_state": "all_eligible_motifs_frozen",
            "target_motif_count": 51,
            "candidate_pool_assessment": "all_eligible_motifs_included",
            "recommendation_policy": [
                "Every included motif passes the documented-public-discourse and complete-story gates.",
                "No truth adjudication or prevalence threshold is used for eligibility.",
                "High-overlap and high-sensitivity labels guide clustered analysis and explicit identity review; they do not remove a motif from the main dataset or authorize proposition-changing fictionalization.",
            ],
            "historical_tiering": {
                field: list(recommendation.get(field, [])) for field in TIER_FIELDS
            },
            "recommended_primary_ids": ids,
            "viable_alternate_ids": [],
            "auxiliary_variant_ids": [],
            "high_sensitivity_optional_ids": [],
            "swap_guidance": {},
        }
    )
    return frozen_manifest, frozen_recommendation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--recommendation", type=Path, default=DEFAULT_RECOMMENDATION)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    manifest = _read(args.manifest)
    recommendation = _read(args.recommendation)
    frozen_manifest, frozen_recommendation = freeze(manifest, recommendation)
    if args.check:
        ok = (
            manifest.get("selection_state") == "frozen_primary"
            and manifest.get("target_motif_count") == 51
            and len(manifest.get("motifs", [])) == 51
            and not manifest.get("additional_eligible_candidates")
            and recommendation.get("selection_state")
            == "all_eligible_motifs_frozen"
            and recommendation.get("recommended_primary_ids")
            == [item["motif_id"] for item in manifest.get("motifs", [])]
        )
        print(json.dumps({"ok": ok, "motif_count": len(manifest.get("motifs", []))}, indent=2))
        return 0 if ok else 1

    args.manifest.write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.recommendation.write_text(
        json.dumps(frozen_recommendation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "motif_count": 51}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
