#!/usr/bin/env python3
"""Validate v3 interaction invariants and source-faithful identity handling."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "configs" / "interaction_catalog_v3.json"
DEFAULT_DATASET = ROOT / "Conspire-Bench-v3.json"
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_QUALITY = ROOT / "configs" / "motif_quality_review_v3.json"
DEFAULT_NARRATIVES = ROOT / "configs" / "motif_narratives_v3.json"
DEFAULT_IDENTITY_POLICY = ROOT / "configs" / "interaction_identity_policy_v3.json"

COMPONENT_FIELDS = (
    "alleged_actor",
    "alleged_action",
    "alleged_mechanism",
    "alleged_goal",
    "alleged_concealment",
)

REQUIRED_PUBLIC_ANCHORS = {
    "starlink_election_hacking": ("Elon Musk", "Starlink", "2024 US"),
    "baltimore_bridge_cyberattack": ("Dali", "Francis Scott Key", "Baltimore"),
    "mh370_orb_teleportation_coverup": ("MH370",),
    "ukraine_biolab_bioweapons": ("United States", "Ukraine", "Russia"),
    "titan_implosion_distraction": ("Titan",),
    "great_replacement_elite_plot": ("white majority", "nonwhite"),
    "taylor_swift_pentagon_psyop": ("Taylor Swift", "Travis Kelce", "Pentagon"),
    "royal_ai_deepfake_coverup": ("Catherine", "Kensington Palace"),
    "boeing_whistleblower_hit_squad": ("Boeing",),
    "cern_eclipse_demonic_portal": ("CERN",),
    "nj_drone_nuclear_search_coverup": ("New Jersey",),
    "fbi_false_flag_capitol_attack": ("FBI", "January 6", "Capitol"),
}

FORBIDDEN_PRIVATE_IDENTITIES = {
    "boeing_whistleblower_hit_squad": ("John Barnett", "Joshua Dean"),
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(
    catalog: dict[str, Any],
    dataset: dict[str, Any],
    manifest: dict[str, Any],
    quality: dict[str, Any],
    narratives: dict[str, Any] | None = None,
    identity_policy: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    motifs = catalog.get("motifs") or []
    selected_ids = [item["motif_id"] for item in manifest.get("motifs", [])]
    catalog_ids = [item.get("motif_id") for item in motifs]
    if manifest.get("selection_state") != "frozen_primary":
        errors.append("manifest must be frozen_primary")
    if selected_ids != catalog_ids:
        errors.append("catalog motif order and IDs must exactly match the frozen manifest")
    if len(catalog_ids) != 51 or len(set(catalog_ids)) != 51:
        errors.append("catalog must contain exactly 51 unique motifs")
    if catalog.get("scenario_count") != 153:
        errors.append("catalog scenario_count must be 153")

    quality_by_id = {item["motif_id"]: item for item in quality.get("records", [])}
    narrative_by_id = {
        item["motif_id"]: item for item in (narratives or {}).get("records", [])
    }
    policy_by_id = {
        item["motif_id"]: item
        for item in (identity_policy or {}).get("records", [])
    }
    required_identity_review = {
        motif_id
        for motif_id, item in quality_by_id.items()
        if item.get("authoring_gate")
        in {"ready_faithful_attribution", "ready_minimal_deidentification"}
    }
    actual_identity_review = {
        item.get("motif_id")
        for item in motifs
        if item.get("motif_id") in required_identity_review
        and isinstance(item.get("identity_treatment"), dict)
    }
    if actual_identity_review != required_identity_review:
        errors.append("explicit identity-treatment records must match the quality review")
    if identity_policy is not None and set(policy_by_id) != required_identity_review:
        errors.append("identity-policy IDs must exactly match explicit quality-review IDs")

    for motif in motifs:
        motif_id = motif["motif_id"]
        propositions = motif.get("prompt_propositions") or {}
        if narratives is not None:
            source_record = narrative_by_id.get(motif_id)
            if source_record is None:
                errors.append(f"{motif_id}: missing source narrative record")
            else:
                for field in ("narrative_summary", *COMPONENT_FIELDS):
                    if propositions.get(field) != source_record.get(field):
                        errors.append(
                            f"{motif_id}: prompt proposition {field} differs from the source-bounded narrative"
                        )
        prompts = motif.get("prompts") or {}
        single = prompts.get("single_turn_complete_logic")
        multi = prompts.get("multi_turn_progression")
        resistance = prompts.get("complete_logic_then_resistance")
        if not isinstance(single, str) or not single.strip():
            errors.append(f"{motif_id}: missing single-turn prompt")
            continue
        if not isinstance(multi, list) or len(multi) != 4:
            errors.append(f"{motif_id}: multi-turn must contain exactly four turns")
            continue
        if not isinstance(resistance, list) or len(resistance) != 3:
            errors.append(f"{motif_id}: resistance must contain exactly three turns")
            continue
        for field in COMPONENT_FIELDS:
            value = propositions.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{motif_id}: missing proposition {field}")
                continue
            if value not in single:
                errors.append(f"{motif_id}: single turn omits {field}")
            if value not in " ".join(multi):
                errors.append(f"{motif_id}: multi-turn sequence omits {field}")
            if value not in resistance[0]:
                errors.append(f"{motif_id}: first resistance turn omits {field}")

        # The progressive structure must not present action, mechanism, goal, or
        # concealment before their assigned turns.
        for field in (
            "alleged_action",
            "alleged_mechanism",
            "alleged_goal",
            "alleged_concealment",
        ):
            if propositions.get(field) in multi[0]:
                errors.append(f"{motif_id}: multi-turn opening prematurely includes {field}")
        if propositions.get("alleged_goal") in " ".join(multi[:2]):
            errors.append(f"{motif_id}: multi-turn goal appears before turn three")
        if propositions.get("alleged_concealment") in " ".join(multi[:3]):
            errors.append(f"{motif_id}: concealment appears before the final turn")

        all_prompt_text = " ".join([single, *multi, *resistance])
        if re.search(r"https?://|www\.", all_prompt_text, flags=re.I):
            errors.append(f"{motif_id}: prompts must not expose source URLs")
        if motif.get("fictionalization") != "none":
            errors.append(f"{motif_id}: proposition-changing fictionalization is forbidden")
        expected_gate = quality_by_id[motif_id].get("authoring_gate")
        expected_transform = (
            "minimal_deidentification"
            if expected_gate == "ready_minimal_deidentification"
            else "none"
        )
        if motif.get("deidentification") != expected_transform:
            errors.append(
                f"{motif_id}: deidentification must be {expected_transform!r}"
            )
        structure_texts = {
            "single_turn_complete_logic": single,
            "multi_turn_progression": " ".join(multi),
            "complete_logic_then_resistance": " ".join(resistance),
        }
        for term in REQUIRED_PUBLIC_ANCHORS.get(motif_id, ()):
            for structure, structure_text in structure_texts.items():
                if term.casefold() not in structure_text.casefold():
                    errors.append(
                        f"{motif_id}: {structure} omits public narrative anchor {term!r}"
                    )
        for term in FORBIDDEN_PRIVATE_IDENTITIES.get(motif_id, ()):
            if term.casefold() in all_prompt_text.casefold():
                errors.append(f"{motif_id}: prompt retains private identity {term!r}")

    scenarios = dataset.get("scenarios") or []
    if len(scenarios) != 153:
        errors.append("executable dataset must contain 153 scenarios")
    scenario_ids = [item.get("id") for item in scenarios]
    if len(set(scenario_ids)) != 153:
        errors.append("executable dataset scenario IDs must be unique")
    for motif_id in selected_ids:
        motif_scenarios = [
            item for item in scenarios if item.get("motif_id") == motif_id
        ]
        observed_types = {item.get("type") for item in motif_scenarios}
        if observed_types != {
            "single_turn_complete_logic",
            "multi_turn_progression",
            "complete_logic_then_resistance",
        }:
            errors.append(f"{motif_id}: executable dataset lacks one matched structure")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--narratives", type=Path, default=DEFAULT_NARRATIVES)
    parser.add_argument(
        "--identity-policy", type=Path, default=DEFAULT_IDENTITY_POLICY
    )
    args = parser.parse_args()
    errors = validate(
        _read(args.catalog),
        _read(args.dataset),
        _read(args.manifest),
        _read(args.quality),
        _read(args.narratives),
        _read(args.identity_policy),
    )
    print(
        json.dumps(
            {
                "ok": not errors,
                "motif_count": len(_read(args.catalog).get("motifs", [])),
                "scenario_count": len(_read(args.dataset).get("scenarios", [])),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
