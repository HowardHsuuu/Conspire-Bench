from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from experiment_conditions import stable_digest


CATALOG_FORMAT = "conspire_expansion_catalog"
PROMPT_FIELDS = {
    "multi_turn_progression": "multi_turn",
    "single_turn_complete_logic": "single_turn",
    "complete_logic_then_resistance": "resistance",
}
SELECTION_STATUSES = {
    "documented_current_narrative",
    "current_risk_analogue",
    "cross_domain_composite",
}
REVIEW_STATE_FIELDS = {
    "fact_check_status",
    "review_status",
    "review_approval_id",
    "reviewed_at_utc",
}


def scenario_content_digest(dataset: dict[str, Any]) -> str:
    """Hash authored scenario content while ignoring mutable approval state."""
    scenarios = [
        {
            key: value
            for key, value in scenario.items()
            if key not in REVIEW_STATE_FIELDS
        }
        for scenario in dataset.get("scenarios", [])
    ]
    return stable_digest(scenarios, length=64)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source_packets(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    packets = payload.get("packets")
    if not isinstance(packets, list) or not packets:
        return ["source packet payload must contain a non-empty 'packets' list"]

    seen: set[str] = set()
    for index, packet in enumerate(packets):
        label = packet.get("id", f"<packet:{index}>") if isinstance(packet, dict) else f"<packet:{index}>"
        if not isinstance(packet, dict):
            errors.append(f"{label}: packet must be an object")
            continue
        packet_id = packet.get("id")
        if not isinstance(packet_id, str) or not packet_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
        elif packet_id in seen:
            errors.append(f"{label}: duplicate packet id")
        else:
            seen.add(packet_id)

        for field in ("verified_premises", "unknowns", "risk_flags"):
            value = packet.get(field)
            if not isinstance(value, list) or not value:
                errors.append(f"{label}: {field} must be a non-empty list")
            elif not all(isinstance(item, str) and item.strip() for item in value):
                errors.append(f"{label}: {field} items must be non-empty strings")
        unsupported = packet.get("unsupported_inference")
        if not isinstance(unsupported, str) or not unsupported.strip():
            errors.append(f"{label}: unsupported_inference must be a non-empty string")

        sources = packet.get("sources")
        if not isinstance(sources, list) or len(sources) < 2:
            errors.append(f"{label}: at least two sources are required")
            continue
        source_hosts: set[str] = set()
        for source_index, source in enumerate(sources):
            source_label = f"{label}.sources[{source_index}]"
            if not isinstance(source, dict):
                errors.append(f"{source_label}: source must be an object")
                continue
            for field in ("title", "url", "kind"):
                if not isinstance(source.get(field), str) or not source[field].strip():
                    errors.append(f"{source_label}: {field} must be a non-empty string")
            parsed = urlparse(str(source.get("url", "")))
            if parsed.scheme != "https" or not parsed.netloc:
                errors.append(f"{source_label}: url must be an absolute HTTPS URL")
            elif parsed.netloc:
                source_hosts.add(parsed.netloc.lower())
        if len(source_hosts) < 2:
            errors.append(f"{label}: sources must include at least two independent hosts")
    return errors


def validate_selection_evidence(
    payload: dict[str, Any],
    motif_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        return ["selection-evidence payload must contain a non-empty 'items' list"]
    seen: set[str] = set()
    for index, item in enumerate(items):
        label = item.get("motif_id", f"<selection:{index}>") if isinstance(item, dict) else f"<selection:{index}>"
        if not isinstance(item, dict):
            errors.append(f"{label}: selection item must be an object")
            continue
        motif_id = item.get("motif_id")
        if not isinstance(motif_id, str) or not motif_id.strip():
            errors.append(f"{label}: motif_id must be a non-empty string")
        elif motif_id in seen:
            errors.append(f"{label}: duplicate motif_id")
        else:
            seen.add(motif_id)
        if item.get("selection_status") not in SELECTION_STATUSES:
            errors.append(f"{label}: unsupported selection_status")
        for field in ("evidence_date", "evidence_url", "rationale"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"{label}: {field} must be a non-empty string")
        parsed = urlparse(str(item.get("evidence_url", "")))
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"{label}: evidence_url must be an absolute HTTPS URL")
    missing = sorted(motif_ids - seen)
    extra = sorted(seen - motif_ids)
    if missing:
        errors.append(f"motifs without selection evidence: {', '.join(missing)}")
    if extra:
        errors.append(f"selection evidence without authored motifs: {', '.join(extra)}")
    return errors


def validate_review_approval(
    catalog: dict[str, Any],
    packet_payload: dict[str, Any],
    approval: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    review_kind = approval.get("review_kind", "independent_human")
    if review_kind == "ai_assisted_author_qa":
        if approval.get("status") != "approved_for_experiment":
            errors.append(
                "AI-assisted author QA status must be approved_for_experiment"
            )
        if approval.get("independent_human_review") is not False:
            errors.append(
                "AI-assisted author QA must explicitly set independent_human_review=false"
            )
        if approval.get("catalog_digest") != stable_digest(catalog, length=64):
            errors.append("scenario QA catalog_digest does not match")
        if approval.get("source_packet_digest") != stable_digest(
            packet_payload, length=64
        ):
            errors.append("scenario QA source_packet_digest does not match")
        motif_ids = {motif.get("id") for motif in catalog.get("motifs", [])}
        reviewed = set(approval.get("reviewed_motif_ids") or [])
        if reviewed != motif_ids:
            errors.append(
                "scenario QA reviewed_motif_ids must exactly match the catalog motifs"
            )
        criteria = approval.get("criteria") or {}
        for field in (
            "fact_boundaries_source_supported",
            "prompts_construct_aligned",
            "matched_controls_valid",
            "no_prompt_asserts_claim_as_fact",
        ):
            if criteria.get(field) is not True:
                errors.append(f"scenario QA criteria.{field} must be true")
        for field in ("approval_id", "reviewed_at_utc", "limitations"):
            if not approval.get(field):
                errors.append(f"scenario QA {field} is required")
        return errors
    if review_kind != "independent_human":
        return [f"unsupported scenario review_kind: {review_kind}"]
    if approval.get("status") != "approved":
        errors.append("scenario review approval status must be approved")
    if approval.get("catalog_digest") != stable_digest(catalog, length=64):
        errors.append("scenario review approval catalog_digest does not match")
    if approval.get("source_packet_digest") != stable_digest(
        packet_payload, length=64
    ):
        errors.append("scenario review approval source_packet_digest does not match")
    motif_ids = {motif.get("id") for motif in catalog.get("motifs", [])}
    reviews = approval.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        return errors + ["scenario review approval must contain reviews"]
    reviewer_ids = {
        str(review.get("reviewer_id"))
        for review in reviews
        if isinstance(review, dict) and review.get("reviewer_id")
    }
    if not reviewer_ids:
        errors.append("scenario review approval must identify at least one reviewer")
    for reviewer_id in reviewer_ids:
        reviewed = {
            review.get("motif_id")
            for review in reviews
            if isinstance(review, dict) and str(review.get("reviewer_id")) == reviewer_id
        }
        missing = sorted(motif_ids - reviewed)
        extra = sorted(reviewed - motif_ids)
        if missing:
            errors.append(f"scenario reviewer {reviewer_id} is missing motifs: {missing}")
        if extra:
            errors.append(f"scenario reviewer {reviewer_id} has unknown motifs: {extra}")
    for index, review in enumerate(reviews):
        label = f"scenario_review[{index}]"
        if not isinstance(review, dict):
            errors.append(f"{label}: review must be an object")
            continue
        if review.get("decision") != "approve":
            errors.append(f"{label}: decision must be approve")
        for field in (
            "independent_of_scenario_authoring",
            "fact_boundary_accurate",
            "prompts_construct_aligned",
            "matched_control_valid",
        ):
            if review.get(field) is not True:
                errors.append(f"{label}: {field} must be true")
    return errors


def _scenario_common(
    motif: dict[str, Any],
    packet: dict[str, Any],
    selection: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, Any]:
    return {
        "category": motif["category"],
        "subcategory": motif["id"],
        "risk_level": motif["risk_level"],
        "belief_domain": motif["belief_domain"],
        "symptom_domain": motif["symptom_domain"],
        "grounding_trap": packet["unsupported_inference"],
        "expected_safe_behavior": motif["expected_safe_behavior"],
        "red_flags": sorted(set(packet["risk_flags"] + motif.get("additional_red_flags", []))),
        "motif_id": motif["id"],
        "source_packet_id": packet["id"],
        "selection_status": selection["selection_status"],
        "selection_evidence_date": selection["evidence_date"],
        "scenario_version": catalog.get("scenario_version", "2.0-expansion-draft"),
        "fact_check_status": catalog.get("fact_check_status", "source_reviewed"),
        "review_status": catalog.get("review_status", "ai_reviewed_pending_human"),
        "review_approval_id": catalog.get("review_approval_id"),
        "reviewed_at_utc": catalog.get("reviewed_at_utc"),
        "is_control": False,
        "control_pair_id": None,
    }


def _make_scenarios(
    motif: dict[str, Any],
    packet: dict[str, Any],
    selection: dict[str, Any],
    catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    common = _scenario_common(motif, packet, selection, catalog)
    prompts = motif["prompts"]
    prefix = f"v2_{motif['id']}"
    pair_id = f"pair_{prefix}_single_001"
    return [
        {
            "id": f"{prefix}_multi_001",
            "type": "multi_turn_progression",
            **common,
            "pressure_type": "progressive_confirmation_and_narrative_expansion",
            "conversation": [
                {"turn": turn, "user": text}
                for turn, text in enumerate(prompts["multi_turn"], start=1)
            ],
        },
        {
            "id": f"{prefix}_single_001",
            "type": "single_turn_complete_logic",
            **common,
            "pressure_type": "compressed_complete_case_and_direct_confirmation",
            "control_pair_id": pair_id,
            "user": prompts["single_turn"],
        },
        {
            "id": f"{prefix}_control_001",
            "type": "single_turn_complete_logic",
            **common,
            "pressure_type": "matched_nonconspiratorial_evidence_evaluation_control",
            "control_pair_id": pair_id,
            "is_control": True,
            "user": prompts["control"],
        },
        {
            "id": f"{prefix}_resist_001",
            "type": "complete_logic_then_resistance",
            **common,
            "pressure_type": "confirmation_followed_by_resistance_to_grounding",
            "conversation": [
                {"turn": turn, "user": text}
                for turn, text in enumerate(prompts["resistance"], start=1)
            ],
        },
    ]


def validate_expansion_catalog(
    catalog: dict[str, Any],
    packet_payload: dict[str, Any],
    selection_payload: dict[str, Any] | None = None,
) -> list[str]:
    errors = validate_source_packets(packet_payload)
    packets = {
        packet.get("id"): packet
        for packet in packet_payload.get("packets", [])
        if isinstance(packet, dict)
    }
    motifs = catalog.get("motifs")
    if not isinstance(motifs, list) or not motifs:
        return errors + ["expansion catalog must contain a non-empty 'motifs' list"]

    seen: set[str] = set()
    for index, motif in enumerate(motifs):
        label = motif.get("id", f"<motif:{index}>") if isinstance(motif, dict) else f"<motif:{index}>"
        if not isinstance(motif, dict):
            errors.append(f"{label}: motif must be an object")
            continue
        motif_id = motif.get("id")
        if not isinstance(motif_id, str) or not motif_id.strip():
            errors.append(f"{label}: id must be a non-empty string")
        elif motif_id in seen:
            errors.append(f"{label}: duplicate motif id")
        else:
            seen.add(motif_id)
        if motif_id not in packets:
            errors.append(f"{label}: no matching source packet")
        for field in ("category", "risk_level", "belief_domain", "symptom_domain"):
            if not isinstance(motif.get(field), str) or not motif[field].strip():
                errors.append(f"{label}: {field} must be a non-empty string")
        expected = motif.get("expected_safe_behavior")
        if not isinstance(expected, list) or not expected:
            errors.append(f"{label}: expected_safe_behavior must be a non-empty list")
        prompts = motif.get("prompts")
        if not isinstance(prompts, dict):
            errors.append(f"{label}: prompts must be an object")
            continue
        multi = prompts.get("multi_turn")
        resistance = prompts.get("resistance")
        single = prompts.get("single_turn")
        control = prompts.get("control")
        if not isinstance(multi, list) or len(multi) != 4:
            errors.append(f"{label}: multi_turn must contain exactly four user turns")
        elif not all(isinstance(item, str) and item.strip() for item in multi):
            errors.append(f"{label}: multi_turn items must be non-empty strings")
        if not isinstance(resistance, list) or len(resistance) != 3:
            errors.append(f"{label}: resistance must contain exactly three user turns")
        elif not all(isinstance(item, str) and item.strip() for item in resistance):
            errors.append(f"{label}: resistance items must be non-empty strings")
        if not isinstance(single, str) or not single.strip():
            errors.append(f"{label}: single_turn must be a non-empty string")
        if not isinstance(control, str) or not control.strip():
            errors.append(f"{label}: control must be a non-empty string")
    extra_packets = sorted(set(packets) - seen)
    if extra_packets:
        errors.append(f"source packets without authored motifs: {', '.join(extra_packets)}")
    if selection_payload is not None:
        errors.extend(validate_selection_evidence(selection_payload, seen))
    return errors


def build_dataset_from_catalog(
    catalog: dict[str, Any],
    *,
    catalog_path: Path,
) -> dict[str, Any]:
    if catalog.get("format") != CATALOG_FORMAT:
        raise ValueError(f"Unsupported catalog format: {catalog.get('format')}")
    base_path = (catalog_path.parent / catalog["base_dataset"]).resolve()
    packet_path = (catalog_path.parent / catalog["source_packets"]).resolve()
    selection_path = (catalog_path.parent / catalog["selection_evidence"]).resolve()
    base = _read_json(base_path)
    packet_payload = _read_json(packet_path)
    selection_payload = _read_json(selection_path)
    errors = validate_expansion_catalog(catalog, packet_payload, selection_payload)
    if errors:
        raise ValueError("Invalid expansion catalog:\n- " + "\n- ".join(errors))

    effective_catalog = deepcopy(catalog)
    approval_path = None
    approval_value = catalog.get("review_approval")
    if approval_value:
        approval_path = (catalog_path.parent / str(approval_value)).resolve()
        if approval_path.exists():
            approval = _read_json(approval_path)
            approval_errors = validate_review_approval(
                catalog, packet_payload, approval
            )
            if approval_errors:
                raise ValueError(
                    "Invalid scenario review approval:\n- "
                    + "\n- ".join(approval_errors)
                )
            if approval.get("review_kind") == "ai_assisted_author_qa":
                effective_catalog["fact_check_status"] = "source_reviewed"
                effective_catalog["review_status"] = "ai_author_reviewed"
            else:
                effective_catalog["fact_check_status"] = "verified"
                effective_catalog["review_status"] = "approved"
            effective_catalog["review_approval_id"] = approval["approval_id"]
            effective_catalog["reviewed_at_utc"] = approval["reviewed_at_utc"]

    packets = {packet["id"]: packet for packet in packet_payload["packets"]}
    selections = {item["motif_id"]: item for item in selection_payload["items"]}
    expanded = deepcopy(base)
    for motif in catalog["motifs"]:
        expanded["scenarios"].extend(
            _make_scenarios(
                motif,
                packets[motif["id"]],
                selections[motif["id"]],
                effective_catalog,
            )
        )

    ids = [scenario["id"] for scenario in expanded["scenarios"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Expanded dataset contains duplicate scenario IDs")

    categories: dict[str, int] = {}
    scenario_types: dict[str, int] = {}
    for scenario in expanded["scenarios"]:
        categories[scenario["category"]] = categories.get(scenario["category"], 0) + 1
        scenario_types[scenario["type"]] = scenario_types.get(scenario["type"], 0) + 1
    expanded["metadata"] = {
        **expanded.get("metadata", {}),
        "dataset_name": "Conspire-Bench-v2-draft",
        "version": catalog.get("dataset_version", "2.0-draft"),
        "total_scenarios": len(expanded["scenarios"]),
        "description": "Legacy 24-scenario seed plus source-reviewed v2 expansion motifs",
        "scenario_types": scenario_types,
        "categories": categories,
        "expansion_catalog": str(catalog_path),
        "source_packets": str(packet_path),
        "selection_evidence": str(selection_path),
        "review_approval": str(approval_path) if approval_path else None,
        "review_approval_applied": bool(
            effective_catalog.get("review_approval_id")
        ),
        "review_approval_kind": (
            approval.get("review_kind", "independent_human")
            if approval_path and approval_path.exists()
            else None
        ),
        "source_review": packet_payload.get("release_policy", {}),
    }
    return expanded


def load_benchmark_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    payload = _read_json(dataset_path)
    if payload.get("format") == CATALOG_FORMAT:
        return build_dataset_from_catalog(payload, catalog_path=dataset_path)
    return payload
