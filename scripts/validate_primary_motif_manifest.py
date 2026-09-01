#!/usr/bin/env python3
"""Validate the v3 primary-motif selection boundary before prompt authoring."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_CATALOG = ROOT / "configs" / "scenario_expansion_v2.json"
DATE_RE = re.compile(r"^(2023|2024|2025|2026)(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12][0-9]|3[01]))?)?$")
AUTHORING_STATES = {"authored_revision_required", "not_authored", "authored_validated"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(payload: dict, current_catalog: dict | None = None) -> list[str]:
    errors: list[str] = []
    policy = payload.get("selection_policy") or {}
    if policy.get("eligibility_gate") != "documented_public_discourse_only":
        errors.append(
            "selection_policy.eligibility_gate must be documented_public_discourse_only"
        )
    if policy.get("public_discourse_required") is not True:
        errors.append("selection_policy.public_discourse_required must be true")
    if policy.get("truth_resolution_required") is not False:
        errors.append("selection_policy.truth_resolution_required must be false")
    if policy.get("population_prevalence_required") is not False:
        errors.append("selection_policy.population_prevalence_required must be false")

    motifs = payload.get("motifs")
    if not isinstance(motifs, list):
        return errors + ["motifs must be a list"]
    target = payload.get("target_motif_count")
    if not isinstance(target, int) or target <= 0:
        errors.append("target_motif_count must be a positive integer")
    elif len(motifs) != target:
        errors.append(f"expected {target} selected motifs, found {len(motifs)}")

    motif_ids: list[str] = []
    signatures: list[str] = []
    categories: list[str] = []
    for index, motif in enumerate(motifs):
        label = f"motifs[{index}]"
        if not isinstance(motif, dict):
            errors.append(f"{label} must be an object")
            continue
        motif_id = motif.get("motif_id")
        if not isinstance(motif_id, str) or not motif_id.strip():
            errors.append(f"{label}.motif_id must be a non-empty string")
            motif_id = label
        motif_ids.append(motif_id)
        for field in ("display_name", "category", "origin", "reasoning_signature"):
            if not isinstance(motif.get(field), str) or not motif[field].strip():
                errors.append(f"{motif_id}.{field} must be a non-empty string")
        if motif.get("public_discourse_status") != "documented_direct":
            errors.append(f"{motif_id}.public_discourse_status must be documented_direct")
        if motif.get("authoring_status") not in AUTHORING_STATES:
            errors.append(f"{motif_id}.authoring_status is unsupported")
        evidence_date = motif.get("evidence_date")
        if not isinstance(evidence_date, str) or not DATE_RE.fullmatch(evidence_date):
            errors.append(f"{motif_id}.evidence_date must fall in 2023-2026")
        evidence_url = motif.get("evidence_url")
        parsed = urlparse(evidence_url if isinstance(evidence_url, str) else "")
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"{motif_id}.evidence_url must be an absolute HTTPS URL")
        if motif.get("origin") == "replacement_v3" and not motif.get("replaces"):
            errors.append(f"{motif_id}: replacement_v3 entries require replaces")
        if isinstance(motif.get("reasoning_signature"), str):
            signatures.append(motif["reasoning_signature"])
        if isinstance(motif.get("category"), str):
            categories.append(motif["category"])

    duplicate_ids = sorted(key for key, count in Counter(motif_ids).items() if count > 1)
    duplicate_signatures = sorted(key for key, count in Counter(signatures).items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate motif ids: {duplicate_ids}")
    if duplicate_signatures:
        errors.append(f"duplicate reasoning signatures: {duplicate_signatures}")
    if len(set(categories)) < 7:
        errors.append(f"expected at least 7 substantive categories, found {len(set(categories))}")
    if "uap_crash_retrieval_coverup" not in motif_ids:
        errors.append("documented UAP crash-retrieval conspiracy motif must remain represented")

    additional = payload.get("additional_eligible_candidates")
    if not isinstance(additional, list):
        errors.append("additional_eligible_candidates must be a list")
        additional = []
    additional_ids: list[str] = []
    additional_signatures: list[str] = []
    for index, motif in enumerate(additional):
        label = f"additional_eligible_candidates[{index}]"
        if not isinstance(motif, dict):
            errors.append(f"{label} must be an object")
            continue
        motif_id = motif.get("motif_id")
        if not isinstance(motif_id, str) or not motif_id.strip():
            errors.append(f"{label}.motif_id must be a non-empty string")
            motif_id = label
        additional_ids.append(motif_id)
        for field in (
            "display_name",
            "category",
            "origin",
            "reasoning_signature",
            "design_note",
        ):
            if not isinstance(motif.get(field), str) or not motif[field].strip():
                errors.append(f"{motif_id}.{field} must be a non-empty string")
        if motif.get("public_discourse_status") != "documented_direct":
            errors.append(f"{motif_id}.public_discourse_status must be documented_direct")
        if motif.get("selection_status") != "eligible_pending_user_selection":
            errors.append(
                f"{motif_id}.selection_status must be eligible_pending_user_selection"
            )
        evidence_date = motif.get("evidence_date")
        if not isinstance(evidence_date, str) or not DATE_RE.fullmatch(evidence_date):
            errors.append(f"{motif_id}.evidence_date must fall in 2023-2026")
        evidence_url = motif.get("evidence_url")
        parsed = urlparse(evidence_url if isinstance(evidence_url, str) else "")
        if parsed.scheme != "https" or not parsed.netloc:
            errors.append(f"{motif_id}.evidence_url must be an absolute HTTPS URL")
        if isinstance(motif.get("reasoning_signature"), str):
            additional_signatures.append(motif["reasoning_signature"])

    all_candidate_ids = motif_ids + additional_ids
    duplicate_candidate_ids = sorted(
        key for key, count in Counter(all_candidate_ids).items() if count > 1
    )
    if duplicate_candidate_ids:
        errors.append(f"duplicate candidate ids: {duplicate_candidate_ids}")
    all_signatures = signatures + additional_signatures
    duplicate_candidate_signatures = sorted(
        key for key, count in Counter(all_signatures).items() if count > 1
    )
    if duplicate_candidate_signatures:
        errors.append(
            f"duplicate candidate reasoning signatures: {duplicate_candidate_signatures}"
        )
    declared_candidate_count = payload.get("candidate_motif_count")
    if declared_candidate_count != len(all_candidate_ids):
        errors.append(
            "candidate_motif_count must equal proposed plus additional candidates: "
            f"{declared_candidate_count} != {len(all_candidate_ids)}"
        )

    removed = payload.get("removed_v2_motifs")
    if not isinstance(removed, list) or not removed:
        errors.append("removed_v2_motifs must be a non-empty decision ledger")
        removed_ids: set[str] = set()
    else:
        removed_ids = {
            item.get("motif_id") for item in removed if isinstance(item, dict) and item.get("motif_id")
        }
        for item in removed:
            if not isinstance(item, dict) or not item.get("motif_id") or not item.get("reason"):
                errors.append("every removed_v2_motifs entry requires motif_id and reason")
    overlap = sorted(set(motif_ids) & removed_ids)
    if overlap:
        errors.append(f"selected and removed motif overlap: {overlap}")

    semantic_exclusions = payload.get("semantic_qc_exclusions")
    if not isinstance(semantic_exclusions, list) or not semantic_exclusions:
        errors.append("semantic_qc_exclusions must be a non-empty decision ledger")
        excluded_ids: list[str] = []
    else:
        excluded_ids = []
        for index, item in enumerate(semantic_exclusions):
            label = f"semantic_qc_exclusions[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{label} must be an object")
                continue
            motif_id = item.get("motif_id")
            if not isinstance(motif_id, str) or not motif_id.strip():
                errors.append(f"{label}.motif_id must be a non-empty string")
                continue
            excluded_ids.append(motif_id)
            if item.get("reason_code") != "author_composite_no_complete_source":
                errors.append(
                    f"{motif_id}.reason_code must be author_composite_no_complete_source"
                )
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                errors.append(f"{motif_id}.reason must be a non-empty string")
            component_sources = item.get("component_sources")
            if not isinstance(component_sources, list) or len(component_sources) < 2:
                errors.append(f"{motif_id}.component_sources must contain at least two sources")
                continue
            for source_index, source in enumerate(component_sources):
                source_label = f"{motif_id}.component_sources[{source_index}]"
                if not isinstance(source, dict):
                    errors.append(f"{source_label} must be an object")
                    continue
                date = source.get("date")
                if not isinstance(date, str) or not DATE_RE.fullmatch(date):
                    errors.append(f"{source_label}.date must fall in 2023-2026")
                parsed = urlparse(source.get("url") if isinstance(source.get("url"), str) else "")
                if parsed.scheme != "https" or not parsed.netloc:
                    errors.append(f"{source_label}.url must be an absolute HTTPS URL")
                if not isinstance(source.get("supports"), str) or not source["supports"].strip():
                    errors.append(f"{source_label}.supports must be a non-empty string")
        duplicate_exclusions = sorted(
            key for key, count in Counter(excluded_ids).items() if count > 1
        )
        if duplicate_exclusions:
            errors.append(f"duplicate semantic QC exclusions: {duplicate_exclusions}")
        retained_exclusions = sorted(set(all_candidate_ids) & set(excluded_ids))
        if retained_exclusions:
            errors.append(
                "semantic QC exclusions must not remain in the candidate pool: "
                f"{retained_exclusions}"
            )

    if current_catalog is not None:
        current_ids = {
            motif.get("id")
            for motif in current_catalog.get("motifs", [])
            if isinstance(motif, dict) and motif.get("id")
        }
        retained_ids = {
            motif["motif_id"] for motif in motifs if motif.get("origin") == "retained_v2"
        }
        expected_removed = current_ids - retained_ids
        if removed_ids != expected_removed:
            errors.append(
                "removed_v2_motifs must exactly equal current catalog minus retained_v2 motifs: "
                f"expected {sorted(expected_removed)}, found {sorted(removed_ids)}"
            )
        missing_retained = sorted(retained_ids - current_ids)
        if missing_retained:
            errors.append(f"retained_v2 motifs absent from current catalog: {missing_retained}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args()
    payload = _read_json(args.manifest)
    current_catalog = _read_json(args.catalog) if args.catalog else None
    errors = validate_manifest(payload, current_catalog=current_catalog)
    report = {
        "ok": not errors,
        "manifest": str(args.manifest),
        "selected_motif_count": len(payload.get("motifs") or []),
        "candidate_motif_count": len(payload.get("motifs") or [])
        + len(payload.get("additional_eligible_candidates") or []),
        "category_count": len({item.get("category") for item in payload.get("motifs") or []}),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
