#!/usr/bin/env python3
"""Validate that every v3 motif is grounded in a documented conspiracy narrative."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = ROOT / "configs" / "motif_narratives_v3.json"
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DATE_RE = re.compile(
    r"^(2023|2024|2025|2026)(?:-(?:0[1-9]|1[0-2])(?:-(?:0[1-9]|[12][0-9]|3[01]))?)?$"
)
REQUIRED_TEXT_FIELDS = (
    "narrative_summary",
    "alleged_actor",
    "alleged_action",
    "alleged_goal",
    "alleged_mechanism",
    "alleged_concealment",
    "authoring_scope",
)
TRUTH_ADJUDICATION_FIELDS = {
    "epistemic_status",
    "fact_check_status",
    "truth_status",
    "verified_premises",
    "unsupported_inference",
    "unknowns",
    "response_boundary",
}
SOURCE_MATCHES = {"direct_complete", "direct_scoped", "direct_close"}
ELIGIBILITY_MATCHES = {"direct_complete", "direct_scoped"}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_url(value: str) -> str:
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return parsed._replace(netloc=host).geturl()


def validate_records(payload: dict, manifest: dict) -> list[str]:
    errors: list[str] = []
    if payload.get("eligibility_gate") != "documented_public_discourse_only":
        errors.append("eligibility_gate must be documented_public_discourse_only")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return errors + ["records must be a non-empty list"]

    manifest_groups = (
        manifest.get("motifs", []),
        manifest.get("additional_eligible_candidates", []),
    )
    manifest_items: dict[str, dict] = {
        str(item["motif_id"]): item
        for group in manifest_groups
        for item in group
        if isinstance(item, dict)
        and isinstance(item.get("motif_id"), str)
        and item["motif_id"]
    }
    expected_ids = {motif_id for motif_id in manifest_items}
    record_ids: list[str] = []
    for index, record in enumerate(records):
        label = f"records[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label} must be an object")
            continue
        motif_id = record.get("motif_id")
        if not isinstance(motif_id, str) or not motif_id.strip():
            errors.append(f"{label}.motif_id must be a non-empty string")
            motif_id = label
        record_ids.append(motif_id)

        forbidden = sorted(TRUTH_ADJUDICATION_FIELDS & set(record))
        if forbidden:
            errors.append(
                f"{motif_id}: narrative records must not adjudicate truth via {forbidden}"
            )
        for field in REQUIRED_TEXT_FIELDS:
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"{motif_id}.{field} must be a non-empty string")

        sources = record.get("circulation_sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{motif_id}.circulation_sources must be a non-empty list")
            continue
        direct_matches = 0
        eligibility_matches = 0
        source_pairs: set[tuple[str, str]] = set()
        source_matches_by_pair: dict[tuple[str, str], str] = {}
        for source_index, source in enumerate(sources):
            source_label = f"{motif_id}.circulation_sources[{source_index}]"
            if not isinstance(source, dict):
                errors.append(f"{source_label} must be an object")
                continue
            for field in ("date", "title", "url", "match"):
                value = source.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{source_label}.{field} must be a non-empty string")
            if not DATE_RE.fullmatch(str(source.get("date", ""))):
                errors.append(f"{source_label}.date must fall in 2023-2026")
            parsed = urlparse(str(source.get("url", "")))
            if parsed.scheme != "https" or not parsed.netloc:
                errors.append(f"{source_label}.url must be an absolute HTTPS URL")
            if source.get("match") not in SOURCE_MATCHES:
                errors.append(f"{source_label}.match is unsupported")
            else:
                direct_matches += 1
                if source.get("match") in ELIGIBILITY_MATCHES:
                    eligibility_matches += 1
            source_pair = (
                str(source.get("date", "")),
                _canonical_url(str(source.get("url", ""))),
            )
            source_pairs.add(source_pair)
            source_matches_by_pair[source_pair] = str(source.get("match", ""))
        if not direct_matches:
            errors.append(f"{motif_id} has no direct circulation match")
        if not eligibility_matches:
            errors.append(
                f"{motif_id} has only close/adjacent sources; at least one source must "
                "document the complete or explicitly scoped narrative"
            )
        manifest_item = manifest_items.get(motif_id, {})
        manifest_anchor = (
            str(manifest_item.get("evidence_date", "")),
            _canonical_url(str(manifest_item.get("evidence_url", ""))),
        )
        if manifest_anchor not in source_pairs:
            errors.append(
                f"{motif_id}: manifest evidence date/URL is absent from circulation_sources"
            )
        elif source_matches_by_pair.get(manifest_anchor) not in ELIGIBILITY_MATCHES:
            errors.append(
                f"{motif_id}: manifest evidence must be a direct_complete or "
                "direct_scoped source, not a merely direct_close source"
            )

    duplicates = sorted(
        motif_id for motif_id, count in Counter(record_ids).items() if count > 1
    )
    if duplicates:
        errors.append(f"duplicate narrative record ids: {duplicates}")
    actual_ids = set(record_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        errors.append(f"manifest candidates without narrative records: {missing}")
    if extra:
        errors.append(f"narrative records absent from candidate manifest: {extra}")

    declared_count = manifest.get("candidate_motif_count")
    if declared_count != len(expected_ids):
        errors.append(
            "candidate_motif_count does not match the union of proposed and "
            f"additional eligible candidates: {declared_count} != {len(expected_ids)}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="?", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    payload = _read_json(args.records)
    manifest = _read_json(args.manifest)
    errors = validate_records(payload, manifest)
    report = {
        "ok": not errors,
        "records": str(args.records),
        "record_count": len(payload.get("records") or []),
        "eligibility_gate": payload.get("eligibility_gate"),
        "truth_adjudication_fields_allowed": False,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
