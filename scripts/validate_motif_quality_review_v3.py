#!/usr/bin/env python3
"""Validate the semantic quality-review layer for all v3 motif candidates."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW = ROOT / "configs" / "motif_quality_review_v3.json"
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_NARRATIVES = ROOT / "configs" / "motif_narratives_v3.json"

CIRCULATION_SUPPORT = {"direct_single_source", "direct_multi_source"}
NARRATIVE_FIDELITY = {"source_exact", "source_scoped"}
DISTINCTNESS = {"standalone", "partial_overlap", "high_overlap"}
SENSITIVITY = {
    "standard",
    "health",
    "political_or_geopolitical",
    "bereavement_or_public_figure",
    "child_abuse_extremism",
    "racial_extremism",
    "religious_or_occult_extremism",
}
AUTHORING_GATES = {
    "ready_with_scope_controls",
    "ready_faithful_attribution",
    "ready_minimal_deidentification",
}
COMPLETE_STORY_STATUS = "pass_complete_source_bounded"
ELIGIBILITY_MATCHES = {"direct_complete", "direct_scoped"}
MANDATORY_EXPLICIT_IDENTITY_REVIEW = {
    "bereavement_or_public_figure",
    "child_abuse_extremism",
    "racial_extremism",
    "religious_or_occult_extremism",
}
FORBIDDEN_TRUTH_FIELDS = {
    "epistemic_status",
    "fact_check_status",
    "truth_status",
    "verified_premises",
    "unsupported_inference",
    "unknowns",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_ids(manifest: dict) -> set[str]:
    return {
        item["motif_id"]
        for key in ("motifs", "additional_eligible_candidates")
        for item in manifest.get(key, [])
        if isinstance(item, dict) and isinstance(item.get("motif_id"), str)
    }


def validate_review(review: dict, manifest: dict, narratives: dict) -> list[str]:
    errors: list[str] = []
    if review.get("content_eligibility_gate") != "documented_public_discourse_only":
        errors.append("content_eligibility_gate must be documented_public_discourse_only")

    expected_ids = _candidate_ids(manifest)
    narrative_sources = {
        item.get("motif_id"): item.get("circulation_sources", [])
        for item in narratives.get("records", [])
        if isinstance(item, dict)
    }
    records = review.get("records")
    if not isinstance(records, list) or not records:
        return errors + ["records must be a non-empty list"]

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

        forbidden = sorted(FORBIDDEN_TRUTH_FIELDS & set(record))
        if forbidden:
            errors.append(f"{motif_id}: truth-adjudication fields are forbidden: {forbidden}")
        if record.get("content_eligibility") != "pass":
            errors.append(f"{motif_id}.content_eligibility must be pass")
        if record.get("circulation_support") not in CIRCULATION_SUPPORT:
            errors.append(f"{motif_id}.circulation_support is unsupported")
        if record.get("narrative_fidelity") not in NARRATIVE_FIDELITY:
            errors.append(f"{motif_id}.narrative_fidelity is unsupported")
        if record.get("distinctness") not in DISTINCTNESS:
            errors.append(f"{motif_id}.distinctness is unsupported")
        if record.get("sensitivity") not in SENSITIVITY:
            errors.append(f"{motif_id}.sensitivity is unsupported")
        if record.get("authoring_gate") not in AUTHORING_GATES:
            errors.append(f"{motif_id}.authoring_gate is unsupported")
        if record.get("complete_story_status") != COMPLETE_STORY_STATUS:
            errors.append(
                f"{motif_id}.complete_story_status must be {COMPLETE_STORY_STATUS}"
            )
        complete_anchor_url = record.get("complete_story_anchor_url")
        if not isinstance(complete_anchor_url, str) or not complete_anchor_url.strip():
            errors.append(f"{motif_id}.complete_story_anchor_url must be a non-empty string")
        if not isinstance(record.get("review_note"), str) or not record["review_note"].strip():
            errors.append(f"{motif_id}.review_note must be a non-empty string")

        overlap = record.get("overlap_with")
        if not isinstance(overlap, list) or not all(isinstance(value, str) for value in overlap):
            errors.append(f"{motif_id}.overlap_with must be a list of motif IDs")
            overlap = []
        invalid_overlap = sorted(set(overlap) - expected_ids)
        if invalid_overlap:
            errors.append(f"{motif_id}.overlap_with has unknown IDs: {invalid_overlap}")
        if motif_id in overlap:
            errors.append(f"{motif_id}.overlap_with must not contain itself")
        if len(overlap) != len(set(overlap)):
            errors.append(f"{motif_id}.overlap_with contains duplicates")
        if record.get("distinctness") == "high_overlap" and not overlap:
            errors.append(f"{motif_id}: high_overlap requires at least one overlap target")

        source_count = len(narrative_sources.get(motif_id, []))
        expected_support = "direct_multi_source" if source_count > 1 else "direct_single_source"
        if record.get("circulation_support") != expected_support:
            errors.append(
                f"{motif_id}.circulation_support must reflect {source_count} narrative sources"
            )
        eligible_anchor_urls = {
            source.get("url")
            for source in narrative_sources.get(motif_id, [])
            if isinstance(source, dict) and source.get("match") in ELIGIBILITY_MATCHES
        }
        if complete_anchor_url not in eligible_anchor_urls:
            errors.append(
                f"{motif_id}.complete_story_anchor_url must identify a direct_complete "
                "or direct_scoped narrative source"
            )
        if (
            record.get("sensitivity") in MANDATORY_EXPLICIT_IDENTITY_REVIEW
            and record.get("authoring_gate")
            not in {"ready_faithful_attribution", "ready_minimal_deidentification"}
        ):
            errors.append(f"{motif_id}: sensitivity requires explicit identity review")

    duplicates = sorted(key for key, count in Counter(record_ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate quality-review IDs: {duplicates}")
    actual_ids = set(record_ids)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing:
        errors.append(f"candidate motifs without quality review: {missing}")
    if extra:
        errors.append(f"quality-review motifs absent from manifest: {extra}")
    if set(narrative_sources) != expected_ids:
        errors.append("narrative records and manifest candidate IDs must match before quality review")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", nargs="?", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--narratives", type=Path, default=DEFAULT_NARRATIVES)
    args = parser.parse_args()
    review = _read_json(args.review)
    manifest = _read_json(args.manifest)
    narratives = _read_json(args.narratives)
    errors = validate_review(review, manifest, narratives)
    report = {
        "ok": not errors,
        "review": str(args.review),
        "reviewed_motif_count": len(review.get("records") or []),
        "high_overlap_count": sum(
            item.get("distinctness") == "high_overlap" for item in review.get("records") or []
        ),
        "explicit_identity_review_count": sum(
            item.get("authoring_gate")
            in {"ready_faithful_attribution", "ready_minimal_deidentification"}
            for item in review.get("records") or []
        ),
        "minimal_deidentification_count": sum(
            item.get("authoring_gate") == "ready_minimal_deidentification"
            for item in review.get("records") or []
        ),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
