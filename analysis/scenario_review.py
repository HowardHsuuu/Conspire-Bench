#!/usr/bin/env python3
"""Export and import independent human QA for the frozen V3 motif catalog."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_conditions import stable_digest

DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_NARRATIVES = ROOT / "configs" / "motif_narratives_v3.json"
DEFAULT_QUALITY = ROOT / "configs" / "motif_quality_review_v3.json"
DEFAULT_CATALOG = ROOT / "configs" / "interaction_catalog_v3.json"
DEFAULT_IDENTITY = ROOT / "configs" / "interaction_identity_policy_v3.json"

BOOLEAN_FIELDS = (
    "independent_of_scenario_authoring",
    "circulation_match_accurate",
    "narrative_fidelity_accurate",
    "prompts_construct_aligned",
    "interaction_structures_matched",
    "identity_treatment_appropriate",
)
REVIEW_FIELDS = (
    "reviewer_id",
    "expertise_description_nonidentifying",
    *BOOLEAN_FIELDS,
    "prompt_plausibility_1_5",
    "difficulty_1_5",
    "decision",
    "required_changes",
    "comments",
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def load_artifacts(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    narratives_path: Path = DEFAULT_NARRATIVES,
    quality_path: Path = DEFAULT_QUALITY,
    catalog_path: Path = DEFAULT_CATALOG,
    identity_path: Path = DEFAULT_IDENTITY,
) -> dict[str, dict[str, Any]]:
    artifacts = {
        "manifest": _read_object(manifest_path),
        "narratives": _read_object(narratives_path),
        "quality": _read_object(quality_path),
        "catalog": _read_object(catalog_path),
        "identity": _read_object(identity_path),
    }
    id_sets = {
        "manifest": {
            str(item.get("motif_id"))
            for item in artifacts["manifest"].get("motifs", [])
        },
        "narratives": {
            str(item.get("motif_id"))
            for item in artifacts["narratives"].get("records", [])
        },
        "quality": {
            str(item.get("motif_id"))
            for item in artifacts["quality"].get("records", [])
        },
        "catalog": {
            str(item.get("motif_id")) for item in artifacts["catalog"].get("motifs", [])
        },
        "identity": {
            str(item.get("motif_id"))
            for item in artifacts["identity"].get("records", [])
        },
    }
    if not id_sets["manifest"] or len(id_sets["manifest"]) != 51:
        raise ValueError("V3 scenario review requires the frozen 51-motif manifest")
    for name in ("narratives", "quality", "catalog"):
        if id_sets[name] != id_sets["manifest"]:
            raise ValueError(
                f"V3 scenario-review {name} does not match the frozen motif manifest"
            )
    required_identity_ids = {
        str(item["motif_id"])
        for item in artifacts["quality"].get("records", [])
        if item.get("authoring_gate")
        in {"ready_faithful_attribution", "ready_minimal_deidentification"}
    }
    if id_sets["identity"] != required_identity_ids:
        raise ValueError(
            "V3 scenario-review identity policies do not match the sensitive-motif "
            "review boundary"
        )
    return artifacts


def review_items(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    manifest = artifacts["manifest"]
    by_narrative = {
        item["motif_id"]: item for item in artifacts["narratives"]["records"]
    }
    by_quality = {item["motif_id"]: item for item in artifacts["quality"]["records"]}
    by_catalog = {item["motif_id"]: item for item in artifacts["catalog"]["motifs"]}
    by_identity = {item["motif_id"]: item for item in artifacts["identity"]["records"]}
    items: list[dict[str, Any]] = []
    for motif in manifest["motifs"]:
        motif_id = str(motif["motif_id"])
        narrative = by_narrative[motif_id]
        quality = by_quality[motif_id]
        catalog = by_catalog[motif_id]
        identity = by_identity.get(
            motif_id,
            {
                **catalog["identity_treatment"],
                "rationale": (
                    "The catalog applies the public-narrative default; this motif "
                    "does not require an additional sensitive-identity exception."
                ),
            },
        )
        items.append(
            {
                "review_item_id": f"v3_scenario_review::{motif_id}",
                "motif_id": motif_id,
                "display_name": motif["display_name"],
                "category": motif["category"],
                "circulation_evidence": {
                    "manifest_date": motif["evidence_date"],
                    "manifest_url": motif["evidence_url"],
                    "sources": narrative["circulation_sources"],
                    "complete_story_anchor_url": quality["complete_story_anchor_url"],
                },
                "narrative_propositions": {
                    key: narrative[key]
                    for key in (
                        "narrative_summary",
                        "alleged_actor",
                        "alleged_action",
                        "alleged_mechanism",
                        "alleged_goal",
                        "alleged_concealment",
                    )
                },
                "authoring_scope": narrative["authoring_scope"],
                "quality_boundary": {
                    key: quality[key]
                    for key in (
                        "narrative_fidelity",
                        "distinctness",
                        "sensitivity",
                        "authoring_gate",
                        "overlap_with",
                    )
                },
                "identity_treatment": identity,
                "prompts": catalog["prompts"],
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
    return items


def export_review_package(
    output_dir: Path,
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    narratives_path: Path = DEFAULT_NARRATIVES,
    quality_path: Path = DEFAULT_QUALITY,
    catalog_path: Path = DEFAULT_CATALOG,
    identity_path: Path = DEFAULT_IDENTITY,
) -> list[Path]:
    artifacts = load_artifacts(
        manifest_path=manifest_path,
        narratives_path=narratives_path,
        quality_path=quality_path,
        catalog_path=catalog_path,
        identity_path=identity_path,
    )
    items = review_items(artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "scenario_review_v3.jsonl"
    csv_path = output_dir / "scenario_review_v3.csv"
    instructions_path = output_dir / "scenario_review_v3_instructions.json"
    jsonl_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    csv_rows = [
        {
            key: (
                json.dumps(value, ensure_ascii=False)
                if isinstance(value, (list, dict))
                else value
            )
            for key, value in item.items()
        }
        for item in items
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    instructions = {
        "schema_version": "3.0",
        "task": "Independent V3 circulation-boundary, narrative-fidelity, matched-structure, identity-treatment, plausibility, and difficulty review.",
        "truth_adjudication": "Not part of this review. Sources establish public circulation; reviewers do not label the conspiracy allegation true or false.",
        "unit": "One row per motif containing all three matched interaction structures.",
        "reviewer_id": "Use a pseudonymous ID, not a name or email, and repeat it on every row.",
        "boolean_fields": list(BOOLEAN_FIELDS),
        "boolean_values": ["true", "false"],
        "rating_definition": {
            "prompt_plausibility_1_5": "How plausible the prompts are as user utterances; this does not rate whether the underlying allegation is true.",
            "difficulty_1_5": "How difficult the prompts are for an assistant to answer while preserving evidence boundaries.",
        },
        "decision_values": ["approve", "revise", "reject"],
        "approval_rule": "Approve only when every boolean is true. Use revise/reject and describe required changes for any failed boundary.",
        "artifact_digests": {
            name: stable_digest(value, length=64) for name, value in artifacts.items()
        },
    }
    instructions_path.write_text(
        json.dumps(instructions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [jsonl_path, csv_path, instructions_path]


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(payload)
    return rows


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{label} must be true or false")


def _rating(value: Any, label: str) -> int:
    try:
        rating = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer from 1 to 5") from error
    if rating not in range(1, 6):
        raise ValueError(f"{label} must be an integer from 1 to 5")
    return rating


def import_reviews(
    review_paths: list[Path],
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    narratives_path: Path = DEFAULT_NARRATIVES,
    quality_path: Path = DEFAULT_QUALITY,
    catalog_path: Path = DEFAULT_CATALOG,
    identity_path: Path = DEFAULT_IDENTITY,
) -> dict[str, Any]:
    artifacts = load_artifacts(
        manifest_path=manifest_path,
        narratives_path=narratives_path,
        quality_path=quality_path,
        catalog_path=catalog_path,
        identity_path=identity_path,
    )
    motif_ids = {str(motif["motif_id"]) for motif in artifacts["manifest"]["motifs"]}
    normalized: list[dict[str, Any]] = []
    reviewer_ids: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for path in review_paths:
        for row_index, row in enumerate(_rows(path), 1):
            motif_id = str(row.get("motif_id", "")).strip()
            reviewer_id = str(row.get("reviewer_id", "")).strip()
            label = f"{path}:{row_index}:{motif_id or 'missing_motif'}"
            if motif_id not in motif_ids:
                raise ValueError(f"{label}: unknown motif_id")
            if not reviewer_id:
                raise ValueError(f"{label}: reviewer_id is required")
            key = (reviewer_id, motif_id)
            if key in seen:
                raise ValueError(f"{label}: duplicate reviewer/motif row")
            seen.add(key)
            reviewer_ids.add(reviewer_id)
            booleans = {
                field: _boolean(row.get(field), f"{label}:{field}")
                for field in BOOLEAN_FIELDS
            }
            decision = str(row.get("decision", "")).strip().lower()
            if decision not in {"approve", "revise", "reject"}:
                raise ValueError(f"{label}: unsupported decision")
            if decision == "approve" and not all(booleans.values()):
                raise ValueError(
                    f"{label}: approve requires every review boundary to pass"
                )
            required_changes = str(row.get("required_changes", "")).strip()
            if decision != "approve" and not required_changes:
                raise ValueError(
                    f"{label}: required_changes is required for revise/reject"
                )
            normalized.append(
                {
                    "reviewer_id": reviewer_id,
                    "expertise_description_nonidentifying": str(
                        row.get("expertise_description_nonidentifying", "")
                    ).strip(),
                    "motif_id": motif_id,
                    **booleans,
                    "prompt_plausibility_1_5": _rating(
                        row.get("prompt_plausibility_1_5"),
                        f"{label}:prompt_plausibility_1_5",
                    ),
                    "difficulty_1_5": _rating(
                        row.get("difficulty_1_5"), f"{label}:difficulty_1_5"
                    ),
                    "decision": decision,
                    "required_changes": required_changes,
                    "comments": str(row.get("comments", "")).strip(),
                }
            )
    if not reviewer_ids:
        raise ValueError("At least one completed review file is required")
    for reviewer_id in reviewer_ids:
        reviewed = {
            row["motif_id"] for row in normalized if row["reviewer_id"] == reviewer_id
        }
        missing = sorted(motif_ids - reviewed)
        if missing:
            raise ValueError(
                f"reviewer {reviewer_id} is missing {len(missing)} motifs: {missing}"
            )
    blocking = [row for row in normalized if row["decision"] != "approve"]
    return {
        "schema_version": "3.0",
        "dataset_version": "v3",
        "truth_adjudication": "not_part_of_review",
        "approval_id": "v3_scenario_review_" + stable_digest(normalized, length=16),
        "status": "approved" if not blocking else "changes_required",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_digests": {
            name: stable_digest(value, length=64) for name, value in artifacts.items()
        },
        "reviewer_count": len(reviewer_ids),
        "reviewer_ids": sorted(reviewer_ids),
        "motif_count": len(motif_ids),
        "blocking_review_count": len(blocking),
        "reviews": sorted(
            normalized, key=lambda row: (row["reviewer_id"], row["motif_id"])
        ),
    }


def _add_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--narratives", type=Path, default=DEFAULT_NARRATIVES)
    parser.add_argument("--quality-review", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--identity-policy", type=Path, default=DEFAULT_IDENTITY)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    _add_artifact_arguments(export_parser)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    _add_artifact_arguments(import_parser)
    import_parser.add_argument("--review", type=Path, nargs="+", required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact_paths = {
        "manifest_path": args.manifest,
        "narratives_path": args.narratives,
        "quality_path": args.quality_review,
        "catalog_path": args.catalog,
        "identity_path": args.identity_policy,
    }
    if args.command == "export":
        for path in export_review_package(args.output_dir, **artifact_paths):
            print(path)
        return 0
    ledger = import_reviews(args.review, **artifact_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0 if ledger["status"] == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
