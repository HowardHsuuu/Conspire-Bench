#!/usr/bin/env python3
"""Export and import independent human review of v2 scenario/source packets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_conditions import stable_digest

REVIEW_FIELDS = [
    "reviewer_id",
    "expertise_description_nonidentifying",
    "independent_of_scenario_authoring",
    "fact_boundary_accurate",
    "prompts_construct_aligned",
    "matched_control_valid",
    "plausibility_1_5",
    "difficulty_1_5",
    "decision",
    "required_changes",
    "comments",
]


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def review_items(
    catalog: dict[str, Any], packets: dict[str, Any]
) -> list[dict[str, Any]]:
    packet_by_id = {packet["id"]: packet for packet in packets["packets"]}
    items = []
    for motif in catalog["motifs"]:
        packet = packet_by_id[motif["id"]]
        items.append(
            {
                "review_item_id": f"scenario_review::{motif['id']}",
                "motif_id": motif["id"],
                "category": motif["category"],
                "risk_level": motif["risk_level"],
                "belief_domain": motif["belief_domain"],
                "symptom_domain": motif["symptom_domain"],
                "verified_premises": packet["verified_premises"],
                "unknowns": packet["unknowns"],
                "unsupported_inference": packet["unsupported_inference"],
                "risk_flags": packet["risk_flags"],
                "sources": packet["sources"],
                "expected_safe_behavior": motif["expected_safe_behavior"],
                "prompts": motif["prompts"],
                **{field: "" for field in REVIEW_FIELDS},
            }
        )
    return items


def export_review_package(catalog_path: Path, output_dir: Path) -> list[Path]:
    catalog = _read_object(catalog_path)
    packet_path = (catalog_path.parent / catalog["source_packets"]).resolve()
    packets = _read_object(packet_path)
    items = review_items(catalog, packets)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "scenario_review.jsonl"
    csv_path = output_dir / "scenario_review.csv"
    instructions_path = output_dir / "scenario_review_instructions.json"
    jsonl_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    csv_rows = []
    for item in items:
        csv_rows.append(
            {
                key: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
                for key, value in item.items()
            }
        )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    instructions = {
        "schema_version": "1.0",
        "task": "Independent scenario construction, source-boundary, plausibility, difficulty, and matched-control review.",
        "unit": "One row per motif covering four prompts: multi-turn, single-turn, matched control, and resistance.",
        "reviewer_id": "Use a pseudonymous ID, not a name or email, and repeat it on every row.",
        "boolean_fields": [
            "independent_of_scenario_authoring",
            "fact_boundary_accurate",
            "prompts_construct_aligned",
            "matched_control_valid",
        ],
        "boolean_values": ["true", "false"],
        "decision_values": ["approve", "revise", "reject"],
        "approval_rule": (
            "approve only when all four booleans are true; use revise and describe "
            "required changes whenever a factual boundary, prompt, or control needs editing"
        ),
        "catalog_digest": stable_digest(catalog, length=64),
        "source_packet_digest": stable_digest(packets, length=64),
    }
    instructions_path.write_text(
        json.dumps(instructions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [jsonl_path, csv_path, instructions_path]


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if line.strip():
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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer from 1 to 5") from exc
    if rating not in range(1, 6):
        raise ValueError(f"{label} must be an integer from 1 to 5")
    return rating


def import_reviews(
    catalog_path: Path,
    review_paths: list[Path],
) -> dict[str, Any]:
    catalog = _read_object(catalog_path)
    packet_path = (catalog_path.parent / catalog["source_packets"]).resolve()
    packets = _read_object(packet_path)
    motif_ids = {motif["id"] for motif in catalog["motifs"]}
    normalized = []
    reviewer_ids = set()
    seen = set()
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
                for field in (
                    "independent_of_scenario_authoring",
                    "fact_boundary_accurate",
                    "prompts_construct_aligned",
                    "matched_control_valid",
                )
            }
            decision = str(row.get("decision", "")).strip().lower()
            if decision not in {"approve", "revise", "reject"}:
                raise ValueError(f"{label}: unsupported decision")
            if decision == "approve" and not all(booleans.values()):
                raise ValueError(
                    f"{label}: approve requires all four booleans to be true"
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
                    "plausibility_1_5": _rating(
                        row.get("plausibility_1_5"), f"{label}:plausibility_1_5"
                    ),
                    "difficulty_1_5": _rating(
                        row.get("difficulty_1_5"), f"{label}:difficulty_1_5"
                    ),
                    "decision": decision,
                    "required_changes": required_changes,
                    "comments": str(row.get("comments", "")).strip(),
                }
            )
    for reviewer_id in reviewer_ids:
        reviewed = {
            row["motif_id"] for row in normalized if row["reviewer_id"] == reviewer_id
        }
        missing = sorted(motif_ids - reviewed)
        if missing:
            raise ValueError(
                f"reviewer {reviewer_id} is missing {len(missing)} motifs: {missing}"
            )
    if not reviewer_ids:
        raise ValueError("At least one completed review file is required")
    blocking = [row for row in normalized if row["decision"] != "approve"]
    return {
        "schema_version": "1.0",
        "approval_id": "scenario_review_" + stable_digest(normalized, length=16),
        "status": "approved" if not blocking else "changes_required",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_digest": stable_digest(catalog, length=64),
        "source_packet_digest": stable_digest(packets, length=64),
        "reviewer_count": len(reviewer_ids),
        "reviewer_ids": sorted(reviewer_ids),
        "motif_count": len(motif_ids),
        "blocking_review_count": len(blocking),
        "reviews": sorted(
            normalized, key=lambda row: (row["reviewer_id"], row["motif_id"])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--catalog", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path, required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--catalog", type=Path, required=True)
    import_parser.add_argument("--review", type=Path, nargs="+", required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        for path in export_review_package(args.catalog, args.output_dir):
            print(path)
        return 0
    ledger = import_reviews(args.catalog, args.review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0 if ledger["status"] == "approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
