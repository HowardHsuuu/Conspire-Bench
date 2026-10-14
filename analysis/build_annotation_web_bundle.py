#!/usr/bin/env python3
"""Build a self-contained, preloaded expert-annotation web bundle.

The bundle can target one fixed pseudonymous annotator or authenticate multiple
experts through private invitation links while keeping raw invite codes local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.export_annotations import validate_public_items
from experiment_conditions import stable_digest


def load_assigned_items(
    path: Path, *, invitations_enabled: bool = False
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no annotation items")
    validate_public_items(rows)
    annotator_ids = {
        str(row.get("annotator_id") or "").strip()
        for row in rows
        if str(row.get("annotator_id") or "").strip()
    }
    if invitations_enabled and len(annotator_ids) <= 1:
        normalized = []
        for row in rows:
            copy = dict(row)
            copy["annotator_id"] = ""
            normalized.append(copy)
        return "", normalized
    if len(annotator_ids) != 1:
        raise ValueError(
            "Assigned items must contain exactly one pseudonymous annotator ID"
        )
    return next(iter(annotator_ids)), rows


def load_private_invites(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    invitations = payload.get("invitations") if isinstance(payload, dict) else None
    if not isinstance(invitations, list) or not invitations:
        raise ValueError(f"{path}: no invitations")
    normalized = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for invitation in invitations:
        annotator_id = str(invitation.get("annotator_id") or "").strip()
        invite_code = str(invitation.get("invite_code") or "")
        invite_hash = str(invitation.get("invite_hash") or "").lower()
        calculated = hashlib.sha256(invite_code.encode("utf-8")).hexdigest()
        if (
            not annotator_id
            or annotator_id in seen_ids
            or not invite_code
            or invite_hash != calculated
            or invite_hash in seen_hashes
        ):
            raise ValueError(f"{path}: invalid or duplicate invitation")
        seen_ids.add(annotator_id)
        seen_hashes.add(invite_hash)
        normalized.append(
            {"annotatorId": annotator_id, "inviteHash": invite_hash}
        )
    return normalized


def _javascript_assignment(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return '"use strict";\nwindow.CONSPIRE_BENCH_ASSIGNMENT = ' + encoded + ";\n"


def _javascript_constraints(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return "export const ASSIGNMENT_CONSTRAINTS = Object.freeze(" + encoded + ");\n"


def build_web_bundle(
    assignment_path: Path,
    output_dir: Path,
    *,
    title: str = "Expert conversation annotation",
    minutes_per_item: float = 8.0,
    invite_manifest_path: Path | None = None,
) -> list[Path]:
    if minutes_per_item <= 0:
        raise ValueError("minutes_per_item must be positive")
    private_invites = (
        load_private_invites(invite_manifest_path)
        if invite_manifest_path is not None
        else []
    )
    annotator_id, rows = load_assigned_items(
        assignment_path, invitations_enabled=bool(private_invites)
    )
    public_digest = stable_digest(rows, length=64)
    assignment_id = f"assigned-{public_digest[:16]}"
    payload = {
        "assignmentId": assignment_id,
        "annotatorId": annotator_id,
        "requiresInvite": bool(private_invites),
        "title": title,
        "minutesPerItem": minutes_per_item,
        "demo": False,
        "publicItemDigest": public_digest,
        "items": rows,
    }

    source_dir = ROOT / "annotation_ui"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_text = (source_dir / "index.html").read_text(encoding="utf-8")
    index_text = index_text.replace(
        "demo_assignment.js?v=20261014f",
        f"assignment.js?v={public_digest[:12]}",
    )
    if "demo_assignment.js" in index_text:
        raise ValueError("Failed to replace the demo assignment in the web bundle")

    written = []
    index_path = output_dir / "index.html"
    index_path.write_text(index_text, encoding="utf-8")
    written.append(index_path)
    for name in (
        "app.js",
        "styles.css",
        "vercel.json",
        "robots.txt",
        "package.json",
        "package-lock.json",
    ):
        destination = output_dir / name
        shutil.copyfile(source_dir / name, destination)
        written.append(destination)
    assignment_output = output_dir / "assignment.js"
    assignment_output.write_text(_javascript_assignment(payload), encoding="utf-8")
    written.append(assignment_output)

    api_output_dir = output_dir / "api"
    api_output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "submit.js",
        "submission_validation.js",
        "invite_auth.js",
        "invite.js",
    ):
        destination = api_output_dir / name
        shutil.copyfile(source_dir / "api" / name, destination)
        written.append(destination)
    constraint_items = [
        {
            "annotationItemId": row["annotation_item_id"],
            "assistantTurnCount": sum(
                message.get("role") == "assistant"
                for message in row.get("transcript") or []
            ),
        }
        for row in rows
    ]
    item_ids = [item["annotationItemId"] for item in constraint_items]
    public_invites = []
    for invitation in private_invites:
        item_order = list(item_ids)
        random.Random(int(invitation["inviteHash"][:16], 16)).shuffle(item_order)
        public_invites.append({**invitation, "itemOrder": item_order})
    constraints = {
        "assignmentId": assignment_id,
        "annotatorId": annotator_id,
        "invites": public_invites,
        "publicItemDigest": public_digest,
        "itemType": rows[0]["item_type"],
        "annotationSchemaVersion": rows[0].get("annotation_schema_version"),
        "rubricVersion": rows[0].get("rubric_version"),
        "expectedItemCount": len(rows),
        "items": constraint_items,
    }
    constraints_path = api_output_dir / "assignment_constraints.js"
    constraints_path.write_text(_javascript_constraints(constraints), encoding="utf-8")
    written.append(constraints_path)

    manifest = {
        "schema_version": "1.0",
        "bundle_type": "preloaded_public_annotation_assignment",
        "assignment_id": assignment_id,
        "annotator_id": annotator_id or None,
        "annotator_ids": [row["annotatorId"] for row in public_invites],
        "invitation_mode": "private_fragment_links" if public_invites else "fixed_annotator",
        "item_type": rows[0]["item_type"],
        "item_count": len(rows),
        "planned_minutes_per_item": minutes_per_item,
        "public_item_digest": public_digest,
        "contains_private_key": False,
        "network_dependencies": True,
        "annotation_schema_version": rows[0].get("annotation_schema_version"),
        "rubric_content_validity_included": rows[0]["item_type"]
        == "expert_conversation",
        "submission_format": "server_posted_json_with_rubric_feedback_and_annotations",
        "submission_storage": "private_vercel_blob",
        "researcher_download_location": "vercel_storage_dashboard",
        "files": [path.relative_to(output_dir).as_posix() for path in written],
    }
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append(manifest_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assignment", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="Expert conversation annotation")
    parser.add_argument("--minutes-per-item", type=float, default=8.0)
    parser.add_argument(
        "--invite-manifest",
        type=Path,
        help="Private invitation manifest; raw invitation codes are never copied",
    )
    args = parser.parse_args()
    if args.minutes_per_item <= 0:
        parser.error("--minutes-per-item must be positive")
    for path in build_web_bundle(
        args.assignment,
        args.out_dir,
        title=args.title,
        minutes_per_item=args.minutes_per_item,
        invite_manifest_path=args.invite_manifest,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
