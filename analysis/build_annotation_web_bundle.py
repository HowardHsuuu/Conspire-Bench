#!/usr/bin/env python3
"""Build a self-contained, preloaded annotation page for one annotator."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.export_annotations import validate_public_items
from experiment_conditions import stable_digest


def load_assigned_items(path: Path) -> tuple[str, list[dict[str, Any]]]:
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
    if len(annotator_ids) != 1:
        raise ValueError(
            "Assigned items must contain exactly one pseudonymous annotator ID"
        )
    return next(iter(annotator_ids)), rows


def _javascript_assignment(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return '"use strict";\nwindow.CONSPIRE_BENCH_ASSIGNMENT = ' + encoded + ";\n"


def build_web_bundle(
    assignment_path: Path,
    output_dir: Path,
    *,
    title: str = "Expert conversation annotation",
    minutes_per_item: float = 8.0,
) -> list[Path]:
    if minutes_per_item <= 0:
        raise ValueError("minutes_per_item must be positive")
    annotator_id, rows = load_assigned_items(assignment_path)
    public_digest = stable_digest(rows, length=64)
    assignment_id = f"assigned-{public_digest[:16]}"
    payload = {
        "assignmentId": assignment_id,
        "annotatorId": annotator_id,
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
        "demo_assignment.js?v=20261014e",
        f"assignment.js?v={public_digest[:12]}",
    )
    if "demo_assignment.js" in index_text:
        raise ValueError("Failed to replace the demo assignment in the web bundle")

    written = []
    index_path = output_dir / "index.html"
    index_path.write_text(index_text, encoding="utf-8")
    written.append(index_path)
    for name in ("app.js", "styles.css", "vercel.json", "robots.txt"):
        destination = output_dir / name
        shutil.copyfile(source_dir / name, destination)
        written.append(destination)
    assignment_output = output_dir / "assignment.js"
    assignment_output.write_text(_javascript_assignment(payload), encoding="utf-8")
    written.append(assignment_output)

    manifest = {
        "schema_version": "1.0",
        "bundle_type": "preloaded_public_annotation_assignment",
        "assignment_id": assignment_id,
        "annotator_id": annotator_id,
        "item_type": rows[0]["item_type"],
        "item_count": len(rows),
        "planned_minutes_per_item": minutes_per_item,
        "public_item_digest": public_digest,
        "contains_private_key": False,
        "network_dependencies": False,
        "annotation_schema_version": rows[0].get("annotation_schema_version"),
        "rubric_content_validity_included": rows[0]["item_type"] == "expert_conversation",
        "submission_format": "single_json_with_rubric_feedback_and_annotations",
        "files": [path.name for path in written],
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
    args = parser.parse_args()
    if args.minutes_per_item <= 0:
        parser.error("--minutes-per-item must be positive")
    for path in build_web_bundle(
        args.assignment,
        args.out_dir,
        title=args.title,
        minutes_per_item=args.minutes_per_item,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
