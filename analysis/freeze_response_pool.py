#!/usr/bin/env python3
"""Validate completeness and create an immutable response-pool freeze manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.export_annotations import (
    formal_readiness_errors,
    load_results,
    usable_rows,
)
from experiment_conditions import stable_digest


def freeze_manifest(path: Path, *, require_judges: bool = True) -> dict[str, Any]:
    metadata, raw_rows = load_results(path)
    rows = usable_rows(raw_rows)
    errors = []
    if len(rows) != len(raw_rows):
        errors.append(
            f"{len(raw_rows) - len(rows)} rows are incomplete or contain errors"
        )
    response_ids = [row.get("response_id") for row in rows]
    condition_ids = [row.get("condition_id") for row in rows]
    if None in response_ids or len(response_ids) != len(set(response_ids)):
        errors.append("response_id values are missing or duplicated")
    if None in condition_ids or len(condition_ids) != len(set(condition_ids)):
        errors.append("condition_id values are missing or duplicated")
    errors.extend(formal_readiness_errors(rows))
    if require_judges:
        missing = [
            row.get("response_id") for row in rows if not row.get("judge_scores")
        ]
        if missing:
            errors.append(
                f"{len(missing)} rows have no successful primary judge scores"
            )

    expected_scenarios = int(metadata.get("total_scenarios") or 0)
    expected_models = len(metadata.get("models") or [])
    expected_contexts = len((metadata.get("filters") or {}).get("contexts") or [])
    expected_total = expected_scenarios * expected_models * expected_contexts
    if expected_total and len(rows) != expected_total:
        errors.append(
            f"expected {expected_total} rows from metadata, found {len(rows)}"
        )
    if errors:
        preview = "; ".join(errors[:10])
        raise ValueError(f"Response pool cannot be frozen: {preview}")

    return {
        "schema_version": "1.0",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(path),
        "source_digest": stable_digest(
            {"metadata": metadata, "rows": raw_rows}, length=64
        ),
        "response_count": len(rows),
        "require_judges": require_judges,
        "models": dict(
            sorted(
                Counter(
                    str(row.get("model_name") or row.get("target_model"))
                    for row in rows
                ).items()
            )
        ),
        "frame_families": dict(
            sorted(
                Counter(
                    str(row.get("frame_family") or row.get("context_label"))
                    for row in rows
                ).items()
            )
        ),
        "rubric_versions": sorted(
            {
                str(judge.get("rubric_version"))
                for row in rows
                for judge in (row.get("judge_results") or [])
                if judge.get("rubric_version")
            }
        ),
        "response_ids_digest": stable_digest(sorted(response_ids), length=64),
        "condition_ids_digest": stable_digest(sorted(condition_ids), length=64),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unjudged", action="store_true")
    args = parser.parse_args()
    manifest = freeze_manifest(args.input, require_judges=not args.allow_unjudged)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
