#!/usr/bin/env python3
"""Merge disjoint generation-only result bundles for one judge-only pass."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_bundle(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("detailed_results"), list
    ):
        raise ValueError(f"{path} is not a benchmark result bundle")
    return payload.get("metadata") or {}, payload["detailed_results"]


def merge_bundles(
    paths: list[Path],
    *,
    expected_rows: int | None = None,
    expected_model_names: set[str] | None = None,
    expected_rows_per_model: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_metadata: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}

    for path in paths:
        metadata, source_rows = load_bundle(path)
        source_metadata.append({"path": str(path), "metadata": metadata})
        for row in source_rows:
            response_id = row.get("response_id")
            if not isinstance(response_id, str) or not response_id:
                raise ValueError(f"{path} contains a row without response_id")
            if response_id in seen:
                raise ValueError(
                    f"duplicate response_id {response_id!r} in {seen[response_id]} and {path}"
                )
            if row.get("error") or not row.get("conversation_log"):
                raise ValueError(
                    f"incomplete generated response {response_id!r} in {path}"
                )
            seen[response_id] = path
            rows.append(row)

    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")

    model_counts = Counter(str(row.get("model_name")) for row in rows)
    if expected_model_names is not None:
        actual_model_names = set(model_counts)
        if actual_model_names != expected_model_names:
            missing = sorted(expected_model_names - actual_model_names)
            unexpected = sorted(actual_model_names - expected_model_names)
            raise ValueError(
                "target model mismatch: "
                f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
            )
    if expected_rows_per_model is not None:
        invalid_counts = {
            model_name: count
            for model_name, count in sorted(model_counts.items())
            if count != expected_rows_per_model
        }
        if invalid_counts:
            raise ValueError(
                f"expected {expected_rows_per_model} rows per model; "
                f"found {invalid_counts}"
            )

    return {
        "metadata": {
            "schema_version": "2.0",
            "execution_mode": "merged-generation-only",
            "merged_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_bundles": source_metadata,
            "row_count": len(rows),
            "model_names": sorted(model_counts),
            "rows_per_model": dict(sorted(model_counts.items())),
        },
        "summary": {},
        "detailed_results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Require the exact provider/model set declared in this config.",
    )
    parser.add_argument("--expected-rows-per-model", type=int)
    args = parser.parse_args()

    expected_model_names = None
    if args.model_config:
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        models = config.get("models") if isinstance(config, dict) else None
        if not isinstance(models, list) or not models:
            raise ValueError(f"{args.model_config} has no models list")
        expected_model_names = {
            f"{model['provider']}/{model['model']}" for model in models
        }

    payload = merge_bundles(
        args.inputs,
        expected_rows=args.expected_rows,
        expected_model_names=expected_model_names,
        expected_rows_per_model=args.expected_rows_per_model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"merged {len(payload['detailed_results'])} rows into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
