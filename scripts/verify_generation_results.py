#!/usr/bin/env python3
"""Verify a complete generation-only result bundle before judging."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_generation_bundle(
    bundle: dict[str, Any],
    config: dict[str, Any],
    *,
    expected_rows_per_model: int,
) -> dict[str, Any]:
    rows = bundle.get("detailed_results") if isinstance(bundle, dict) else None
    models = config.get("models") if isinstance(config, dict) else None
    if not isinstance(rows, list):
        raise ValueError("result bundle has no detailed_results list")
    if not isinstance(models, list) or not models:
        raise ValueError("config has no models list")

    expected_models = {f"{model['provider']}/{model['model']}" for model in models}
    model_counts: Counter[str] = Counter()
    response_ids: set[str] = set()
    condition_ids: set[str] = set()
    errors: list[str] = []

    for index, row in enumerate(rows):
        response_id = row.get("response_id")
        condition_id = row.get("condition_id")
        model_name = str(row.get("model_name"))
        label = str(response_id or f"row[{index}]")

        if not isinstance(response_id, str) or not response_id:
            errors.append(f"{label}: missing response_id")
        elif response_id in response_ids:
            errors.append(f"{label}: duplicate response_id")
        else:
            response_ids.add(response_id)

        if not isinstance(condition_id, str) or not condition_id:
            errors.append(f"{label}: missing condition_id")
        elif condition_id in condition_ids:
            errors.append(f"{label}: duplicate condition_id")
        else:
            condition_ids.add(condition_id)

        if model_name not in expected_models:
            errors.append(f"{label}: unexpected target model {model_name!r}")
        else:
            model_counts[model_name] += 1
        if row.get("error"):
            errors.append(f"{label}: generation error: {row['error']}")
        if not row.get("generation_complete"):
            errors.append(f"{label}: generation_complete is not true")
        if not row.get("conversation_log"):
            errors.append(f"{label}: missing conversation_log")

        if len(errors) >= 50:
            break

    actual_models = set(model_counts)
    if actual_models != expected_models:
        errors.append(
            "target model set mismatch: "
            f"missing={sorted(expected_models - actual_models) or 'none'}, "
            f"unexpected={sorted(actual_models - expected_models) or 'none'}"
        )
    invalid_counts = {
        name: model_counts.get(name, 0)
        for name in sorted(expected_models)
        if model_counts.get(name, 0) != expected_rows_per_model
    }
    if invalid_counts:
        errors.append(
            f"expected {expected_rows_per_model} rows per model; found {invalid_counts}"
        )

    expected_total = len(expected_models) * expected_rows_per_model
    if len(rows) != expected_total:
        errors.append(f"expected {expected_total} rows, found {len(rows)}")

    report = {
        "ok": not errors,
        "row_count": len(rows),
        "target_model_count": len(expected_models),
        "rows_per_model": dict(sorted(model_counts.items())),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-rows-per-model", type=int, default=765)
    args = parser.parse_args()

    report = verify_generation_bundle(
        _read_json(args.bundle),
        _read_json(args.config),
        expected_rows_per_model=args.expected_rows_per_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
