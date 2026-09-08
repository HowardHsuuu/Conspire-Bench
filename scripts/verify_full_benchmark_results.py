#!/usr/bin/env python3
"""Verify a completed all-model, all-judge Conspire-Bench result bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_name(config: dict[str, Any]) -> str:
    return f"{config['provider']}/{config['model']}"


def verify_bundle(
    bundle: dict[str, Any],
    config: dict[str, Any],
    *,
    expected_rows_per_model: int,
) -> dict[str, Any]:
    rows = bundle.get("detailed_results") if isinstance(bundle, dict) else None
    if not isinstance(rows, list):
        raise ValueError("result bundle has no detailed_results list")

    models = config.get("models")
    judges = config.get("judges")
    if not isinstance(models, list) or not models:
        raise ValueError("config has no models list")
    if not isinstance(judges, list) or not judges:
        raise ValueError("config has no judges list")

    expected_models = {_model_name(model): model for model in models}
    expected_judges = {_model_name(judge): judge for judge in judges}
    model_counts: Counter[str] = Counter()
    response_ids: set[str] = set()
    errors: list[str] = []

    for index, row in enumerate(rows):
        response_id = row.get("response_id")
        model_name = row.get("model_name")
        label = str(response_id or f"row[{index}]")

        if not isinstance(response_id, str) or not response_id:
            errors.append(f"{label}: missing response_id")
        elif response_id in response_ids:
            errors.append(f"{label}: duplicate response_id")
        else:
            response_ids.add(response_id)

        if model_name not in expected_models:
            errors.append(f"{label}: unexpected target model {model_name!r}")
            continue
        model_counts[model_name] += 1

        if row.get("error"):
            errors.append(f"{label}: row error: {row['error']}")
        if not row.get("conversation_log"):
            errors.append(f"{label}: missing conversation_log")

        judge_results = row.get("judge_results")
        if not isinstance(judge_results, list):
            errors.append(f"{label}: missing judge_results")
            continue
        successful = {
            str(judge_result.get("judge_name")): judge_result
            for judge_result in judge_results
            if not judge_result.get("error") and judge_result.get("scores")
        }
        if set(successful) != set(expected_judges):
            missing = sorted(set(expected_judges) - set(successful))
            unexpected = sorted(set(successful) - set(expected_judges))
            errors.append(
                f"{label}: judge mismatch; missing={missing or 'none'}, "
                f"unexpected={unexpected or 'none'}"
            )
            continue

        target_family = expected_models[model_name].get("model_family")
        same_family = [
            result
            for result in successful.values()
            if result.get("same_family_as_target")
        ]
        if len(same_family) != 1:
            errors.append(
                f"{label}: expected exactly one same-family judge, found "
                f"{len(same_family)}"
            )
        elif same_family[0].get("judge_model_family") != target_family:
            errors.append(f"{label}: same-family judge metadata is inconsistent")

        expected_primary = {
            name
            for name, judge in expected_judges.items()
            if judge.get("model_family") != target_family
        }
        if set(row.get("primary_judge_names") or []) != expected_primary:
            errors.append(f"{label}: primary_judge_names do not exclude own family")
        if not row.get("judge_scores"):
            errors.append(f"{label}: missing cross-family primary aggregate")
        if not row.get("same_family_excluded"):
            errors.append(f"{label}: same_family_excluded is not true")
        if not row.get("evaluation_complete"):
            errors.append(f"{label}: evaluation_complete is not true")

        if len(errors) >= 50:
            break

    expected_model_set = set(expected_models)
    actual_model_set = set(model_counts)
    if actual_model_set != expected_model_set:
        errors.append(
            "target model set mismatch: "
            f"missing={sorted(expected_model_set - actual_model_set) or 'none'}, "
            f"unexpected={sorted(actual_model_set - expected_model_set) or 'none'}"
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
        "judge_count": len(expected_judges),
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

    report = verify_bundle(
        _read_json(args.bundle),
        _read_json(args.config),
        expected_rows_per_model=args.expected_rows_per_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
