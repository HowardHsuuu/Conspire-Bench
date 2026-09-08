#!/usr/bin/env python3
"""Combine split target-generation configs into one judge-only config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def read_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("models"), list):
        raise ValueError(f"{path} has no models list")
    return payload


def merge_configs(
    generation_paths: list[Path], judge_source_path: Path
) -> dict[str, Any]:
    generation_configs = [read_config(path) for path in generation_paths]
    judge_source = read_config(judge_source_path)
    judges = judge_source.get("judges")
    if not isinstance(judges, list) or not judges:
        raise ValueError(f"{judge_source_path} has no judges list")

    merged = deepcopy(generation_configs[0])
    merged_models: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for config in generation_configs:
        for model in config["models"]:
            key = (str(model.get("provider")), str(model.get("model")))
            if key in seen:
                raise ValueError(f"duplicate target model {key[0]}/{key[1]}")
            seen.add(key)
            merged_models.append(deepcopy(model))

    merged["models"] = merged_models
    merged["judges"] = deepcopy(judges)
    merged["experiment"] = {
        **(judge_source.get("experiment") or {}),
        "stage": "gb10_all_open_models_four_family_judges",
        "purpose": "judge_only_cross_family_primary_and_family_sensitivity",
    }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_configs", nargs="+", type=Path)
    parser.add_argument("--judge-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = merge_configs(args.generation_configs, args.judge_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"wrote {len(payload['models'])} targets and {len(payload['judges'])} judges "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
