#!/usr/bin/env python3
"""Select a deterministic, stratified scenario subset for robustness sweeps."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scenario_metadata import enrich_dataset
from scenario_expansion import load_benchmark_dataset, scenario_content_digest


def select_subset(
    dataset: dict[str, Any],
    count: int,
    seed: int,
    *,
    include_controls: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    raw_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in dataset.get("scenarios", []):
        if scenario.get("is_control") and not include_controls:
            continue
        raw_groups[str(scenario.get("category"))].append(scenario)
    selected = []
    type_counts: dict[str, int] = defaultdict(int)
    active = sorted(raw_groups)
    while active and len(selected) < count:
        next_active = []
        for category in active:
            candidates = raw_groups[category]
            if candidates and len(selected) < count:
                minimum = min(type_counts[str(item.get("type"))] for item in candidates)
                tied = [
                    item
                    for item in candidates
                    if type_counts[str(item.get("type"))] == minimum
                ]
                chosen = rng.choice(tied)
                candidates.remove(chosen)
                selected.append(chosen)
                type_counts[str(chosen.get("type"))] += 1
            if candidates:
                next_active.append(category)
        active = next_active
    return selected


def build_manifest(
    dataset_path: Path,
    count: int,
    seed: int,
    *,
    include_controls: bool = False,
    purpose: str = "reviewer_rephrasing_robustness_subset",
) -> dict[str, Any]:
    raw = load_benchmark_dataset(dataset_path)
    dataset = enrich_dataset(raw)
    selected = select_subset(
        dataset,
        max(0, count),
        seed,
        include_controls=include_controls,
    )
    ids = [scenario["id"] for scenario in selected]
    return {
        "schema_version": "1.0",
        "purpose": purpose,
        "sampling_seed": seed,
        "requested_count": count,
        "selected_count": len(ids),
        "include_controls": include_controls,
        "dataset_digest": scenario_content_digest(dataset),
        "scenario_ids": ids,
        "cli_argument": "--scenario-ids " + " ".join(ids),
        "strata": [
            [
                scenario["id"],
                scenario.get("category"),
                scenario.get("type"),
                scenario.get("risk_level"),
            ]
            for scenario in selected
        ],
        "strata_columns": [
            "scenario_id",
            "category",
            "scenario_type",
            "risk_level",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Permit matched non-conspiratorial controls in the robustness subset.",
    )
    parser.add_argument(
        "--purpose",
        default="reviewer_rephrasing_robustness_subset",
        help="Machine-readable purpose recorded in the manifest.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        args.dataset,
        args.count,
        args.seed,
        include_controls=args.include_controls,
        purpose=args.purpose,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(manifest["cli_argument"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
