#!/usr/bin/env python3
"""Run the frozen 12-motif Gemini replication through the standard CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import cli as benchmark_cli

DEFAULT_CONFIG = ROOT / "configs" / "experiment_v3_gemini_paid_replication.json"
DEFAULT_DATASET = ROOT / "Conspire-Bench-v3.json"
DEFAULT_SELECTION = ROOT / "configs" / "paraphrase_robustness_v1.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def selected_scenario_ids(
    dataset_path: Path = DEFAULT_DATASET,
    selection_path: Path = DEFAULT_SELECTION,
) -> list[str]:
    dataset = _read_json(dataset_path)
    design = _read_json(selection_path)
    selection = design.get("selection") or {}
    motif_ids = selection.get("motif_ids") or []
    structures = selection.get("required_structures") or []
    if len(motif_ids) != 12 or len(set(motif_ids)) != 12:
        raise ValueError("Gemini replication requires 12 unique frozen motifs")
    if len(structures) != 3 or len(set(structures)) != 3:
        raise ValueError("Gemini replication requires three unique structures")

    by_cell: dict[tuple[str, str], list[str]] = {}
    for scenario in dataset.get("scenarios") or []:
        motif_id = scenario.get("subcategory")
        structure = scenario.get("type")
        if motif_id in motif_ids and structure in structures:
            by_cell.setdefault((str(motif_id), str(structure)), []).append(
                str(scenario["id"])
            )

    scenario_ids: list[str] = []
    for motif_id in motif_ids:
        for structure in structures:
            matches = by_cell.get((motif_id, structure), [])
            if len(matches) != 1:
                raise ValueError(
                    "Expected one scenario for frozen cell "
                    f"{motif_id}/{structure}; found {len(matches)}"
                )
            scenario_ids.append(matches[0])
    if len(scenario_ids) != 36 or len(set(scenario_ids)) != 36:
        raise ValueError("Gemini replication must resolve to 36 unique scenarios")
    return scenario_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", default="gemini_paid_replication.json")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "results" / "gemini_paid_replication_v1" / "status.tsv",
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark_args = [
        "--config",
        str(args.config),
        "--dataset",
        str(args.dataset),
        "--context-set",
        "main_v3",
        "--execution-mode",
        "phased",
        "--scenario-ids",
        *selected_scenario_ids(args.dataset, args.selection),
        "--output",
        args.output,
        "--status-file",
        str(args.status_file),
    ]
    if args.resume_from:
        benchmark_args.extend(["--resume-from", str(args.resume_from)])
    if args.dry_run:
        benchmark_args.append("--dry-run")
    return benchmark_cli(benchmark_args)


if __name__ == "__main__":
    raise SystemExit(main())
