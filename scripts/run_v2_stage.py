#!/usr/bin/env python3
"""Run one frozen Conspire-Bench v2 stage without copying long ID lists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import cli


class StageConfig(TypedDict):
    config: str
    context_set: str
    manifest: str | None
    output: str


STAGES: dict[str, StageConfig] = {
    "pilot": {
        "config": "configs/experiment_v2_api_pilot.json",
        "context_set": "main_v2",
        "manifest": "configs/pilot_subset_v2_seed11.json",
        "output": "v2_pilot_seed11.json",
    },
    "main": {
        "config": "configs/experiment_v2_api_full.json",
        "context_set": "main_v2",
        "manifest": None,
        "output": "v2_main.json",
    },
    "robustness": {
        "config": "configs/experiment_v2_api_robustness.json",
        "context_set": "reviewer_robustness_v1",
        "manifest": "configs/robustness_subset_v2_seed24.json",
        "output": "v2_robustness_seed24.json",
    },
    "exploratory": {
        "config": "configs/experiment_v2_api_pilot.json",
        "context_set": "exploratory_elicitation_v1",
        "manifest": "configs/pilot_subset_v2_seed11.json",
        "output": "v2_exploratory_frames_seed11.json",
    },
}


def _manifest_ids(relative_path: str) -> list[str]:
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
    ids = payload.get("scenario_ids") or []
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(
            f"Manifest has missing or duplicate scenario IDs: {relative_path}"
        )
    return [str(value) for value in ids]


def build_stage_args(args: argparse.Namespace) -> list[str]:
    stage = STAGES[args.stage]
    output = args.output or stage["output"]
    command = [
        "--config",
        str(ROOT / stage["config"]),
        "--dataset",
        str(ROOT / "configs/scenario_expansion_v2.json"),
        "--context-variants-file",
        str(ROOT / "configs/context_variants.json"),
        "--context-set",
        stage["context_set"],
        "--strict-dataset-metadata",
        "--execution-mode",
        args.execution_mode,
        "--output",
        output,
    ]
    if stage["manifest"]:
        command.extend(["--scenario-ids", *_manifest_ids(stage["manifest"])])
    if args.resume_from:
        command.extend(["--resume-from", args.resume_from])
    if args.status_file:
        command.extend(["--status-file", args.status_file])
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument(
        "--execution-mode",
        choices=["standard", "phased", "generation-only", "judge-only"],
        default="phased",
    )
    parser.add_argument("--resume-from")
    parser.add_argument("--status-file")
    parser.add_argument("--output")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return cli(build_stage_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
