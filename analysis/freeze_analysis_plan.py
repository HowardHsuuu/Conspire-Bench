#!/usr/bin/env python3
"""Create a digest-bound frozen v2 analysis plan after expert calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.validate_analysis_plan import validate_plan
from experiment_conditions import stable_digest
from scenario_expansion import load_benchmark_dataset, scenario_content_digest
from scenario_metadata import enrich_dataset


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _inside_root(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Freeze evidence must be stored under {root}: {resolved}"
        ) from exc


def build_frozen_plan(
    plan: dict[str, Any],
    *,
    root: Path,
    content_validity_report: Path,
    calibration_exclusion_manifest: Path,
    calibration_decision_record: Path,
    git_commit: str,
    approved_by: list[str],
    frozen_at: str,
    human_annotation_plan: Path | None = None,
) -> dict[str, Any]:
    working_plan = deepcopy(plan)
    if human_annotation_plan is not None:
        working_plan["artifacts"]["human_annotation_plan"] = _inside_root(
            human_annotation_plan, root
        )
    draft_errors = validate_plan(working_plan, root=root, require_frozen=False)
    if draft_errors:
        raise ValueError("Draft plan is invalid:\n- " + "\n- ".join(draft_errors))
    if not approved_by or len(approved_by) != len(set(approved_by)):
        raise ValueError("Provide one or more unique pseudonymous approver IDs")

    validity = _read(content_validity_report)
    calibration = _read(calibration_exclusion_manifest)
    decision = _read(calibration_decision_record)
    artifacts = working_plan["artifacts"]
    dataset = enrich_dataset(load_benchmark_dataset(root / str(artifacts["dataset"])))
    main_config = _read(root / str(artifacts["main_config"]))
    robustness_config = _read(root / str(artifacts["robustness_config"]))
    human_annotation_config = _read(root / str(artifacts["human_annotation_plan"]))
    contexts = _read(root / str(artifacts["context_registry"]))

    frozen = deepcopy(working_plan)
    frozen["status"] = "frozen"
    frozen["rubric_freeze_record"] = {
        "content_validity_report": _inside_root(content_validity_report, root),
        "content_validity_report_digest": stable_digest(validity, length=64),
        "calibration_exclusion_manifest": _inside_root(
            calibration_exclusion_manifest, root
        ),
        "calibration_exclusion_manifest_digest": stable_digest(calibration, length=64),
        "calibration_decision_record": _inside_root(calibration_decision_record, root),
        "calibration_decision_record_digest": stable_digest(decision, length=64),
        "final_rubric_version": str(artifacts["rubric_version"]),
    }
    frozen["freeze_record"] = {
        "frozen_at": frozen_at,
        "git_commit": git_commit,
        "dataset_digest": scenario_content_digest(dataset),
        "config_digest": stable_digest(
            {
                "main": main_config,
                "robustness": robustness_config,
                "human_annotation": human_annotation_config,
            },
            length=64,
        ),
        "context_registry_digest": stable_digest(contexts, length=64),
        "approved_by": approved_by,
    }
    errors = validate_plan(frozen, root=root, require_frozen=True)
    if errors:
        raise ValueError("Frozen plan is invalid:\n- " + "\n- ".join(errors))
    return frozen


def _git_state(root: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "configs" / "analysis_plan_v2.json"
    )
    parser.add_argument("--content-validity-report", type=Path, required=True)
    parser.add_argument("--calibration-exclusion-manifest", type=Path, required=True)
    parser.add_argument("--calibration-decision-record", type=Path, required=True)
    parser.add_argument(
        "--human-annotation-plan",
        type=Path,
        help="Frozen workload plan produced after the timed calibration/UI pilot.",
    )
    parser.add_argument("--approved-by", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    commit, dirty = _git_state(ROOT)
    if dirty:
        raise SystemExit(
            "Refusing to freeze a dirty worktree. Commit the exact dataset, prompts, "
            "configs, rubric, code, and expert-derived amendments first."
        )
    frozen = build_frozen_plan(
        _read(args.plan),
        root=ROOT,
        content_validity_report=args.content_validity_report,
        calibration_exclusion_manifest=args.calibration_exclusion_manifest,
        calibration_decision_record=args.calibration_decision_record,
        git_commit=commit,
        approved_by=args.approved_by,
        frozen_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        human_annotation_plan=args.human_annotation_plan,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
