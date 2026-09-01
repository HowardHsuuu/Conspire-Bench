#!/usr/bin/env python3
"""Freeze the human annotation workload after a timed calibration/UI pilot."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.human_annotation_plan import (
    human_annotation_plan_digest,
    load_human_annotation_plan,
)


def freeze_human_plan(
    plan: dict,
    *,
    approved_by: list[str],
    expert_minutes_median: float,
    student_minutes_median: float,
    change_summary: str,
    frozen_at: str,
) -> dict:
    if not approved_by or len(approved_by) != len(set(approved_by)):
        raise ValueError("Provide one or more unique pseudonymous approver IDs")
    if expert_minutes_median <= 0 or student_minutes_median <= 0:
        raise ValueError("Timed item medians must be positive")
    if not change_summary.strip():
        raise ValueError("Record whether the timed pilot changed the plan")
    frozen = deepcopy(plan)
    frozen["status"] = "frozen"
    frozen["freeze_record"] = {
        "frozen_at": frozen_at,
        "approved_by": approved_by,
        "timed_calibration_completed": True,
        "expert_calibration_item_minutes_median": expert_minutes_median,
        "student_pair_ui_item_minutes_median": student_minutes_median,
        "workload_feasibility_confirmed": True,
        "change_summary": change_summary.strip(),
    }
    # Re-run the complete frozen-state validator before writing anything.
    from analysis.human_annotation_plan import validate_human_annotation_plan

    errors = validate_human_annotation_plan(frozen, require_frozen=True)
    if errors:
        raise ValueError(
            "Frozen human annotation plan is invalid:\n- " + "\n- ".join(errors)
        )
    return frozen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=ROOT / "configs" / "human_annotation_plan_v2.json",
    )
    parser.add_argument("--approved-by", action="append", required=True)
    parser.add_argument("--expert-minutes-median", type=float, required=True)
    parser.add_argument("--student-minutes-median", type=float, required=True)
    parser.add_argument("--change-summary", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = freeze_human_plan(
        load_human_annotation_plan(args.plan),
        approved_by=args.approved_by,
        expert_minutes_median=args.expert_minutes_median,
        student_minutes_median=args.student_minutes_median,
        change_summary=args.change_summary,
        frozen_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{args.output} {human_annotation_plan_digest(frozen)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
