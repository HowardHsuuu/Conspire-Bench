#!/usr/bin/env python3
"""Validate the prespecified Conspire-Bench v2 human annotation workload."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.human_annotation_plan import load_human_annotation_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plan",
        nargs="?",
        type=Path,
        default=ROOT / "configs" / "human_annotation_plan_v2.json",
    )
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    try:
        load_human_annotation_plan(args.plan, require_frozen=args.require_frozen)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Human annotation plan is internally consistent: {args.plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
