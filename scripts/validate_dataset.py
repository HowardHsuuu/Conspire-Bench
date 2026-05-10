#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation import format_validation_report, validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Conspire-Bench dataset JSON.")
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("CONSPIRE-Bench.json"),
        help="Dataset JSON path. Default: CONSPIRE-Bench.json",
    )
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Require expansion metadata fields on every scenario.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable validation report.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    report = validate_dataset(dataset, strict_metadata=args.strict_metadata)

    if args.json:
        print(json.dumps({
            "ok": report.ok,
            "scenario_count": report.scenario_count,
            "categories": report.categories,
            "scenario_types": report.scenario_types,
            "errors": report.errors,
            "warnings": report.warnings,
        }, indent=2, ensure_ascii=False))
    else:
        print(format_validation_report(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
