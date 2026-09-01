#!/usr/bin/env python3
"""Export the expert rubric-validity form before any response pool exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .export_annotations import rubric_feedback_template, write_rubric_feedback_csv
except ImportError:
    from export_annotations import rubric_feedback_template, write_rubric_feedback_csv


def export_forms(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    template = rubric_feedback_template()
    json_path = output_dir / "expert_rubric_feedback.json"
    csv_path = output_dir / "expert_rubric_feedback.csv"
    json_path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_rubric_feedback_csv(csv_path, template)
    return [json_path, csv_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in export_forms(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
