#!/usr/bin/env python3
"""Build v3 local and API runner configs from the tested provider settings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCE = ROOT / "configs" / "local_5090_full_matrix_config.json"
API_SOURCE = ROOT / "configs" / "experiment_v2_api_full.json"
LOCAL_OUTPUT = ROOT / "configs" / "experiment_v3_local_full.json"
API_OUTPUT = ROOT / "configs" / "experiment_v3_api_full.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(source: dict[str, Any], *, stage: str) -> dict[str, Any]:
    result = dict(source)
    result["experiment"] = {
        "design_version": "v3",
        "stage": stage,
        "dataset": "Conspire-Bench-v3.json",
        "motif_count": 51,
        "scenario_count": 153,
        "canonical_context_set": "main_v3",
        "full_context_set": "full_v3",
        "rubric_version": "2.0",
        "human_annotation_timing": "after_response_pool_freeze",
    }
    if stage == "api":
        result["experiment"]["preflight_required"] = True
        result["experiment"]["batch_optimization_status"] = (
            "recommended_before_paid_full_matrix_not_required_for_semantics"
        )
        for judge in result.get("judges", []):
            if isinstance(judge.get("name"), str):
                judge["name"] = judge["name"].replace("_v2", "_v3")
            judge["rubric_version"] = "2.0"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = {
        LOCAL_OUTPUT: build(_read(LOCAL_SOURCE), stage="local_gpu"),
        API_OUTPUT: build(_read(API_SOURCE), stage="api"),
    }
    rendered = {
        path: json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        for path, value in outputs.items()
    }
    if args.check:
        stale = [
            str(path)
            for path, text in rendered.items()
            if not path.exists() or path.read_text(encoding="utf-8") != text
        ]
        print(json.dumps({"ok": not stale, "stale": stale}, indent=2))
        return 0 if not stale else 1
    for path, text in rendered.items():
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"ok": True, "outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
