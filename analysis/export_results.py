#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


NUMERIC_METRICS = [
    "conspiracy_amplification",
    "reality_grounding",
    "appropriate_uncertainty",
    "overall_safety",
]
BOOLEAN_METRICS = ["harmful_elaboration"]
ALL_METRICS = NUMERIC_METRICS + BOOLEAN_METRICS


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: Iterable[Any]) -> str:
    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    if not clean:
        return ""
    return f"{mean(clean):.4f}"


def _rate(values: Iterable[Any]) -> str:
    clean = [bool(value) for value in values if value is not None and value != ""]
    if not clean:
        return ""
    return f"{sum(clean) / len(clean):.4f}"


def _load_results(path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {}, data
    return data.get("metadata", {}), data.get("detailed_results", [])


def _judge_count(result: Dict[str, Any]) -> int:
    return len(result.get("judge_results") or [])


def _judge_error_count(result: Dict[str, Any]) -> int:
    return sum(1 for judge in result.get("judge_results") or [] if judge.get("error"))


def _joined_judges(result: Dict[str, Any]) -> str:
    names = [judge.get("judge_name", "") for judge in result.get("judge_results") or []]
    return " | ".join(name for name in names if name)


def scenario_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        scores = result.get("judge_scores") or {}
        disagreement = result.get("judge_disagreement") or {}
        row = {
            "scenario_id": result.get("scenario_id", ""),
            "model_name": result.get("model_name") or result.get("target_model", ""),
            "category": result.get("category", ""),
            "subcategory": result.get("scenario_subcategory", ""),
            "scenario_type": result.get("scenario_type", ""),
            "context_setting": result.get("context_setting", ""),
            "error": result.get("error") or "",
            "judge_count": _judge_count(result),
            "judge_error_count": _judge_error_count(result),
            "judges": _joined_judges(result),
            "total_safety_score": result.get("total_safety_score", ""),
        }
        for metric in ALL_METRICS:
            row[metric] = scores.get(metric, "")
            row[f"disagreement_{metric}"] = disagreement.get(metric, "")
        rows.append(row)
    return rows


def judge_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        for judge in result.get("judge_results") or []:
            scores = judge.get("scores") or {}
            row = {
                "scenario_id": result.get("scenario_id", ""),
                "model_name": result.get("model_name") or result.get("target_model", ""),
                "category": result.get("category", ""),
                "subcategory": result.get("scenario_subcategory", ""),
                "scenario_type": result.get("scenario_type", ""),
                "judge_name": judge.get("judge_name", ""),
                "judge_provider": judge.get("provider", ""),
                "judge_model": judge.get("model", ""),
                "error": judge.get("error") or "",
            }
            for metric in ALL_METRICS:
                row[metric] = scores.get(metric, "")
            rows.append(row)
    return rows


def model_summary_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        model_name = result.get("model_name") or result.get("target_model", "")
        grouped[model_name].append(result)

    rows = []
    for model_name, model_results in sorted(grouped.items()):
        valid = [result for result in model_results if not result.get("error")]
        row = {
            "model_name": model_name,
            "completed": len(valid),
            "failed": len(model_results) - len(valid),
            "avg_total_safety_score": _avg(result.get("total_safety_score") for result in valid),
        }
        for metric in NUMERIC_METRICS:
            row[f"avg_{metric}"] = _avg(
                (result.get("judge_scores") or {}).get(metric)
                for result in valid
            )
            row[f"avg_disagreement_{metric}"] = _avg(
                (result.get("judge_disagreement") or {}).get(metric)
                for result in valid
            )
        row["harmful_elaboration_rate"] = _rate(
            (result.get("judge_scores") or {}).get("harmful_elaboration")
            for result in valid
        )
        row["harmful_elaboration_disagreement_rate"] = _rate(
            (result.get("judge_disagreement") or {}).get("harmful_elaboration")
            for result in valid
        )
        rows.append(row)
    return rows


def interesting_case_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for row in scenario_rows(results):
        total = _safe_float(row.get("total_safety_score"))
        max_disagreement = max(
            [
                _safe_float(row.get(f"disagreement_{metric}")) or 0.0
                for metric in ALL_METRICS
            ],
            default=0.0,
        )
        harmful = bool(row.get("harmful_elaboration")) and row.get("harmful_elaboration") != "False"
        if row.get("error") or (total is not None and total <= 2.5) or max_disagreement >= 2.0 or harmful:
            row["max_disagreement"] = f"{max_disagreement:.4f}"
            rows.append(row)

    return sorted(
        rows,
        key=lambda item: (
            item.get("error") == "",
            -(_safe_float(item.get("max_disagreement")) or 0.0),
            _safe_float(item.get("total_safety_score")) or 99.0,
        ),
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results(input_path: Path, out_dir: Path) -> List[Path]:
    _, results = _load_results(input_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "model_summary.csv": model_summary_rows(results),
        "scenario_results.csv": scenario_rows(results),
        "judge_scores.csv": judge_rows(results),
        "interesting_cases.csv": interesting_case_rows(results),
    }

    written = []
    for filename, rows in outputs.items():
        path = out_dir / filename
        write_csv(path, rows)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Conspire-Bench result JSON into paper-friendly CSV tables."
    )
    parser.add_argument("input", type=Path, help="Path to benchmark_results.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Default: <result-dir>/paper_tables",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = args.out_dir or args.input.parent / "paper_tables"
    written = export_results(args.input, out_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
