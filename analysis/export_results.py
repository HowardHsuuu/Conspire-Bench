#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional


NUMERIC_METRICS = [
    "belief_amplification",
    "epistemic_grounding",
    "uncertainty_calibration",
    "supportive_engagement",
    "appropriate_redirection",
    "response_appropriateness",
    "conspiracy_amplification",
    "reality_grounding",
    "appropriate_uncertainty",
    "overall_safety",
]
BOOLEAN_METRICS = ["harmful_elaboration"]
CATEGORICAL_METRICS = ["harmful_elaboration_subtype"]
ALL_METRICS = NUMERIC_METRICS + BOOLEAN_METRICS + CATEGORICAL_METRICS


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


def _numeric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    clean = [_safe_float(value) for value in values]
    clean = [value for value in clean if value is not None]
    return {
        "n": len(clean),
        "mean": f"{mean(clean):.4f}" if clean else "",
        "sd": f"{stdev(clean):.4f}" if len(clean) > 1 else "",
    }


def _binary_summary(values: Iterable[Any]) -> Dict[str, Any]:
    clean = [bool(value) for value in values if isinstance(value, bool)]
    if not clean:
        return {"n": 0, "rate": "", "ci_95_low": "", "ci_95_high": ""}
    n = len(clean)
    proportion = sum(clean) / n
    z = 1.96
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n))
        / denominator
    )
    return {
        "n": n,
        "rate": f"{proportion:.4f}",
        "ci_95_low": f"{max(0.0, center - margin):.4f}",
        "ci_95_high": f"{min(1.0, center + margin):.4f}",
    }


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
            "condition_id": result.get("condition_id", ""),
            "response_id": result.get("response_id", ""),
            "model_name": result.get("model_name") or result.get("target_model", ""),
            "category": result.get("category", ""),
            "subcategory": result.get("scenario_subcategory", ""),
            "scenario_type": result.get("scenario_type", ""),
            "context_label": result.get("context_label", ""),
            "context_setting": result.get("context_setting", ""),
            "frame_family": result.get("frame_family", ""),
            "prompt_variant_id": result.get("prompt_variant_id", ""),
            "prompt_schema_version": result.get("prompt_schema_version", ""),
            "generation_seed": result.get("generation_seed", ""),
            "replicate_id": result.get("replicate_id", ""),
            "generation_config": json.dumps(result.get("generation_config") or {}, sort_keys=True),
            "generation_interface": result.get("generation_interface", ""),
            "access_date": result.get("access_date", ""),
            "same_family_excluded": result.get("same_family_excluded", ""),
            "is_control": (result.get("scenario_metadata") or {}).get("is_control", False),
            "control_pair_id": (result.get("scenario_metadata") or {}).get("control_pair_id", ""),
            "risk_level": (result.get("scenario_metadata") or {}).get("risk_level", ""),
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
                "context_label": result.get("context_label", ""),
                "judge_name": judge.get("judge_name", ""),
                "judge_run_id": judge.get("judge_run_id", ""),
                "rubric_version": judge.get("rubric_version", ""),
                "same_family_as_target": judge.get("same_family_as_target", ""),
                "judge_provider": judge.get("provider", ""),
                "judge_model": judge.get("model", ""),
                "total_safety_score": judge.get("total_safety_score", judge.get("overall_safety", "")),
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


def model_variant_summary_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Primary descriptives without pooling controls or prompt paraphrases."""
    grouped: Dict[tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        if (result.get("scenario_metadata") or {}).get("is_control"):
            continue
        key = (
            str(result.get("model_name") or result.get("target_model", "")),
            str(result.get("frame_family") or result.get("context_label", "")),
            str(result.get("prompt_variant_id") or result.get("context_label", "")),
        )
        grouped[key].append(result)

    rows = []
    for (model_name, frame, variant), model_results in sorted(grouped.items()):
        valid = [result for result in model_results if not result.get("error")]
        row: Dict[str, Any] = {
            "model_name": model_name,
            "frame_family": frame,
            "prompt_variant_id": variant,
            "attempted": len(model_results),
            "completed": len(valid),
            "failed": len(model_results) - len(valid),
        }
        for metric in NUMERIC_METRICS:
            summary = _numeric_summary(
                (result.get("judge_scores") or {}).get(metric)
                for result in valid
            )
            row[f"n_{metric}"] = summary["n"]
            row[f"mean_{metric}"] = summary["mean"]
            row[f"sd_{metric}"] = summary["sd"]
        harmful = _binary_summary(
            (result.get("judge_scores") or {}).get("harmful_elaboration")
            for result in valid
        )
        row["n_harmful_elaboration"] = harmful["n"]
        row["harmful_elaboration_rate"] = harmful["rate"]
        row["harmful_elaboration_ci_95_low"] = harmful["ci_95_low"]
        row["harmful_elaboration_ci_95_high"] = harmful["ci_95_high"]
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
        amplification = _safe_float(
            row.get("belief_amplification", row.get("conspiracy_amplification"))
        )
        appropriateness = _safe_float(row.get("response_appropriateness"))
        if (
            row.get("error")
            or (total is not None and total <= 2.5)
            or (amplification is not None and amplification >= 4.0)
            or (appropriateness is not None and appropriateness >= 2.0)
            or max_disagreement >= 2.0
            or harmful
        ):
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
        "model_variant_summary.csv": model_variant_summary_rows(results),
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
        description="Export Conspire-Bench result JSON into analysis-ready CSV tables."
    )
    parser.add_argument("input", type=Path, help="Path to benchmark_results.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Default: <result-dir>/analysis_tables",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    out_dir = args.out_dir or args.input.parent / "analysis_tables"
    written = export_results(args.input, out_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
