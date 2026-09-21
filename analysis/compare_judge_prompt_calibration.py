#!/usr/bin/env python3
"""Compare original and calibrated judge prompts on the same transcripts."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_v2 import RUBRIC_DIMENSIONS

FAMILIES = ("qwen", "llama", "gemma", "gpt_oss")
METRICS = tuple(RUBRIC_DIMENSIONS)
TEMPLATE = (1, 1, 1, False, 1, None, 0)
IDEAL = (1, 5, 5, False, 5, None, 0)
POSITIVE_GROUNDING = re.compile(
    r"\b(clearly separates|clearly distinguishes|meticulously separates|strong grounding)\b",
    re.I,
)
PLACEHOLDERS = {
    "brief evidence",
    "brief evidence or not applicable",
    "brief global assessment",
}


def rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(value, list):
        raise ValueError(f"No detailed_results in {path}")
    return value


def judges(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(result.get("judge_model_family")): result
        for result in row.get("judge_results") or []
    }


def vector(result: dict[str, Any]) -> tuple[Any, ...]:
    scores = result.get("scores") or {}
    return tuple(scores.get(metric) for metric in METRICS)


def entropy(values: list[Any]) -> float:
    counts = Counter(map(str, values))
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def tension(result: dict[str, Any]) -> bool:
    scores = result.get("scores") or {}
    reasoning = result.get("reasoning") or {}
    return scores.get("epistemic_grounding") == 1 and bool(
        POSITIVE_GROUNDING.search(str(reasoning.get("epistemic_grounding") or ""))
    )


def all_placeholders(result: dict[str, Any]) -> bool:
    reasoning = result.get("reasoning") or {}
    normalized = {str(value).strip().lower() for value in reasoning.values()}
    return bool(normalized) and normalized <= PLACEHOLDERS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("calibrated", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = {str(row["response_id"]): row for row in rows(args.baseline)}
    calibrated = {str(row["response_id"]): row for row in rows(args.calibrated)}
    if set(baseline) != set(calibrated) or len(baseline) != 120:
        raise ValueError("Baseline and calibrated bundles must share 120 response IDs")

    report: dict[str, Any] = {
        "schema_version": "1.0",
        "interpretation": (
            "Diagnostic comparison only. Greater score diversity or fewer automated "
            "score-rationale flags does not establish agreement with expert humans."
        ),
        "row_count": len(baseline),
        "judges": [],
    }
    for family in FAMILIES:
        old_results, new_results = [], []
        for response_id in sorted(baseline):
            old = judges(baseline[response_id]).get(family)
            new = judges(calibrated[response_id]).get(family)
            if not old or not new or old.get("error") or new.get("error"):
                raise ValueError(f"Missing successful {family} pair for {response_id}")
            old_results.append(old)
            new_results.append(new)
        old_vectors = [vector(result) for result in old_results]
        new_vectors = [vector(result) for result in new_results]
        metrics = []
        for index, metric in enumerate(METRICS):
            pairs = [
                (old[index], new[index])
                for old, new in zip(old_vectors, new_vectors, strict=True)
                if old[index] is not None and new[index] is not None
            ]
            numeric = [
                (float(old), float(new))
                for old, new in pairs
                if isinstance(old, (int, float)) and isinstance(new, (int, float))
            ]
            metrics.append(
                {
                    "metric": metric,
                    "paired_nonmissing": len(pairs),
                    "exact_agreement": (
                        sum(old == new for old, new in pairs) / len(pairs)
                        if pairs
                        else None
                    ),
                    "mean_calibrated_minus_original": (
                        mean(new - old for old, new in numeric) if numeric else None
                    ),
                    "original_entropy_bits": entropy([old for old, _ in pairs])
                    if pairs
                    else None,
                    "calibrated_entropy_bits": entropy([new for _, new in pairs])
                    if pairs
                    else None,
                    "original_distribution": dict(
                        Counter(map(str, (old for old, _ in pairs)))
                    ),
                    "calibrated_distribution": dict(
                        Counter(map(str, (new for _, new in pairs)))
                    ),
                }
            )
        old_counts = Counter(old_vectors)
        new_counts = Counter(new_vectors)
        report["judges"].append(
            {
                "judge_family": family,
                "prompt_variant": new_results[0].get("judge_prompt_variant"),
                "changed_score_vectors": sum(
                    old != new
                    for old, new in zip(old_vectors, new_vectors, strict=True)
                ),
                "original_unique_score_vectors": len(old_counts),
                "calibrated_unique_score_vectors": len(new_counts),
                "original_template_vector_count": old_counts[TEMPLATE],
                "calibrated_template_vector_count": new_counts[TEMPLATE],
                "original_ideal_vector_count": old_counts[IDEAL],
                "calibrated_ideal_vector_count": new_counts[IDEAL],
                "original_grounding_reason_tension_count": sum(
                    map(tension, old_results)
                ),
                "calibrated_grounding_reason_tension_count": sum(
                    map(tension, new_results)
                ),
                "original_all_placeholder_rationale_count": sum(
                    map(all_placeholders, old_results)
                ),
                "calibrated_all_placeholder_rationale_count": sum(
                    map(all_placeholders, new_results)
                ),
                "metrics": metrics,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
