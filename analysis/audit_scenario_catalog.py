#!/usr/bin/env python3
"""Audit expansion structure, prompt lengths, leakage cues, and cross-motif similarity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_validation import validate_dataset
from scenario_expansion import load_benchmark_dataset
from scenario_metadata import enrich_dataset

LEAKAGE_PATTERNS = {
    "benchmark_instruction": re.compile(
        r"\b(benchmark|annotator|rubric|safe behavior)\b", re.I
    ),
    "target_answer_instruction": re.compile(
        r"\b(do not endorse|you must refuse|correct answer)\b", re.I
    ),
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _normalized(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def _prompt_rows(dataset: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for scenario in dataset.get("scenarios", []):
        if not scenario.get("source_packet_id"):
            continue
        if "user" in scenario:
            rows.append(
                {
                    "scenario_id": scenario["id"],
                    "motif_id": scenario["motif_id"],
                    "location": "user",
                    "text": scenario["user"],
                }
            )
        for turn in scenario.get("conversation", []):
            rows.append(
                {
                    "scenario_id": scenario["id"],
                    "motif_id": scenario["motif_id"],
                    "location": f"turn_{turn.get('turn')}",
                    "text": turn.get("user", ""),
                }
            )
    return rows


def build_report(
    dataset: dict[str, Any], similarity_threshold: float = 0.82
) -> dict[str, Any]:
    validation = validate_dataset(dataset, strict_metadata=True)
    rows = _prompt_rows(dataset)
    exact_seen: dict[str, dict[str, str]] = {}
    exact_duplicates = []
    length_violations = []
    leakage_hits = []

    for row in rows:
        normalized = _normalized(row["text"])
        word_count = len(normalized.split())
        lower, upper = (30, 180) if row["location"] == "user" else (8, 90)
        if not lower <= word_count <= upper:
            length_violations.append(
                {**row, "word_count": word_count, "expected": [lower, upper]}
            )
        if normalized in exact_seen:
            exact_duplicates.append({"first": exact_seen[normalized], "second": row})
        else:
            exact_seen[normalized] = row
        for label, pattern in LEAKAGE_PATTERNS.items():
            if pattern.search(row["text"]):
                leakage_hits.append({**row, "pattern": label})

    near_duplicates = []
    max_cross_motif_similarity = 0.0
    max_pair: dict[str, Any] | None = None
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left["motif_id"] == right["motif_id"]:
                continue
            ratio = SequenceMatcher(
                None,
                _normalized(left["text"]),
                _normalized(right["text"]),
                autojunk=False,
            ).ratio()
            pair = {
                "left": f"{left['scenario_id']}:{left['location']}",
                "right": f"{right['scenario_id']}:{right['location']}",
                "similarity": round(ratio, 4),
            }
            if ratio > max_cross_motif_similarity:
                max_cross_motif_similarity = ratio
                max_pair = pair
            if ratio >= similarity_threshold:
                near_duplicates.append(pair)

    scenario_ids = [scenario["id"] for scenario in dataset.get("scenarios", [])]
    expansion_ids = [
        scenario["id"]
        for scenario in dataset.get("scenarios", [])
        if scenario.get("source_packet_id")
    ]
    failures = []
    if validation.errors:
        failures.extend(validation.errors)
    if len(scenario_ids) != 108:
        failures.append(f"expected 108 total scenarios, found {len(scenario_ids)}")
    if len(expansion_ids) != 84:
        failures.append(f"expected 84 expansion scenarios, found {len(expansion_ids)}")
    if exact_duplicates:
        failures.append(f"found {len(exact_duplicates)} exact prompt duplicates")
    if near_duplicates:
        failures.append(
            f"found {len(near_duplicates)} cross-motif near-duplicate prompt pairs"
        )
    if length_violations:
        failures.append(f"found {len(length_violations)} prompt length violations")
    if leakage_hits:
        failures.append(f"found {len(leakage_hits)} target-answer leakage cues")

    return {
        "ok": not failures,
        "scenario_count": len(scenario_ids),
        "expansion_scenario_count": len(expansion_ids),
        "prompt_segment_count": len(rows),
        "similarity_threshold": similarity_threshold,
        "max_cross_motif_similarity": round(max_cross_motif_similarity, 4),
        "max_similarity_pair": max_pair,
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "length_violations": length_violations,
        "leakage_hits": leakage_hits,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("configs/scenario_expansion_v2.json"),
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    args = parser.parse_args()
    report = build_report(
        enrich_dataset(load_benchmark_dataset(args.dataset)),
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
