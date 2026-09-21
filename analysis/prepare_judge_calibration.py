#!/usr/bin/env python3
"""Select a frozen 120-conversation judge-prompt calibration subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

FRAMES = (
    "neutral",
    "brainstorming",
    "critical_review",
    "sensemaking",
    "supportive_listening",
)
STRUCTURES = (
    "single_turn_complete_logic",
    "multi_turn_progression",
    "complete_logic_then_resistance",
)
FAMILIES = ("qwen", "llama", "gemma", "gpt_oss")
POSITIVE_GROUNDING = (
    "clearly separates",
    "clearly distinguishes",
    "meticulously separates",
    "strong grounding",
)
PLACEHOLDERS = {
    "brief evidence",
    "brief evidence or not applicable",
    "brief global assessment",
}


def read_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Input does not contain detailed_results")
    return payload if isinstance(payload, dict) else {}, rows


def stable_rank(seed: int, *parts: str) -> str:
    return hashlib.sha256("|".join((str(seed), *parts)).encode("utf-8")).hexdigest()


def diagnostic_judges(rows: list[dict[str, Any]]) -> set[str]:
    judges = set()
    for row in rows:
        for result in row.get("judge_results") or []:
            family = str(result.get("judge_model_family") or "")
            scores = result.get("scores") or {}
            reasoning = result.get("reasoning") or {}
            grounding_reason = str(reasoning.get("epistemic_grounding") or "").lower()
            score_reason_tension = scores.get("epistemic_grounding") == 1 and any(
                marker in grounding_reason for marker in POSITIVE_GROUNDING
            )
            normalized = {str(value).strip().lower() for value in reasoning.values()}
            placeholder = bool(normalized) and normalized <= PLACEHOLDERS
            if family in FAMILIES and (score_reason_tension or placeholder):
                judges.add(family)
    return judges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument(
        "--exclude-bundle",
        type=Path,
        help="Exclude every motif present in another calibration bundle.",
    )
    parser.add_argument(
        "--outcome-blind",
        action="store_true",
        help="Disable diagnostic-output enrichment for every stratum.",
    )
    args = parser.parse_args()

    source, rows = read_rows(args.bundle)
    excluded_motifs: set[str] = set()
    if args.exclude_bundle:
        _, excluded_rows = read_rows(args.exclude_bundle)
        excluded_motifs = {
            str((row.get("scenario_metadata") or {}).get("motif_id") or "")
            for row in excluded_rows
        }
        excluded_motifs.discard("")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_configs = {
        f"{model['provider']}/{model['model']}": model for model in config["models"]
    }
    by_family: dict[str, list[str]] = defaultdict(list)
    for name, model in model_configs.items():
        by_family[str(model["model_family"])].append(name)
    endpoints = {}
    for family in FAMILIES:
        models = sorted(
            by_family[family],
            key=lambda name: (float(model_configs[name]["parameter_scale_b"]), name),
        )
        endpoints[family] = (models[0], models[-1])

    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model = str(row.get("model_name"))
        if model not in model_configs:
            continue
        motif = str((row.get("scenario_metadata") or {}).get("motif_id") or "")
        structure = str(row.get("scenario_type") or "")
        frame = str(row.get("frame_family") or "")
        if motif and structure in STRUCTURES and frame in FRAMES:
            if frame in groups[(model, motif, structure)]:
                raise ValueError(
                    f"Duplicate frame in group: {model}, {motif}, {structure}"
                )
            groups[(model, motif, structure)][frame] = row

    selected = []
    used_motifs: set[str] = set()
    diagnostic_assignment = {family: family for family in FAMILIES}
    for family in FAMILIES:
        for endpoint_index, model in enumerate(endpoints[family]):
            tier = "small_endpoint" if endpoint_index == 0 else "large_endpoint"
            for structure_index, structure in enumerate(STRUCTURES):
                stratum = f"{family}:{tier}:{structure}"
                candidates = [
                    (motif, frame_rows)
                    for (
                        candidate_model,
                        motif,
                        candidate_structure,
                    ), frame_rows in groups.items()
                    if candidate_model == model
                    and candidate_structure == structure
                    and motif not in excluded_motifs
                    and set(frame_rows) == set(FRAMES)
                ]
                if not candidates:
                    raise ValueError(f"No complete candidate for {stratum}")

                requested_diagnostic = (
                    not args.outcome_blind
                    and endpoint_index == 0
                    and structure_index == 0
                )
                diagnostic_family = diagnostic_assignment[family]
                ranked = []
                for motif, frame_rows in candidates:
                    values = list(frame_rows.values())
                    flags = set().union(*(diagnostic_judges([row]) for row in values))
                    category = str(values[0].get("category") or "")
                    priority = (
                        0
                        if requested_diagnostic and diagnostic_family in flags
                        else 1
                        if requested_diagnostic and flags
                        else 2
                        if requested_diagnostic
                        else 0,
                        0 if motif not in used_motifs else 1,
                        stable_rank(args.seed, stratum, category, motif),
                    )
                    ranked.append((priority, motif, frame_rows, flags))
                _, motif, frame_rows, flags = min(ranked, key=lambda item: item[0])
                used_motifs.add(motif)
                selection_class = (
                    "diagnostic_enriched"
                    if requested_diagnostic and flags
                    else "outcome_blind_stratified"
                )
                selected.append(
                    {
                        "family": family,
                        "tier": tier,
                        "model": model,
                        "structure": structure,
                        "motif": motif,
                        "selection_class": selection_class,
                        "diagnostic_judges_present": sorted(flags),
                        "rows": [frame_rows[frame] for frame in FRAMES],
                    }
                )

    output_rows = []
    for group_index, group in enumerate(selected):
        group_id = f"calibration_group_{group_index + 1:02d}"
        for row in group.pop("rows"):
            copied = deepcopy(row)
            copied["calibration_sampling"] = {
                "group_id": group_id,
                "selection_class": group["selection_class"],
                "stratum": {
                    key: group[key]
                    for key in ("family", "tier", "model", "structure", "motif")
                },
                "selection_seed": args.seed,
            }
            output_rows.append(copied)
        group["group_id"] = group_id

    counts = {
        "rows": len(output_rows),
        "groups": len(selected),
        "frames": dict(Counter(row["frame_family"] for row in output_rows)),
        "structures": dict(Counter(row["scenario_type"] for row in output_rows)),
        "target_families": dict(
            Counter(
                str(model_configs[row["model_name"]]["model_family"])
                for row in output_rows
            )
        ),
        "selection_classes": dict(
            Counter(
                row["calibration_sampling"]["selection_class"] for row in output_rows
            )
        ),
        "unique_motifs": len(
            {row["scenario_metadata"]["motif_id"] for row in output_rows}
        ),
    }
    assert counts["rows"] == 120 and counts["groups"] == 24
    assert counts["frames"] == {frame: 24 for frame in FRAMES}
    assert counts["structures"] == {structure: 40 for structure in STRUCTURES}
    assert counts["target_families"] == {family: 30 for family in FAMILIES}
    assert len({row["response_id"] for row in output_rows}) == 120

    manifest = {
        "schema_version": "1.0",
        "purpose": "Judge-prompt calibration; not a replacement human-validity sample.",
        "source_bundle": str(args.bundle),
        "source_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "selection_seed": args.seed,
        "outcome_blind": args.outcome_blind,
        "excluded_bundle": str(args.exclude_bundle) if args.exclude_bundle else None,
        "excluded_motifs": sorted(excluded_motifs),
        "design": (
            "Two parameter-scale endpoints per target family by three interaction "
            "structures; every selected group contains all five canonical frames. "
            + (
                "All choices use a seeded hash independent of scores."
                if args.outcome_blind
                else "One small/single-turn stratum per family preferentially includes a "
                "pre-existing diagnostic judge-output flag; all other choices use a "
                "seeded hash independent of scores."
            )
        ),
        "counts": counts,
        "groups": selected,
    }
    payload = {
        "metadata": {
            **(source.get("metadata") or {}),
            "calibration_sampling": manifest,
        },
        "summary": source.get("summary") or {},
        "detailed_results": output_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    manifest_path = args.manifest or args.output.with_name(
        args.output.stem + ".manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    print(args.output)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
