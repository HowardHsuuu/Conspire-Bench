#!/usr/bin/env python3
"""Judge the frozen 180-response Gemini replication with the local four-model panel.

This launcher never regenerates target conversations and never calls the Gemini API.
It serves one already-cached judge at a time, checkpoints each stage, and preserves the
Gemini self-judge result as a same-family sensitivity analysis while making the four
open-model judges the primary cross-family aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vllm_judge_calibration_matrix import (
    bundle_rows,
    health_ready,
    local_snapshot_path,
    model_name,
    read_json,
    slug,
    write_json,
)
from scripts.run_vllm_paraphrase_robustness import ensure_idle_gpu, judge_one_stage

EXPECTED_ROWS = 180
EXPECTED_SCENARIOS = 36
EXPECTED_MOTIFS = 12
EXPECTED_VARIANTS = 5
EXPECTED_JUDGE_FAMILIES = {"qwen", "llama", "gemma", "gpt_oss"}
FORMAL_PROMPT_VARIANT = "v3_behavioral_anchors_evidence_first"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    rows = bundle_rows(path)
    response_ids = [str(row.get("response_id") or "") for row in rows]
    scenario_ids = {str(row.get("scenario_id") or "") for row in rows}
    motif_ids = {
        str((row.get("scenario_metadata") or {}).get("motif_id") or "")
        for row in rows
    }
    variants = {str(row.get("prompt_variant_id") or "") for row in rows}
    targets = {str(row.get("model_name") or "") for row in rows}
    target_families = {str(row.get("model_family") or "") for row in rows}
    cells = Counter(
        (str(row.get("scenario_id") or ""), str(row.get("prompt_variant_id") or ""))
        for row in rows
    )

    errors: list[str] = []
    if len(rows) != EXPECTED_ROWS:
        errors.append(f"expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if not all(response_ids) or len(set(response_ids)) != EXPECTED_ROWS:
        errors.append("response IDs are missing or not unique")
    if "" in scenario_ids or len(scenario_ids) != EXPECTED_SCENARIOS:
        errors.append(
            f"expected {EXPECTED_SCENARIOS} scenarios, found {len(scenario_ids)}"
        )
    if "" in motif_ids or len(motif_ids) != EXPECTED_MOTIFS:
        errors.append(f"expected {EXPECTED_MOTIFS} motifs, found {len(motif_ids)}")
    if "" in variants or len(variants) != EXPECTED_VARIANTS:
        errors.append(
            f"expected {EXPECTED_VARIANTS} prompt variants, found {len(variants)}"
        )
    if len(targets) != 1 or target_families != {"gemini"}:
        errors.append(
            f"expected one Gemini target, found targets={sorted(targets)!r}, "
            f"families={sorted(target_families)!r}"
        )
    if len(cells) != EXPECTED_ROWS or any(count != 1 for count in cells.values()):
        errors.append("scenario-by-variant cells are incomplete or duplicated")
    if any(row.get("error") and not row.get("generation_complete") for row in rows):
        errors.append("at least one target generation has an error")
    if any(not row.get("generation_complete") for row in rows):
        errors.append("at least one target generation is incomplete")
    if any(not row.get("conversation_log") for row in rows):
        errors.append("at least one row has no conversation log")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "row_count": len(rows),
        "scenario_count": len(scenario_ids),
        "motif_count": len(motif_ids),
        "variant_count": len(variants),
        "target_models": sorted(targets),
        "source_sha256": actual_sha256,
    }


def validate_judges(config: dict[str, Any]) -> list[dict[str, Any]]:
    judges = list(config.get("judges") or [])
    families = {str(judge.get("model_family") or "") for judge in judges}
    prompt_variants = {
        str(judge.get("judge_prompt_variant") or "") for judge in judges
    }
    names = [model_name(judge) for judge in judges]
    if len(judges) != 4 or len(set(names)) != 4:
        raise ValueError("Exactly four unique local judges are required")
    if families != EXPECTED_JUDGE_FAMILIES:
        raise ValueError(f"Unexpected judge families: {sorted(families)!r}")
    if prompt_variants != {FORMAL_PROMPT_VARIANT}:
        raise ValueError(f"All judges must use {FORMAL_PROMPT_VARIANT}")
    return judges


def validate_final(path: Path, judges: list[dict[str, Any]]) -> dict[str, int]:
    rows = bundle_rows(path)
    expected_names = {model_name(judge) for judge in judges}
    successful = 0
    retained_same_family = 0
    for row in rows:
        response_id = row.get("response_id")
        results = row.get("judge_results") or []
        cross_family = [
            result
            for result in results
            if result.get("judge_name") in expected_names
            and result.get("judge_prompt_variant") == FORMAL_PROMPT_VARIANT
            and result.get("error") is None
            and result.get("scores")
        ]
        if {result.get("judge_name") for result in cross_family} != expected_names:
            raise ValueError(f"Incomplete cross-family panel for {response_id}")
        if any(result.get("same_family_as_target") for result in cross_family):
            raise ValueError(f"Open judge marked same-family for {response_id}")
        same_family = [
            result
            for result in results
            if result.get("same_family_as_target")
            and result.get("error") is None
            and result.get("scores")
        ]
        if len(same_family) != 1:
            raise ValueError(
                f"Expected one retained Gemini self-judge result for {response_id}"
            )
        primary_names = set(row.get("primary_judge_names") or [])
        if primary_names != expected_names:
            raise ValueError(f"Primary judge set mismatch for {response_id}")
        if (
            row.get("error") is not None
            or not row.get("evaluation_complete")
            or not row.get("judge_scores")
        ):
            raise ValueError(f"Primary aggregate is incomplete for {response_id}")
        successful += len(cross_family)
        retained_same_family += len(same_family)
    expected_successes = EXPECTED_ROWS * len(expected_names)
    if len(rows) != EXPECTED_ROWS or successful != expected_successes:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows and {expected_successes} cross-family "
            f"judgments; found {len(rows)} and {successful}"
        )
    return {
        "row_count": len(rows),
        "judge_count": len(expected_names),
        "successful_cross_family_judgments": successful,
        "retained_same_family_judgments": retained_same_family,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_v3_paraphrase_robustness.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runner-python", type=Path, default=ROOT / ".local/venv-gptoss/bin/python"
    )
    parser.add_argument("--vllm", type=Path, default=ROOT / ".local/venv-vllm/bin/vllm")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    for path in (args.source_bundle, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_counts = validate_source(
        args.source_bundle, expected_sha256=args.expected_source_sha256
    )
    config = read_json(args.config)
    judges = validate_judges(config)
    plan = {
        **source_counts,
        "judge_names": [model_name(judge) for judge in judges],
        "expected_cross_family_judgments": EXPECTED_ROWS * len(judges),
        "prompt_variant": FORMAL_PROMPT_VARIANT,
    }
    print(plan, flush=True)
    if args.plan_only:
        print("PLAN ONLY: no model was loaded and no output was written", flush=True)
        return 0

    if args.parallel < 1:
        raise ValueError("parallel must be positive")
    for executable in (args.runner_python, args.vllm):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Missing executable: {executable}")
    for judge in judges:
        local_snapshot_path(str(judge["model"]))
    if health_ready(args.port):
        raise RuntimeError(f"Port {args.port} is already occupied")
    ensure_idle_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    current = args.source_bundle
    for index, judge in enumerate(judges, start=1):
        output = args.output_dir / f"after_{index:02d}_{slug(model_name(judge))}.json"
        current = judge_one_stage(
            base=config,
            judge=judge,
            source=current,
            output=output,
            output_dir=args.output_dir,
            runner_python=args.runner_python,
            vllm=args.vllm,
            port=args.port,
            parallel=args.parallel,
            expected_rows=EXPECTED_ROWS,
        )

    final = args.output_dir / "gemini_cross_family_judged.json"
    temporary = final.with_suffix(final.suffix + f".tmp-{os.getpid()}")
    shutil.copyfile(current, temporary)
    temporary.replace(final)
    final_counts = validate_final(final, judges)
    completion = args.output_dir / "gemini_cross_family_judged.completed.json"
    write_json(
        completion,
        {
            "ok": True,
            **source_counts,
            **final_counts,
            "judge_names": [model_name(judge) for judge in judges],
            "prompt_variant": FORMAL_PROMPT_VARIANT,
            "source_bundle": str(args.source_bundle),
            "final_bundle": str(final),
            "final_sha256": sha256_file(final),
        },
    )
    print(f"GEMINI CROSS-FAMILY JUDGING COMPLETE: {completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
