#!/usr/bin/env python3
"""Rejudge a frozen conversation subset with one configured judge."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_runner import ConspireBenchmarkRunner


def rows_from(payload: object) -> list[dict]:
    rows = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Input bundle has no detailed_results list")
    return rows


def incomplete_result_reason(
    result: dict, *, judge_run_id: str, prompt_variant: str
) -> str | None:
    """Return a precise reason when a calibrated judgment is incomplete."""

    if result.get("error") is not None:
        return str(result.get("error")).strip() or "judge error (empty message)"
    if not result.get("scores"):
        return "missing scores"
    if result.get("judge_run_id") != judge_run_id:
        return "judge_run_id mismatch"
    if result.get("judge_prompt_variant") != prompt_variant:
        return "judge_prompt_variant mismatch"
    return None


async def run(args: argparse.Namespace) -> None:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source_rows = rows_from(payload)
    checkpoint = args.output.with_name(f"temp_{args.output.name}")
    if checkpoint.is_file():
        rows = rows_from(json.loads(checkpoint.read_text(encoding="utf-8")))
        source_ids = {row.get("response_id") for row in source_rows}
        checkpoint_ids = {row.get("response_id") for row in rows}
        if len(rows) != len(source_rows) or checkpoint_ids != source_ids:
            raise ValueError("Calibration checkpoint does not match the input bundle")
    else:
        rows = source_rows
    if len({row.get("response_id") for row in rows}) != len(rows):
        raise ValueError("Calibration input has missing or duplicate response IDs")
    if any(not row.get("conversation_log") for row in rows):
        raise ValueError("Calibration input contains a row without a conversation")

    runner = ConspireBenchmarkRunner(str(args.config), str(args.dataset))
    judges = runner._get_judge_configs()
    if len(judges) != 1:
        raise ValueError("Calibration config must contain exactly one judge")
    judge = judges[0]
    prompt_variant = str(judge.get("judge_prompt_variant", "v2_original"))
    if prompt_variant == "v2_original":
        raise ValueError("Calibration must use a non-default judge_prompt_variant")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner.results_dir = str(args.output.parent)
    runner.config.setdefault("evaluation", {})["save_intermediate_results"] = True
    runner._initialize_status_file(str(args.status))
    await runner._run_judge_for_results(
        all_results=rows,
        judge_config=judge,
        parallel_judgements=max(1, args.parallel),
        save_intermediate=True,
        save_intermediate_every=max(1, args.save_intermediate_every),
        output_file=args.output.name,
        status_file=str(args.status),
    )

    judge_name = runner._judge_name(judge)
    judge_run_id = runner._judge_run_id(judge)
    for row in rows:
        matches = [
            result
            for result in row.get("judge_results") or []
            if result.get("judge_name") == judge_name
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {judge_name} result for {row.get('response_id')}"
            )
        result = matches[0]
        incomplete_reason = incomplete_result_reason(
            result,
            judge_run_id=judge_run_id,
            prompt_variant=prompt_variant,
        )
        if incomplete_reason is not None:
            raise RuntimeError(
                f"Incomplete calibrated result for {row.get('response_id')}: "
                f"{incomplete_reason}"
            )

    source_metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    output = {
        "metadata": {
            **(source_metadata or {}),
            "judge_calibration_stage": {
                "input": str(args.input),
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "judge_name": judge_name,
                "judge_run_id": judge_run_id,
                "judge_prompt_variant": prompt_variant,
                "row_count": len(rows),
            },
        },
        "summary": runner._generate_summary(rows),
        "detailed_results": rows,
    }
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(args.output)
    checkpoint.unlink(missing_ok=True)
    print(args.output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("Conspire-Bench-v3.json"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument(
        "--save-intermediate-every",
        type=int,
        default=5,
        help="Write a resumable checkpoint after this many completed judgments.",
    )
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
