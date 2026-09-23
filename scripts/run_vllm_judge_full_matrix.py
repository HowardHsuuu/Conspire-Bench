#!/usr/bin/env python3
"""Rejudge the frozen 13,005-response matrix with the formal local judge panel.

The launcher never regenerates target conversations, never downloads model weights,
and serves exactly one cached judge at a time on a single GPU. Each completed stage is
resumable and verified before the next judge starts.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_vllm_judge_calibration_matrix import (
    ROOT,
    bundle_rows,
    calibrated_complete,
    health_ready,
    judge_config,
    model_name,
    read_json,
    run_logged,
    slug,
    start_server,
    stop_server,
    write_json,
)

EXPECTED_ROWS = 13_005
EXPECTED_MODELS = 17
EXPECTED_MOTIFS = 51
EXPECTED_JUDGES = 4
FORMAL_PROMPT_VARIANT = "v3_behavioral_anchors_evidence_first"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_source(path: Path) -> dict[str, int]:
    rows = bundle_rows(path)
    response_ids = {row.get("response_id") for row in rows}
    models = {row.get("model_name") for row in rows}
    motifs = {(row.get("scenario_metadata") or {}).get("motif_id") for row in rows}
    if len(rows) != EXPECTED_ROWS or len(response_ids) != EXPECTED_ROWS:
        raise ValueError(
            f"Frozen source must contain {EXPECTED_ROWS} unique responses; "
            f"found {len(rows)} rows and {len(response_ids)} IDs"
        )
    if len(models) != EXPECTED_MODELS:
        raise ValueError(
            f"Frozen source must contain {EXPECTED_MODELS} target models; "
            f"found {len(models)}"
        )
    if len(motifs) != EXPECTED_MOTIFS or None in motifs:
        raise ValueError(
            f"Frozen source must contain {EXPECTED_MOTIFS} identified motifs; "
            f"found {len(motifs)}"
        )
    if any(not row.get("conversation_log") for row in rows):
        raise ValueError("Frozen source contains a response without a conversation")
    if any(not row.get("generation_complete") for row in rows):
        raise ValueError("Frozen source contains an incomplete target generation")
    return {
        "row_count": len(rows),
        "model_count": len(models),
        "motif_count": len(motifs),
    }


def validate_final_panel(
    path: Path, judges: list[dict], *, prompt_variant: str
) -> dict[str, int]:
    rows = bundle_rows(path)
    expected_names = {model_name(judge) for judge in judges}
    successful = 0
    for row in rows:
        matches = [
            result
            for result in row.get("judge_results") or []
            if result.get("judge_name") in expected_names
            and result.get("judge_prompt_variant") == prompt_variant
            and result.get("error") is None
            and result.get("scores")
        ]
        if {result.get("judge_name") for result in matches} != expected_names:
            raise RuntimeError(
                f"Incomplete formal judge panel for {row.get('response_id')}"
            )
        successful += len(matches)
    expected_successes = EXPECTED_ROWS * EXPECTED_JUDGES
    if successful != expected_successes:
        raise RuntimeError(
            f"Expected {expected_successes} judgments; found {successful}"
        )
    return {
        "row_count": len(rows),
        "judge_count": len(expected_names),
        "successful_judgments": successful,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runner-python", type=Path, default=ROOT / ".local/venv-gptoss/bin/python"
    )
    parser.add_argument("--vllm", type=Path, default=ROOT / ".local/venv-vllm/bin/vllm")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--prompt-variant",
        choices=(FORMAL_PROMPT_VARIANT,),
        default=FORMAL_PROMPT_VARIANT,
    )
    args = parser.parse_args()
    if args.parallel < 1 or args.checkpoint_every < 1:
        raise ValueError("parallel and checkpoint-every must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in (args.source_bundle, args.config):
        if not path.is_file():
            raise FileNotFoundError(path)
    for executable in (args.runner_python, args.vllm):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Missing executable: {executable}")
    if health_ready(args.port):
        raise RuntimeError(f"Port {args.port} is already occupied")

    source_counts = validate_frozen_source(args.source_bundle)
    source_sha256 = sha256_file(args.source_bundle)
    base = read_json(args.config)
    judges = base.get("judges") or []
    if len(judges) != EXPECTED_JUDGES:
        raise ValueError(f"Formal matrix requires exactly {EXPECTED_JUDGES} judges")

    current = args.source_bundle
    for index, judge in enumerate(judges, start=1):
        identifier = slug(model_name(judge))
        output = args.output_dir / f"after_{index:02d}_{identifier}.json"
        if calibrated_complete(
            output,
            judge,
            prompt_variant=args.prompt_variant,
            expected_rows=EXPECTED_ROWS,
        ):
            print(f"formal judge already complete: {model_name(judge)}", flush=True)
            current = output
            continue

        generated_config = args.output_dir / f"config_{index:02d}_{identifier}.json"
        stage_config = judge_config(
            base,
            judge,
            port=args.port,
            parallel=args.parallel,
            prompt_variant=args.prompt_variant,
        )
        stage_config.setdefault("experiment", {})["stage"] = (
            f"formal_v3_judging_{identifier}"
        )
        write_json(generated_config, stage_config)

        server = start_server(
            judge,
            vllm=args.vllm,
            port=args.port,
            parallel=args.parallel,
            log_path=args.output_dir / f"server_{index:02d}_{identifier}.log",
        )
        try:
            run_logged(
                [
                    str(args.runner_python),
                    "scripts/run_judge_calibration.py",
                    "--config",
                    str(generated_config),
                    "--dataset",
                    "Conspire-Bench-v3.json",
                    "--input",
                    str(current),
                    "--output",
                    str(output),
                    "--status",
                    str(args.output_dir / f"status_{index:02d}_{identifier}.tsv"),
                    "--parallel",
                    str(args.parallel),
                    "--save-intermediate-every",
                    str(args.checkpoint_every),
                ],
                args.output_dir / f"driver_{index:02d}_{identifier}.log",
            )
        finally:
            stop_server(server)

        if not calibrated_complete(
            output,
            judge,
            prompt_variant=args.prompt_variant,
            expected_rows=EXPECTED_ROWS,
        ):
            raise RuntimeError(f"Formal judge incomplete for {model_name(judge)}")
        current = output
        print(f"formal judge complete: {model_name(judge)}", flush=True)

    final = args.output_dir / "judge_v3_full_calibrated.json"
    temporary = final.with_suffix(final.suffix + f".tmp-{os.getpid()}")
    shutil.copyfile(current, temporary)
    temporary.replace(final)
    final_counts = validate_final_panel(
        final, judges, prompt_variant=args.prompt_variant
    )
    completion = args.output_dir / "judge_v3_full.completed.json"
    write_json(
        completion,
        {
            "ok": True,
            "prompt_variant": args.prompt_variant,
            **source_counts,
            **final_counts,
            "source_bundle": str(args.source_bundle),
            "source_sha256": source_sha256,
            "calibrated": str(final),
            "calibrated_sha256": sha256_file(final),
        },
    )
    print(f"FORMAL V3 JUDGING COMPLETE: {completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
