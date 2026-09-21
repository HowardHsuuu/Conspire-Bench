#!/usr/bin/env python3
"""Run all configured judges on the grounding/uncertainty 2x2 suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from run_vllm_judge_calibration_matrix import (
    ROOT,
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

PROMPT_VARIANTS = (
    "v4_orthogonal_grounding_uncertainty",
    "v5_counterfactual_masked_orthogonal",
)
DEFAULT_PROMPT_VARIANT = "v5_counterfactual_masked_orthogonal"


def bundle_rows(path: Path) -> list[dict]:
    payload = read_json(path)
    rows = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No detailed_results in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "Conspire-Bench-v3.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runner-python", type=Path, default=ROOT / ".local/venv-gptoss/bin/python"
    )
    parser.add_argument("--vllm", type=Path, default=ROOT / ".local/venv-vllm/bin/vllm")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parallel", type=int, default=32)
    parser.add_argument("--base-count", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=20260922)
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=DEFAULT_PROMPT_VARIANT,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for executable in (args.runner_python, args.vllm):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Missing executable: {executable}")
    if health_ready(args.port):
        raise RuntimeError(f"Port {args.port} is already occupied")

    sample = args.output_dir / "judge_orthogonality_original.json"
    if not sample.is_file():
        subprocess.run(
            [
                str(args.runner_python),
                "analysis/prepare_judge_orthogonality_suite.py",
                "--dataset",
                str(args.dataset),
                "--output",
                str(sample),
                "--base-count",
                str(args.base_count),
                "--seed",
                str(args.sample_seed),
            ],
            cwd=ROOT,
            check=True,
        )
    expected_rows = args.base_count * 4
    if len(bundle_rows(sample)) != expected_rows:
        raise RuntimeError(
            f"Orthogonality suite has {len(bundle_rows(sample))}, expected {expected_rows}"
        )

    base = read_json(args.config)
    judges = base.get("judges") or []
    if len(judges) != 4:
        raise ValueError("Orthogonality matrix requires exactly four configured judges")
    current = sample
    for index, judge in enumerate(judges, start=1):
        identifier = slug(model_name(judge))
        output = args.output_dir / f"after_{index:02d}_{identifier}.json"
        if calibrated_complete(
            output,
            judge,
            prompt_variant=args.prompt_variant,
            expected_rows=expected_rows,
        ):
            print(f"orthogonality judge already complete: {model_name(judge)}", flush=True)
            current = output
            continue
        generated_config = args.output_dir / f"config_{index:02d}_{identifier}.json"
        write_json(
            generated_config,
            judge_config(
                base,
                judge,
                port=args.port,
                parallel=args.parallel,
                prompt_variant=args.prompt_variant,
            ),
        )
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
                    str(args.dataset),
                    "--input",
                    str(current),
                    "--output",
                    str(output),
                    "--status",
                    str(args.output_dir / f"status_{index:02d}_{identifier}.tsv"),
                    "--parallel",
                    str(args.parallel),
                ],
                args.output_dir / f"driver_{index:02d}_{identifier}.log",
            )
        finally:
            stop_server(server)
        if not calibrated_complete(
            output,
            judge,
            prompt_variant=args.prompt_variant,
            expected_rows=expected_rows,
        ):
            raise RuntimeError(f"Orthogonality run incomplete for {model_name(judge)}")
        current = output
        print(f"orthogonality judge complete: {model_name(judge)}", flush=True)

    final = args.output_dir / "judge_orthogonality_calibrated.json"
    temporary = final.with_suffix(final.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(current.read_bytes())
    temporary.replace(final)
    analysis = args.output_dir / "judge_orthogonality_analysis.json"
    subprocess.run(
        [
            str(args.runner_python),
            "analysis/analyze_judge_orthogonality.py",
            str(final),
            "--output",
            str(analysis),
        ],
        cwd=ROOT,
        check=True,
    )
    completion = args.output_dir / "judge_orthogonality.completed.json"
    write_json(
        completion,
        {
            "ok": True,
            "prompt_variant": args.prompt_variant,
            "row_count": expected_rows,
            "judge_count": 4,
            "successful_judgments": expected_rows * 4,
            "sample": str(sample),
            "calibrated": str(final),
            "analysis": str(analysis),
            "calibrated_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
        },
    )
    print(f"ORTHOGONALITY COMPLETE: {completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
