#!/usr/bin/env python3
"""Run the four-judge prompt-calibration matrix against a frozen subset.

This launcher is intended for a single-GPU host with complete Hugging Face
snapshots already cached. It never downloads weights and never uses CPU offload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_VARIANT = "v3_behavioral_anchors_evidence_first"
CALIBRATION_PROMPT_VARIANTS = (
    "v2_scale_explicit_no_example_values",
    "v3_behavioral_anchors_evidence_first",
    "v4_orthogonal_grounding_uncertainty",
    "v5_counterfactual_masked_orthogonal",
    "v6_contrastive_feature_coding",
    "v7_split_epistemic_passes",
    "v8_span_evidenced_split_passes",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def model_name(role: dict[str, Any]) -> str:
    return f"{role['provider']}/{role['model']}"


def bundle_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    rows = payload.get("detailed_results") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"No detailed_results in {path}")
    return rows


def local_snapshot_path(model_id: str) -> Path:
    snapshots = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
    )
    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if not (candidate / "config.json").is_file():
            continue
        index = candidate / "model.safetensors.index.json"
        if index.is_file():
            files = {
                candidate / str(filename)
                for filename in (read_json(index).get("weight_map") or {}).values()
            }
            if files and all(path.is_file() for path in files):
                return candidate
        elif any(candidate.glob("*.safetensors")):
            return candidate
    raise FileNotFoundError(f"No complete local snapshot for {model_id}")


def health_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=3
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return ""
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    )


def start_server(
    role: dict[str, Any],
    *,
    vllm: Path,
    port: int,
    parallel: int,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    if health_ready(port):
        raise RuntimeError(f"Port {port} already has a healthy model server")
    name = str(role["model"])
    model_family = str(role.get("model_family") or "")
    command = [
        str(vllm),
        "serve",
        str(local_snapshot_path(name)),
        "--served-model-name",
        name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        str(role.get("dtype") or "auto"),
        "--max-model-len",
        str(max(32768, int(role.get("max_seq_length") or 32768))),
        "--gpu-memory-utilization",
        "0.80",
        "--max-num-seqs",
        str(parallel),
        "--enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--trust-remote-code",
    ]
    if model_family == "gpt_oss":
        command.extend(["--attention-config", '{"backend":"TRITON_ATTN"}'])
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{vllm.parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    if model_family == "gpt_oss":
        environment["TIKTOKEN_ENCODINGS_BASE"] = str(
            ROOT / ".local" / "tiktoken_encodings"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as output:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM server for {name} exited with {process.returncode}:\n"
                f"{tail(log_path)}"
            )
        if health_ready(port):
            return process
        time.sleep(5)
    stop_server(process)
    raise TimeoutError(f"vLLM server for {name} did not become ready")


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def calibrated_complete(
    path: Path,
    judge: dict[str, Any],
    *,
    prompt_variant: str,
    expected_rows: int = 120,
) -> bool:
    if not path.is_file():
        return False
    name = model_name(judge)
    try:
        rows = bundle_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if (
        len(rows) != expected_rows
        or len({row.get("response_id") for row in rows}) != expected_rows
    ):
        return False
    for row in rows:
        matches = [
            result
            for result in row.get("judge_results") or []
            if result.get("judge_name") == name
        ]
        if (
            len(matches) != 1
            or matches[0].get("error")
            or not matches[0].get("scores")
            or matches[0].get("judge_prompt_variant") != prompt_variant
        ):
            return False
    return True


def judge_config(
    base: dict[str, Any],
    judge: dict[str, Any],
    *,
    port: int,
    parallel: int,
    prompt_variant: str,
) -> dict[str, Any]:
    config = deepcopy(base)
    compatible = deepcopy(judge)
    compatible["inference_backend"] = "openai_compatible"
    compatible["judge_prompt_variant"] = prompt_variant
    if compatible.get("model_family") == "gpt_oss":
        compatible["max_tokens"] = 8192
        compatible["max_seq_length"] = 32768
    config["judges"] = [compatible]
    config["openai_compatible"] = {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "timeout": 3600,
    }
    config.setdefault("evaluation", {}).update(
        {
            "parallel_judgements": parallel,
            "save_intermediate_results": False,
            "unload_target_before_judge": False,
            "unload_after_judge": False,
            "unload_after_model": False,
        }
    )
    config.setdefault("experiment", {})["stage"] = (
        f"judge_prompt_calibration_{slug(model_name(judge))}"
    )
    return config


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as output:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


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
    parser.add_argument("--sample-seed", type=int, default=20260919)
    parser.add_argument(
        "--exclude-bundle",
        type=Path,
        help="Exclude motifs from a prior calibration sample.",
    )
    parser.add_argument(
        "--outcome-blind-sample",
        action="store_true",
        help="Disable score-based diagnostic enrichment in sample selection.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=CALIBRATION_PROMPT_VARIANTS,
        default=DEFAULT_PROMPT_VARIANT,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for executable in (args.runner_python, args.vllm):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"Missing executable: {executable}")
    if health_ready(args.port):
        raise RuntimeError(f"Port {args.port} is already occupied")

    sample = args.output_dir / "judge_calibration_120_original.json"
    manifest = args.output_dir / "judge_calibration_120_original.manifest.json"
    if not sample.is_file():
        sample_command = [
            str(args.runner_python),
            "analysis/prepare_judge_calibration.py",
            str(args.source_bundle),
            "--config",
            str(args.config),
            "--output",
            str(sample),
            "--manifest",
            str(manifest),
            "--seed",
            str(args.sample_seed),
        ]
        if args.exclude_bundle:
            sample_command.extend(["--exclude-bundle", str(args.exclude_bundle)])
        if args.outcome_blind_sample:
            sample_command.append("--outcome-blind")
        subprocess.run(
            sample_command,
            cwd=ROOT,
            check=True,
        )
    if len(bundle_rows(sample)) != 120:
        raise RuntimeError("Frozen calibration sample does not have 120 rows")

    base = read_json(args.config)
    current = sample
    judges = base.get("judges") or []
    if len(judges) != 4:
        raise ValueError("Calibration matrix requires exactly four configured judges")
    for index, judge in enumerate(judges, start=1):
        identifier = slug(model_name(judge))
        output = args.output_dir / f"after_{index:02d}_{identifier}.json"
        if calibrated_complete(output, judge, prompt_variant=args.prompt_variant):
            print(f"calibrated judge already complete: {model_name(judge)}", flush=True)
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
        server_log = args.output_dir / f"server_{index:02d}_{identifier}.log"
        driver_log = args.output_dir / f"driver_{index:02d}_{identifier}.log"
        status = args.output_dir / f"status_{index:02d}_{identifier}.tsv"
        server = start_server(
            judge,
            vllm=args.vllm,
            port=args.port,
            parallel=args.parallel,
            log_path=server_log,
        )
        try:
            run_logged(
                [
                    str(args.runner_python),
                    "scripts/run_judge_calibration.py",
                    "--config",
                    str(generated_config),
                    "--input",
                    str(current),
                    "--output",
                    str(output),
                    "--status",
                    str(status),
                    "--parallel",
                    str(args.parallel),
                ],
                driver_log,
            )
        finally:
            stop_server(server)
        if not calibrated_complete(output, judge, prompt_variant=args.prompt_variant):
            raise RuntimeError(f"Calibration incomplete for {model_name(judge)}")
        current = output
        print(f"calibrated judge complete: {model_name(judge)}", flush=True)

    final = args.output_dir / "judge_calibration_120_calibrated.json"
    temporary_final = final.with_suffix(final.suffix + f".tmp-{os.getpid()}")
    temporary_final.write_bytes(current.read_bytes())
    temporary_final.replace(final)
    expected = {model_name(judge) for judge in judges}
    for row in bundle_rows(final):
        successful = {
            result.get("judge_name")
            for result in row.get("judge_results") or []
            if not result.get("error")
            and result.get("scores")
            and result.get("judge_prompt_variant") == args.prompt_variant
        }
        if successful != expected:
            raise RuntimeError(
                f"Final calibrated judge set mismatch for {row.get('response_id')}"
            )
    comparison = args.output_dir / "judge_prompt_calibration_comparison.json"
    subprocess.run(
        [
            str(args.runner_python),
            "analysis/compare_judge_prompt_calibration.py",
            str(sample),
            str(final),
            "--output",
            str(comparison),
        ],
        cwd=ROOT,
        check=True,
    )
    completion = args.output_dir / "judge_prompt_calibration.completed.json"
    write_json(
        completion,
        {
            "ok": True,
            "prompt_variant": args.prompt_variant,
            "row_count": 120,
            "judge_count": 4,
            "successful_judgments": 480,
            "baseline": str(sample),
            "calibrated": str(final),
            "comparison": str(comparison),
            "calibrated_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
        },
    )
    print(f"CALIBRATION COMPLETE: {completion}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
