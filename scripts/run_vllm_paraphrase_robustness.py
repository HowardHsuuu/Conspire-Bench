#!/usr/bin/env python3
"""Run the frozen reviewer-sized paraphrase robustness matrix on one GPU.

The launcher is offline-only: all target and judge snapshots must already be
cached. It serves one model at a time, checkpoints every stage, validates the
full Cartesian product, and never falls back to CPU offload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vllm_judge_calibration_matrix import (
    bundle_rows,
    calibrated_complete,
    health_ready,
    judge_config,
    local_snapshot_path,
    model_name,
    read_json,
    run_logged,
    slug,
    start_server,
    stop_server,
    write_json,
)

DEFAULT_DESIGN = ROOT / "configs" / "paraphrase_robustness_v1.json"
DEFAULT_CONFIG = ROOT / "configs" / "experiment_v3_paraphrase_robustness.json"
DEFAULT_DATASET = ROOT / "Conspire-Bench-v3.json"
FORMAL_PROMPT_VARIANT = "v3_behavioral_anchors_evidence_first"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_model_names(config: dict[str, Any]) -> list[str]:
    return [model_name(role) for role in config.get("models") or []]


def scenario_ids_for_design(
    design: dict[str, Any], dataset: dict[str, Any]
) -> list[str]:
    selection = design.get("selection") or {}
    motifs = list(selection.get("motif_ids") or [])
    required_structures = set(selection.get("required_structures") or [])
    if len(motifs) != len(set(motifs)):
        raise ValueError("design motif_ids must be unique")
    by_motif: dict[str, list[dict[str, Any]]] = {motif: [] for motif in motifs}
    for scenario in dataset.get("scenarios") or []:
        motif = scenario.get("motif_id")
        if motif in by_motif:
            by_motif[str(motif)].append(scenario)

    errors: list[str] = []
    ordered: list[str] = []
    for motif in motifs:
        scenarios = by_motif[motif]
        structures = {str(item.get("type")) for item in scenarios}
        if len(scenarios) != 3 or structures != required_structures:
            errors.append(
                f"{motif}: expected three structures {sorted(required_structures)}, "
                f"found {sorted(structures)}"
            )
        ordered.extend(str(item["id"]) for item in scenarios)
    if errors:
        raise ValueError("; ".join(errors))
    return ordered


def validate_design(
    design: dict[str, Any],
    config: dict[str, Any],
    dataset: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    expected = design.get("expected_counts") or {}
    scenarios = scenario_ids_for_design(design, dataset)
    variants = list((design.get("contexts") or {}).get("variant_ids") or [])
    registry_sets = registry.get("sets") or {}
    context_set = str((design.get("contexts") or {}).get("set"))
    if variants != list(registry_sets.get(context_set) or []):
        raise ValueError("design variants must exactly match the frozen context set")
    if len(variants) != len(set(variants)):
        raise ValueError("design context variants must be unique")

    targets = expected_model_names(config)
    judges = [model_name(role) for role in config.get("judges") or []]
    if targets != list(design.get("target_models") or []):
        raise ValueError("experiment target models do not match the frozen design")
    if judges != list(design.get("judge_models") or []):
        raise ValueError("experiment judges do not match the frozen design")
    target_families = {str(role.get("model_family")) for role in config["models"]}
    judge_families = {str(role.get("model_family")) for role in config["judges"]}
    if target_families != {"qwen", "llama", "gemma", "gpt_oss"}:
        raise ValueError("targets must cover the four frozen model families")
    if judge_families != target_families:
        raise ValueError("judges must cover the same four model families")

    seed = (config.get("evaluation") or {}).get("seed")
    if seed != (design.get("generation") or {}).get("seed"):
        raise ValueError("generation seed differs between design and experiment config")
    prompt_variants = {
        str(role.get("judge_prompt_variant")) for role in config["judges"]
    }
    if prompt_variants != {FORMAL_PROMPT_VARIANT}:
        raise ValueError("all judges must use the frozen formal prompt variant")

    computed = {
        "motifs": len((design.get("selection") or {}).get("motif_ids") or []),
        "scenarios": len(scenarios),
        "context_variants": len(variants),
        "target_models": len(targets),
        "responses_per_model": len(scenarios) * len(variants),
        "responses": len(scenarios) * len(variants) * len(targets),
        "judges": len(judges),
        "successful_judgments": (
            len(scenarios) * len(variants) * len(targets) * len(judges)
        ),
    }
    for key, value in computed.items():
        if expected.get(key) != value:
            raise ValueError(
                f"frozen expected_counts.{key}={expected.get(key)!r}, computed {value}"
            )
    if expected.get("scenarios_per_motif") != 3:
        raise ValueError("frozen design must contain three scenarios per motif")
    return {**computed, "scenario_ids": scenarios, "variant_ids": variants}


def validate_generation_rows(
    rows: list[dict[str, Any]],
    *,
    model_names: set[str],
    scenario_ids: set[str],
    variant_ids: set[str],
    seed: int,
) -> dict[str, Any]:
    expected_rows_per_model = len(scenario_ids) * len(variant_ids)
    expected_total = expected_rows_per_model * len(model_names)
    errors: list[str] = []
    response_ids: set[str] = set()
    condition_ids: set[str] = set()
    cells: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        label = str(row.get("response_id") or f"row[{index}]")
        response_id = row.get("response_id")
        condition_id = row.get("condition_id")
        target = str(row.get("model_name"))
        scenario = str(row.get("scenario_id"))
        variant = str(row.get("prompt_variant_id"))
        if not response_id or response_id in response_ids:
            errors.append(f"{label}: missing or duplicate response_id")
        else:
            response_ids.add(str(response_id))
        if not condition_id or condition_id in condition_ids:
            errors.append(f"{label}: missing or duplicate condition_id")
        else:
            condition_ids.add(str(condition_id))
        cell = (target, scenario, variant)
        if cell in cells:
            errors.append(f"{label}: duplicate target/scenario/variant cell")
        cells.add(cell)
        counts[target] += 1
        if target not in model_names:
            errors.append(f"{label}: unexpected target model {target}")
        if scenario not in scenario_ids:
            errors.append(f"{label}: unexpected scenario {scenario}")
        if variant not in variant_ids:
            errors.append(f"{label}: unexpected prompt variant {variant}")
        if row.get("generation_seed") != seed:
            errors.append(f"{label}: generation seed mismatch")
        if row.get("replicate_id") != 0:
            errors.append(f"{label}: replicate_id must be zero")
        if row.get("error") or not row.get("generation_complete"):
            errors.append(f"{label}: target generation is incomplete")
        if not row.get("conversation_log"):
            errors.append(f"{label}: missing conversation log")
        if len(errors) >= 50:
            break

    invalid_counts = {
        target: counts.get(target, 0)
        for target in sorted(model_names)
        if counts.get(target, 0) != expected_rows_per_model
    }
    if len(rows) != expected_total:
        errors.append(f"expected {expected_total} rows, found {len(rows)}")
    if invalid_counts:
        errors.append(f"invalid rows per target: {invalid_counts}")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "row_count": len(rows),
        "target_model_count": len(model_names),
        "rows_per_model": dict(sorted(counts.items())),
    }


def validate_final_panel(
    rows: list[dict[str, Any]],
    judges: list[dict[str, Any]],
    *,
    prompt_variant: str,
) -> dict[str, int]:
    expected_names = {model_name(role) for role in judges}
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
            raise ValueError(f"incomplete judge panel for {row.get('response_id')}")
        successful += len(matches)
    return {
        "row_count": len(rows),
        "judge_count": len(expected_names),
        "successful_judgments": successful,
    }


def latest_candidate(filename: str, *, include_checkpoints: bool) -> Path | None:
    names = {filename}
    if include_checkpoints:
        names.update({f"temp_{filename}", f"checkpoint_{filename}"})
    candidates = [path for name in names for path in (ROOT / "results").rglob(name)]
    return (
        max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    )


def generation_parallelism(role: dict[str, Any], requested: int) -> int:
    scale = float(role.get("parameter_scale_b") or float("inf"))
    if scale <= 8:
        return max(requested, 64)
    return requested


def compatible_generation_config(
    base: dict[str, Any], role: dict[str, Any], *, port: int, parallel: int
) -> dict[str, Any]:
    config = deepcopy(base)
    compatible = deepcopy(role)
    compatible["inference_backend"] = "openai_compatible"
    config["models"] = [compatible]
    config["judges"] = [deepcopy(base["judges"][0])]
    config["openai_compatible"] = {
        "base_url": f"http://127.0.0.1:{port}/v1",
        "timeout": 3600,
    }
    config.setdefault("evaluation", {}).update(
        {
            "parallel_scenarios": parallel,
            "save_intermediate_results": True,
            "save_intermediate_every": 5,
            "unload_target_before_judge": False,
            "unload_after_judge": False,
            "unload_after_model": False,
        }
    )
    config.setdefault("experiment", {})["stage"] = (
        f"paraphrase_v1_generation_{slug(model_name(role))}"
    )
    return config


def ensure_idle_gpu() -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    active = [
        line
        for line in completed.stdout.splitlines()
        if line.strip() and set(part.strip() for part in line.split(",")) != {"[N/A]"}
    ]
    if active:
        raise RuntimeError(
            "GPU already has a compute process; refusing to overlap experiments"
        )


def generate_one_model(
    *,
    base: dict[str, Any],
    role: dict[str, Any],
    scenario_ids: list[str],
    variant_ids: list[str],
    seed: int,
    output_dir: Path,
    runner_python: Path,
    vllm: Path,
    port: int,
    requested_parallel: int,
) -> Path:
    name = model_name(role)
    identifier = slug(name)
    parallel = generation_parallelism(role, requested_parallel)
    config_path = output_dir / f"generation_config_{identifier}.json"
    write_json(
        config_path,
        compatible_generation_config(base, role, port=port, parallel=parallel),
    )
    output_name = f"paraphrase_v1_generation_{identifier}.json"
    expected_model = {name}
    expected_scenarios = set(scenario_ids)
    expected_variants = set(variant_ids)

    existing = latest_candidate(output_name, include_checkpoints=False)
    if existing is not None:
        try:
            validate_generation_rows(
                bundle_rows(existing),
                model_names=expected_model,
                scenario_ids=expected_scenarios,
                variant_ids=expected_variants,
                seed=seed,
            )
            print(f"generation already complete: {name}: {existing}", flush=True)
            return existing
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    resume = latest_candidate(output_name, include_checkpoints=True)
    server = start_server(
        role,
        vllm=vllm,
        port=port,
        parallel=parallel,
        log_path=output_dir / f"generation_server_{identifier}.log",
    )
    try:
        command = [
            str(runner_python),
            "main.py",
            "--config",
            str(config_path),
            "--context-variants",
            *variant_ids,
            "--scenario-ids",
            *scenario_ids,
            "--execution-mode",
            "generation-only",
            "--status-file",
            str(output_dir / f"generation_status_{identifier}.tsv"),
            "--output",
            output_name,
        ]
        if resume is not None:
            command.extend(["--resume-from", str(resume)])
        run_logged(command, output_dir / f"generation_driver_{identifier}.log")
    finally:
        stop_server(server)

    result = latest_candidate(output_name, include_checkpoints=False)
    if result is None:
        raise RuntimeError(f"generation produced no final bundle for {name}")
    validate_generation_rows(
        bundle_rows(result),
        model_names=expected_model,
        scenario_ids=expected_scenarios,
        variant_ids=expected_variants,
        seed=seed,
    )
    print(f"generation complete: {name}: {result}", flush=True)
    return result


def judge_one_stage(
    *,
    base: dict[str, Any],
    judge: dict[str, Any],
    source: Path,
    output: Path,
    output_dir: Path,
    runner_python: Path,
    vllm: Path,
    port: int,
    parallel: int,
    expected_rows: int,
) -> Path:
    if calibrated_complete(
        output,
        judge,
        prompt_variant=FORMAL_PROMPT_VARIANT,
        expected_rows=expected_rows,
    ):
        print(f"judge already complete: {model_name(judge)}", flush=True)
        return output

    identifier = slug(model_name(judge))
    generated_config = output_dir / f"judge_config_{identifier}.json"
    stage_config = judge_config(
        base,
        judge,
        port=port,
        parallel=parallel,
        prompt_variant=FORMAL_PROMPT_VARIANT,
    )
    stage_config.setdefault("experiment", {})["stage"] = (
        f"paraphrase_v1_judge_{identifier}"
    )
    write_json(generated_config, stage_config)

    for attempt in range(1, 4):
        server = start_server(
            judge,
            vllm=vllm,
            port=port,
            parallel=parallel,
            log_path=output_dir / f"judge_server_{identifier}.log",
        )
        runner_error: subprocess.CalledProcessError | None = None
        try:
            try:
                run_logged(
                    [
                        str(runner_python),
                        "scripts/run_judge_calibration.py",
                        "--config",
                        str(generated_config),
                        "--dataset",
                        str(DEFAULT_DATASET),
                        "--input",
                        str(source),
                        "--output",
                        str(output),
                        "--status",
                        str(output_dir / f"judge_status_{identifier}.tsv"),
                        "--parallel",
                        str(parallel),
                        "--save-intermediate-every",
                        "100",
                    ],
                    output_dir / f"judge_driver_{identifier}.log",
                )
            except subprocess.CalledProcessError as error:
                runner_error = error
        finally:
            stop_server(server)

        if calibrated_complete(
            output,
            judge,
            prompt_variant=FORMAL_PROMPT_VARIANT,
            expected_rows=expected_rows,
        ):
            print(f"judge complete: {model_name(judge)}", flush=True)
            return output
        if attempt < 3:
            detail = (
                f"runner exit {runner_error.returncode}"
                if runner_error is not None
                else "incomplete output"
            )
            print(
                f"judge retry {attempt + 1}/3: {model_name(judge)} ({detail})",
                flush=True,
            )
    raise RuntimeError(f"judge failed after three passes: {model_name(judge)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--runner-python", type=Path, default=ROOT / ".local/venv-gptoss/bin/python"
    )
    parser.add_argument("--vllm", type=Path, default=ROOT / ".local/venv-vllm/bin/vllm")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--generation-parallel", type=int, default=32)
    parser.add_argument("--judge-parallel", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--stop-after-generation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    for path in (args.design, args.config, args.dataset):
        if not path.is_file():
            raise FileNotFoundError(path)
    design = read_json(args.design)
    config = read_json(args.config)
    dataset = read_json(args.dataset)
    registry_path = ROOT / str((design.get("contexts") or {}).get("registry"))
    registry = read_json(registry_path)
    plan = validate_design(design, config, dataset, registry)
    print(
        json.dumps(
            {key: value for key, value in plan.items() if not key.endswith("_ids")},
            indent=2,
        )
    )
    print("target models:")
    for name in expected_model_names(config):
        print(f"  - {name}")
    print("context variants:")
    for variant in plan["variant_ids"]:
        print(f"  - {variant}")
    if args.plan_only:
        print("PLAN ONLY: no model was loaded and no output was written")
        return 0

    if args.generation_parallel < 1 or args.judge_parallel < 1:
        raise ValueError("parallelism must be positive")
    for executable in (args.runner_python, args.vllm):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(f"missing executable: {executable}")
    for role in [*(config.get("models") or []), *(config.get("judges") or [])]:
        local_snapshot_path(str(role["model"]))
    if health_ready(args.port):
        raise RuntimeError(f"port {args.port} is already occupied")
    ensure_idle_gpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed = int((design.get("generation") or {})["seed"])
    generated: list[Path] = []
    for role in config["models"]:
        generated.append(
            generate_one_model(
                base=config,
                role=role,
                scenario_ids=plan["scenario_ids"],
                variant_ids=plan["variant_ids"],
                seed=seed,
                output_dir=args.output_dir,
                runner_python=args.runner_python,
                vllm=args.vllm,
                port=args.port,
                requested_parallel=args.generation_parallel,
            )
        )

    merged = args.output_dir / "paraphrase_v1_generation_merged.json"
    if not merged.is_file():
        run_logged(
            [
                str(args.runner_python),
                "scripts/merge_generation_results.py",
                *(str(path) for path in generated),
                "--output",
                str(merged),
                "--expected-rows",
                str(plan["responses"]),
                "--model-config",
                str(args.config),
                "--expected-rows-per-model",
                str(plan["responses_per_model"]),
            ],
            args.output_dir / "generation_merge.log",
        )
    generation_counts = validate_generation_rows(
        bundle_rows(merged),
        model_names=set(expected_model_names(config)),
        scenario_ids=set(plan["scenario_ids"]),
        variant_ids=set(plan["variant_ids"]),
        seed=seed,
    )
    generation_completion = args.output_dir / "paraphrase_v1_generation.completed.json"
    write_json(
        generation_completion,
        {
            "ok": True,
            **generation_counts,
            "design_sha256": sha256_file(args.design),
            "config_sha256": sha256_file(args.config),
            "dataset_sha256": sha256_file(args.dataset),
            "context_registry_sha256": sha256_file(registry_path),
            "merged_generation": str(merged),
            "merged_generation_sha256": sha256_file(merged),
        },
    )
    if args.stop_after_generation:
        print(f"GENERATION COMPLETE: {generation_completion}")
        return 0

    current = merged
    for index, judge in enumerate(config["judges"], start=1):
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
            parallel=args.judge_parallel,
            expected_rows=plan["responses"],
        )

    final = args.output_dir / "paraphrase_v1_four_judges.json"
    temporary = final.with_suffix(final.suffix + f".tmp-{os.getpid()}")
    shutil.copyfile(current, temporary)
    temporary.replace(final)
    final_rows = bundle_rows(final)
    validate_generation_rows(
        final_rows,
        model_names=set(expected_model_names(config)),
        scenario_ids=set(plan["scenario_ids"]),
        variant_ids=set(plan["variant_ids"]),
        seed=seed,
    )
    final_counts = validate_final_panel(
        final_rows,
        config["judges"],
        prompt_variant=FORMAL_PROMPT_VARIANT,
    )
    if final_counts["successful_judgments"] != plan["successful_judgments"]:
        raise RuntimeError("final successful-judgment count differs from frozen plan")
    completion = args.output_dir / "paraphrase_v1.completed.json"
    write_json(
        completion,
        {
            "ok": True,
            "design_id": design["design_id"],
            "prompt_variant": FORMAL_PROMPT_VARIANT,
            **generation_counts,
            **final_counts,
            "design_sha256": sha256_file(args.design),
            "config_sha256": sha256_file(args.config),
            "dataset_sha256": sha256_file(args.dataset),
            "context_registry_sha256": sha256_file(registry_path),
            "generation_sha256": sha256_file(merged),
            "final": str(final),
            "final_sha256": sha256_file(final),
        },
    )
    print(f"PARAPHRASE ROBUSTNESS COMPLETE: {completion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
