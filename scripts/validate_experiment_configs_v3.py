#!/usr/bin/env python3
"""Validate the public V3 local and API experiment matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL = ROOT / "configs" / "experiment_v3_local_full.json"
DEFAULT_API = ROOT / "configs" / "experiment_v3_api_full.json"
EXPECTED_PROVIDERS = {"openai", "anthropic", "gemini"}
EXPECTED_TIERS = {"large", "medium", "efficient"}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_experiment_metadata(
    config: dict[str, Any], label: str, errors: list[str]
) -> None:
    experiment = config.get("experiment") or {}
    expected = {
        "design_version": "v3",
        "dataset": "Conspire-Bench-v3.json",
        "motif_count": 51,
        "scenario_count": 153,
        "canonical_context_set": "main_v3",
        "full_context_set": "full_v3",
        "rubric_version": "2.0",
    }
    for field, value in expected.items():
        if experiment.get(field) != value:
            errors.append(f"{label}.experiment.{field} must be {value!r}")


def _validate_env_keys(config: dict[str, Any], label: str, errors: list[str]) -> None:
    for provider, value in (config.get("api_keys") or {}).items():
        if not isinstance(value, str) or not value.startswith("env:"):
            errors.append(f"{label}.api_keys.{provider} must use an env: reference")


def validate(local_config: dict[str, Any], api_config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_experiment_metadata(local_config, "local", errors)
    _validate_experiment_metadata(api_config, "api", errors)
    _validate_env_keys(local_config, "local", errors)
    _validate_env_keys(api_config, "api", errors)

    local_models = local_config.get("models") or []
    if len(local_models) != 13:
        errors.append("local.models must contain the frozen 13-model matrix")
    local_families = {model.get("model_family") for model in local_models}
    if local_families != {"llama", "qwen", "gemma"}:
        errors.append("local.models must contain Llama, Qwen, and Gemma families")
    local_tiers = {model.get("capacity_tier") for model in local_models}
    if local_tiers != EXPECTED_TIERS:
        errors.append("local.models must contain efficient, medium, and large tiers")
    parameter_scales = [
        model.get("parameter_scale_b")
        for model in local_models
        if isinstance(model.get("parameter_scale_b"), (int, float))
        and not isinstance(model.get("parameter_scale_b"), bool)
    ]
    if len(parameter_scales) != len(local_models):
        errors.append("every local model must declare parameter_scale_b")
    elif min(parameter_scales) > 1 or max(parameter_scales) < 12:
        errors.append("local.models must span at least <=1B through >=12B scales")
    if len({model.get("model") for model in local_models}) != len(local_models):
        errors.append("local.models may not contain duplicate model IDs")
    for index, model in enumerate(local_models):
        if model.get("provider") != "huggingface":
            errors.append(f"local.models[{index}] must use provider=huggingface")
        config_path = model.get("config_path")
        if not config_path or not (ROOT / config_path).exists():
            errors.append(f"local.models[{index}] has a missing config_path")
        if model.get("capacity_tier") not in EXPECTED_TIERS:
            errors.append(f"local.models[{index}] has an invalid capacity_tier")

    api_models = api_config.get("models") or []
    provider_tiers = {
        (model.get("provider"), model.get("capacity_tier")) for model in api_models
    }
    expected_provider_tiers = {
        (provider, tier) for provider in EXPECTED_PROVIDERS for tier in EXPECTED_TIERS
    }
    if provider_tiers != expected_provider_tiers or len(api_models) != 9:
        errors.append(
            "api.models must contain exactly large/medium/efficient targets from "
            "OpenAI, Anthropic, and Gemini"
        )
    for index, model in enumerate(api_models):
        if int(model.get("max_tokens", 0)) < 8000:
            errors.append(f"api.models[{index}].max_tokens must be at least 8000")
        sampling_must_be_omitted = model.get("provider") == "gemini" or (
            model.get("provider") == "anthropic" and model.get("reasoning_effort")
        )
        if sampling_must_be_omitted and not model.get(
            "omit_sampling_parameters", False
        ):
            errors.append(
                f"api.models[{index}] must omit provider-discouraged sampling parameters"
            )

    judges = api_config.get("judges") or []
    if {judge.get("provider") for judge in judges} != EXPECTED_PROVIDERS or len(
        judges
    ) != 3:
        errors.append("api.judges must contain one strong judge from each API family")
    for index, judge in enumerate(judges):
        if judge.get("rubric_version") != "2.0":
            errors.append(f"api.judges[{index}].rubric_version must be 2.0")
        if int(judge.get("max_tokens", 0)) < 12000:
            errors.append(f"api.judges[{index}].max_tokens must be at least 12000")

    for section_name, entries in (("models", api_models), ("judges", judges)):
        for index, entry in enumerate(entries):
            if entry.get("provider") != "gemini":
                continue
            if "thinking_level" in entry and "thinking_budget" in entry:
                errors.append(
                    f"api.{section_name}[{index}] cannot set both Gemini thinking controls"
                )
            if "thinking_level" in entry:
                errors.append(
                    f"api.{section_name}[{index}] must use the provider default until "
                    "the released Python SDK supports thinking_level"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--api", type=Path, default=DEFAULT_API)
    args = parser.parse_args()
    errors = validate(_read(args.local), _read(args.api))
    print(
        json.dumps(
            {
                "ok": not errors,
                "local_config": str(args.local),
                "api_config": str(args.api),
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
