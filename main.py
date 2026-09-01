#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from bench_runner import ConspireBenchmarkRunner
from benchmark_types import (
    API_KEY_ENV_VARS,
    ModelProvider,
    ScenarioType,
    resolve_api_key,
)
from dataset_io import load_benchmark_dataset
from dataset_validation import (
    ALLOWED_CATEGORIES,
    format_validation_report,
    validate_dataset,
)
from experiment_conditions import (
    ContextCondition,
    adhoc_context_condition,
    load_context_set,
    select_context_conditions,
)
from result_reporting import print_results_summary, write_analysis

DEFAULT_DATASET_PATH = "Conspire-Bench-v3.json"


USAGE_EXAMPLES = """
Conspire-Bench Usage Examples:

Validate or inspect V3 without model calls:
  python main.py --config configs/experiment_v3_local_full.json --validate-only
  python main.py --config configs/experiment_v3_local_full.json --context-set main_v3 --dry-run

Run the five canonical frames:
  python main.py --config configs/experiment_v3_local_full.json --context-set main_v3 --execution-mode phased

Run all nested paraphrases:
  python main.py --config configs/experiment_v3_local_full.json --context-set full_v3 --execution-mode phased

Run one scenario and one frame variant:
  python main.py --config configs/experiment_v3_local_full.json --scenario-ids v3_weather_cloud_seeding_single_001 --context-variants sensemaking_v1 --execution-mode phased
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Conspire-Bench evaluation")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file in configs/ directory (e.g., configs/my_experiment.json)",
    )

    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help=f"Dataset JSON to validate and run. Default: {DEFAULT_DATASET_PATH}",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=sorted(ALLOWED_CATEGORIES),
        help="Categories to test (default: all)",
    )

    parser.add_argument(
        "--types",
        nargs="+",
        choices=["multi_turn", "single_turn", "resistance"],
        help="Scenario types to test (default: all)",
    )

    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        help="Limit scenarios per category",
    )

    parser.add_argument(
        "--scenario-ids",
        nargs="+",
        help="Run an explicit, frozen list of scenario IDs (useful for robustness subsets).",
    )

    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output file name",
    )

    parser.add_argument(
        "--context",
        choices=[
            "none",
            "brainstorming",
            "critical_review",
            "sensemaking",
            "supportive_listening",
            "custom",
        ],
        default="none",
        help="Context setting prompt to prime the AI before testing",
    )

    parser.add_argument(
        "--contexts",
        nargs="+",
        choices=[
            "none",
            "brainstorming",
            "critical_review",
            "sensemaking",
            "supportive_listening",
        ],
        help="Run a direct sweep across the five V3 frame families; prefer --context-set main_v3 for the frozen design.",
    )

    parser.add_argument(
        "--context-variants-file",
        default="configs/context_variants.json",
        help="Versioned context-variant JSON used by --context-set or --context-variants.",
    )

    parser.add_argument(
        "--context-set",
        help="Run a named context set from --context-variants-file, e.g. main_v3 or full_v3.",
    )

    parser.add_argument(
        "--context-variants",
        nargs="+",
        help="Run specific context variant IDs from --context-variants-file.",
    )

    parser.add_argument(
        "--custom-context",
        type=str,
        help="Custom context setting prompt (use with --context custom)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup and print the planned scenario/model/judge counts without model calls.",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config, dataset, and local model config paths, then exit.",
    )

    parser.add_argument(
        "--strict-dataset-metadata",
        action="store_true",
        help="Require expanded scenario metadata fields during dataset validation.",
    )

    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume a standard run from a previous benchmark result JSON. Successful matching results are skipped.",
    )

    parser.add_argument(
        "--status-file",
        type=str,
        help="Write per-scenario run status TSV to this path. Default: <results_dir>/status.tsv.",
    )

    parser.add_argument(
        "--execution-mode",
        choices=["standard", "phased", "generation-only", "judge-only"],
        default="standard",
        help=(
            "standard runs generation+judging per scenario; phased separates the phases; "
            "generation-only freezes conversations without judges; judge-only requires --resume-from."
        ),
    )

    return parser


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset(path: Optional[str] = None) -> dict:
    dataset_path = Path(path or DEFAULT_DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return load_benchmark_dataset(dataset_path)


def _required_api_providers(config: dict) -> list[str]:
    providers = []
    for judge in config.get("judges", []) or []:
        provider = judge.get("provider")
        if provider in API_KEY_ENV_VARS and provider not in providers:
            providers.append(provider)
    for model in config.get("models", []) or []:
        provider = model.get("provider")
        if provider in API_KEY_ENV_VARS and provider not in providers:
            providers.append(provider)
    return providers


def _validate_model_section(config: dict, section: str) -> Optional[str]:
    if section not in config:
        return f"Config must contain '{section}' section"
    if "provider" not in config[section] or "model" not in config[section]:
        return f"Config section '{section}' must contain 'provider' and 'model'"
    try:
        ModelProvider(config[section]["provider"])
    except ValueError:
        return f"Unsupported provider in '{section}': {config[section]['provider']}"
    role_config = config[section]
    model = str(role_config["model"])
    if (
        role_config["provider"] == "anthropic"
        and model in {"claude-opus-5", "claude-sonnet-5"}
        and role_config.get("temperature") not in {None, 1.0}
        and not role_config.get("omit_sampling_parameters")
    ):
        return (
            f"Anthropic model {model} rejects non-default temperature; remove it or set "
            "omit_sampling_parameters=true"
        )
    if role_config["provider"] == "gemini" and model.startswith("gemini-3"):
        temperature = role_config.get("temperature")
        if temperature not in {None, 1.0} and not role_config.get(
            "omit_sampling_parameters"
        ):
            return (
                f"Gemini 3 model {model} should use default temperature 1.0; remove "
                "the override or set omit_sampling_parameters=true"
            )
        thinking_level = role_config.get("thinking_level")
        if thinking_level not in {None, "minimal", "low", "medium", "high"}:
            return (
                f"Gemini model {model} has unsupported thinking_level={thinking_level}"
            )
    return None


def _validate_judge_sections(config: dict) -> Optional[str]:
    if not isinstance(config.get("judges"), list) or not config["judges"]:
        return "Config section 'judges' must be a non-empty list"
    for index, judge in enumerate(config["judges"]):
        temp = {"judge": judge}
        error = _validate_model_section(temp, "judge")
        if error:
            return f"Invalid judges[{index}]: {error}"
    return None


def _target_model_configs(config: dict) -> list[dict]:
    return config["models"]


def _validate_target_model_sections(config: dict) -> Optional[str]:
    if not isinstance(config.get("models"), list) or not config["models"]:
        return "Config section 'models' must be a non-empty list"
    for index, model in enumerate(config["models"]):
        temp = {"model": model}
        error = _validate_model_section(temp, "model")
        if error:
            return f"Invalid models[{index}]: {error}"
    return None


def _judge_configs_for_plan(config: dict) -> list[dict]:
    return config["judges"]


def _scenario_type_values(
    scenario_types: Optional[List[ScenarioType]],
) -> Optional[set[str]]:
    if scenario_types is None:
        return None
    return {scenario_type.value for scenario_type in scenario_types}


def _filter_dataset_scenarios(
    dataset: dict,
    categories: Optional[list[str]],
    scenario_types: Optional[List[ScenarioType]],
    max_per_category: Optional[int],
    scenario_ids: Optional[list[str]] = None,
) -> list[dict]:
    allowed_types = _scenario_type_values(scenario_types)
    category_counts: dict[str, int] = {}
    filtered = []
    allowed_ids = set(scenario_ids or [])

    for scenario in dataset.get("scenarios", []):
        if allowed_ids and scenario.get("id") not in allowed_ids:
            continue
        if categories and scenario.get("category") not in categories:
            continue
        if allowed_types and scenario.get("type") not in allowed_types:
            continue
        if max_per_category:
            category = scenario.get("category")
            if category_counts.get(category, 0) >= max_per_category:
                continue
            category_counts[category] = category_counts.get(category, 0) + 1
        filtered.append(scenario)

    return filtered


def _load_resume_results(path: Optional[str]) -> list[dict]:
    if not path:
        return []

    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return payload.get("detailed_results", [])
    raise ValueError(f"Unsupported resume result file format: {path}")


def _resolve_context_runs(args) -> list[ContextCondition]:
    structured_modes = int(bool(args.context_set)) + int(bool(args.context_variants))
    if structured_modes > 1:
        raise ValueError("Use either --context-set or --context-variants, not both.")
    if structured_modes and (
        args.contexts or args.context != "none" or args.custom_context
    ):
        raise ValueError(
            "Use structured --context-set/--context-variants or exploratory --context/--contexts, not both."
        )
    if args.context_set:
        return load_context_set(args.context_set, args.context_variants_file)
    if args.context_variants:
        return select_context_conditions(
            args.context_variants, args.context_variants_file
        )

    if args.contexts:
        if args.context != "none" or args.custom_context:
            raise ValueError(
                "Use either --context/--custom-context or --contexts, not both."
            )
        return [
            adhoc_context_condition(
                label,
                None
                if label == "none"
                else ConspireBenchmarkRunner.CONTEXT_SETTINGS[label],
            )
            for label in args.contexts
        ]

    if args.context == "none":
        return [adhoc_context_condition("none", None)]

    if args.context == "custom":
        if not args.custom_context:
            raise ValueError("--custom-context required when using --context custom")
        return [adhoc_context_condition("custom", args.custom_context)]

    return [
        adhoc_context_condition(
            args.context,
            ConspireBenchmarkRunner.CONTEXT_SETTINGS[args.context],
        )
    ]


def print_execution_plan(
    *,
    args,
    config: dict,
    dataset: dict,
    model_configs: list[dict],
    scenario_types: Optional[List[ScenarioType]],
    context_runs: list[ContextCondition],
):
    scenarios = _filter_dataset_scenarios(
        dataset,
        args.categories,
        scenario_types,
        args.max_per_category,
        args.scenario_ids,
    )
    judges = _judge_configs_for_plan(config)
    mode = args.execution_mode
    planned_conversations = len(model_configs) * len(scenarios) * len(context_runs)
    target_conversations = (
        0 if args.execution_mode == "judge-only" else planned_conversations
    )
    judge_calls = (
        0
        if args.execution_mode == "generation-only"
        else planned_conversations * len(judges)
    )
    calls_per_model = sum(
        (len(scenario.get("conversation") or []) if scenario.get("conversation") else 1)
        + (1 if context.text else 0)
        for scenario in scenarios
        for context in context_runs
    )
    target_inference_calls = (
        0
        if args.execution_mode == "judge-only"
        else len(model_configs) * calls_per_model
    )

    print("\nExecution plan")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Target models: {len(model_configs)}")
    for model in model_configs:
        print(f"  - {model['provider']}/{model['model']}")
    print(f"Judges: {len(judges)}")
    for judge in judges:
        print(f"  - {judge['provider']}/{judge['model']}")
    print(f"Target conversations to generate: {target_conversations}")
    print(f"Target model calls (turn-level): {target_inference_calls}")
    print(f"Judge calls: {judge_calls}")
    print(f"Total provider/model calls: {target_inference_calls + judge_calls}")
    print(f"Categories: {args.categories or 'all'}")
    print(f"Types: {args.types or 'all'}")
    print(f"Contexts: {', '.join(condition.variant_id for condition in context_runs)}")
    frames = ", ".join(dict.fromkeys(condition.frame for condition in context_runs))
    print(f"Frame families: {frames}")
    if args.resume_from:
        print(f"Resume from: {args.resume_from}")
    if args.status_file:
        print(f"Status file: {args.status_file}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1
    if args.execution_mode == "judge-only" and not args.resume_from:
        print("Error: --execution-mode judge-only requires --resume-from")
        return 1

    config = load_config(args.config)
    dataset = load_dataset(args.dataset)
    if args.scenario_ids:
        known_ids = {scenario.get("id") for scenario in dataset.get("scenarios", [])}
        unknown_ids = sorted(set(args.scenario_ids) - known_ids)
        if unknown_ids:
            print(f"Error: unknown scenario IDs: {', '.join(unknown_ids)}")
            return 1
    dataset_report = validate_dataset(
        dataset,
        strict_metadata=args.strict_dataset_metadata,
    )
    if args.validate_only or args.dry_run or dataset_report.errors:
        print(format_validation_report(dataset_report, max_warnings=8))
    if dataset_report.errors:
        return 1

    error = _validate_target_model_sections(config)
    if error:
        print(f"Error: {error}")
        return 1
    error = _validate_judge_sections(config)
    if error:
        print(f"Error: {error}")
        return 1

    model_configs = _target_model_configs(config)

    scenario_types = None
    if args.types:
        type_map = {
            "multi_turn": ScenarioType.MULTI_TURN,
            "single_turn": ScenarioType.SINGLE_TURN,
            "resistance": ScenarioType.RESISTANCE,
        }
        scenario_types = [type_map[t] for t in args.types]

    try:
        context_runs = _resolve_context_runs(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    if args.validate_only or args.dry_run:
        print_execution_plan(
            args=args,
            config=config,
            dataset=dataset,
            model_configs=model_configs,
            scenario_types=scenario_types,
            context_runs=context_runs,
        )
        if args.validate_only:
            print("\nValidation complete. No model calls were made.")
        else:
            print("\nDry run complete. No model calls were made.")
        return 0

    asyncio.run(
        run_benchmark_async(
            args,
            model_configs,
            scenario_types,
            context_runs,
        )
    )
    return 0


async def run_benchmark_async(
    args,
    model_configs,
    scenario_types,
    context_runs,
):
    model_str = ", ".join(
        f"{model_config['provider']}/{model_config['model']}"
        for model_config in model_configs
    )
    judge_str = ", ".join(
        f"{judge['provider']}/{judge['model']}"
        for judge in load_config(args.config)["judges"]
    )

    print(
        f"Starting Conspire-Bench standard evaluation ({args.execution_mode} execution)"
    )
    print(f"Model: {model_str}")
    print(f"Judge: {judge_str}")
    print(f"Categories: {args.categories or 'all'}")
    print(f"Types: {args.types or 'all'}")

    runner = ConspireBenchmarkRunner(
        config_path=args.config,
        dataset_path=args.dataset,
    )
    run_kwargs = {
        "models_to_test": model_configs,
        "categories": args.categories,
        "scenario_types": scenario_types,
        "max_scenarios_per_category": args.max_per_category,
        "scenario_ids": args.scenario_ids,
        "output_file": args.output,
        "context_setting": context_runs[0].text,
        "context_label": context_runs[0].variant_id,
        "context_runs": context_runs,
        "resume_results": _load_resume_results(args.resume_from),
        "status_file": args.status_file,
    }
    if args.execution_mode in {"phased", "generation-only", "judge-only"}:
        results = await runner.run_benchmark_phased(
            **run_kwargs,
            generation_only=args.execution_mode == "generation-only",
            judge_only=args.execution_mode == "judge-only",
        )
    else:
        results = await runner.run_benchmark(**run_kwargs)

    results_dir = runner.results_dir

    print_results_summary(results)
    print(f"\nFull results saved to: {results_dir}/{args.output}")

    analysis_file = args.output.replace(".json", "_analysis.txt")
    analysis_path = f"{results_dir}/{analysis_file}"
    write_analysis(results, analysis_path)


def validate_setup(
    config_path=None,
    dataset_path: Optional[str] = None,
    check_api_keys: bool = True,
):
    print("Validating setup...")
    if config_path and not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return False

    try:
        dataset_report = validate_dataset(load_dataset(dataset_path))
        if dataset_report.errors:
            print(format_validation_report(dataset_report))
            return False
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return False

    if config_path:
        try:
            config = load_config(config_path)

            error = _validate_target_model_sections(config)
            if error:
                print(f"Error: {error}")
                return False
            error = _validate_judge_sections(config)
            if error:
                print(f"Error: {error}")
                return False

            model_sections = [
                (f"models[{index}]", model)
                for index, model in enumerate(_target_model_configs(config))
            ]
            model_sections.extend(
                (f"judges[{index}]", judge)
                for index, judge in enumerate(config["judges"])
            )

            for section, section_config in model_sections:
                config_path_value = section_config.get("config_path")
                if (
                    section_config.get("provider") == "huggingface"
                    and config_path_value
                ):
                    model_config_path = Path(config_path_value)
                    if not model_config_path.is_absolute():
                        model_config_path = Path.cwd() / model_config_path
                    if not model_config_path.exists():
                        print(
                            f"Error: Local model config not found for '{section}': {config_path_value}"
                        )
                        return False

            required_providers = _required_api_providers(config)
            if check_api_keys:
                missing_keys = []
                for provider in required_providers:
                    if not resolve_api_key(config, provider):
                        missing_keys.append(
                            f"{provider} ({API_KEY_ENV_VARS[provider]})"
                        )

                if missing_keys:
                    print(f"Please configure API keys for: {', '.join(missing_keys)}")
                    return False

        except Exception as e:
            print(f"Error reading config: {e}")
            return False

    print("Setup validation passed!")
    return True


def cli(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv:
        print(USAGE_EXAMPLES)
        return 0

    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument("--config", type=str)
    temp_parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_PATH)
    temp_parser.add_argument("--dry-run", action="store_true")
    temp_parser.add_argument("--validate-only", action="store_true")
    temp_args, _ = temp_parser.parse_known_args(argv)

    if temp_args.config and not validate_setup(
        temp_args.config,
        dataset_path=temp_args.dataset,
        check_api_keys=not (temp_args.dry_run or temp_args.validate_only),
    ):
        return 1

    return main(argv)


if __name__ == "__main__":
    sys.exit(cli())
