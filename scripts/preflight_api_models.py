#!/usr/bin/env python3
"""Make one tiny request per configured API target/judge before a paid run."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_runner import ConspireBenchmarkRunner, ModelProvider
from experiment_conditions import stable_digest
from main import load_config, validate_setup


def role_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    entries = []
    for index, model in enumerate(config.get("models") or [config.get("model")]):
        if model:
            entries.append({"role": "target", "index": index, "config": model})
    for index, judge in enumerate(config.get("judges") or [config.get("judge")]):
        if judge:
            entries.append({"role": "judge", "index": index, "config": judge})
    return entries


async def run_preflight(config_path: Path, dataset_path: Path) -> dict[str, Any]:
    config = load_config(str(config_path))
    runner = ConspireBenchmarkRunner(
        config_path=str(config_path),
        dataset_path=str(dataset_path),
    )
    results = []
    for entry in role_entries(config):
        role = entry["role"]
        role_config = entry["config"]
        provider = ModelProvider(str(role_config["provider"]))
        model = str(role_config["model"])
        started = time.monotonic()
        try:
            response = await runner._get_model_response(
                "Return exactly the word OK.",
                provider,
                model,
                is_judge=role == "judge",
                role_config_override=role_config,
            )
            metadata = getattr(response, "metadata", {})
            results.append(
                {
                    "role": role,
                    "index": entry["index"],
                    "provider": provider.value,
                    "requested_model": model,
                    "resolved_model": metadata.get("resolved_model"),
                    "interface": metadata.get("interface"),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "response_digest": stable_digest(str(response), length=16),
                    "status": "ok",
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "role": role,
                    "index": entry["index"],
                    "provider": provider.value,
                    "requested_model": model,
                    "resolved_model": None,
                    "interface": None,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "response_digest": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    return {
        "schema_version": "1.0",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_digest": stable_digest(config, length=64),
        "request_count": len(results),
        "success_count": sum(row["status"] == "ok" for row in results),
        "failure_count": sum(row["status"] == "failed" for row in results),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "experiment_v3_api_full.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "Conspire-Bench-v3.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not validate_setup(
        str(args.config),
        dataset_path=str(args.dataset),
        check_api_keys=True,
    ):
        return 1
    report = asyncio.run(run_preflight(args.config, args.dataset))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    else:
        print(rendered)
    return 0 if report["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
