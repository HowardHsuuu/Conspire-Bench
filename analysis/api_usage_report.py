#!/usr/bin/env python3
"""Summarize recorded API requests, resolved models, interfaces, and token usage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload.get("detailed_results", [])


def _int(mapping: dict[str, Any], *names: str) -> int | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return None


def normalize_usage(metadata: dict[str, Any]) -> dict[str, int | None]:
    usage = metadata.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    output_details = usage.get("output_tokens_details") or {}
    if not isinstance(output_details, dict):
        output_details = {}
    input_tokens = _int(usage, "input_tokens", "prompt_token_count")
    output_tokens = _int(usage, "output_tokens", "candidates_token_count")
    reasoning_tokens = _int(output_details, "reasoning_tokens") or _int(
        usage, "thoughts_token_count"
    )
    total_tokens = _int(usage, "total_tokens", "total_token_count")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def request_records(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        for turn_index, message in enumerate(row.get("conversation_log") or [], 1):
            if message.get("role") != "assistant":
                continue
            metadata = message.get("response_metadata") or {}
            records.append(
                {
                    "role": "target",
                    "response_id": row.get("response_id"),
                    "request_index": turn_index,
                    "provider": metadata.get("provider")
                    or str(row.get("model_name", "")).split("/", 1)[0],
                    "requested_model": metadata.get("requested_model")
                    or str(row.get("model_name", "")).partition("/")[2],
                    "resolved_model": metadata.get("resolved_model"),
                    "interface": metadata.get("interface")
                    or row.get("generation_interface"),
                    "access_date": row.get("access_date"),
                    **normalize_usage(metadata),
                }
            )
        for judge_index, judge in enumerate(row.get("judge_results") or [], 1):
            metadata = judge.get("response_metadata") or {}
            records.append(
                {
                    "role": "judge",
                    "response_id": row.get("response_id"),
                    "request_index": judge_index,
                    "provider": metadata.get("provider") or judge.get("provider"),
                    "requested_model": metadata.get("requested_model")
                    or judge.get("model"),
                    "resolved_model": metadata.get("resolved_model"),
                    "interface": metadata.get("interface"),
                    "access_date": row.get("access_date"),
                    "request_status": "failed" if judge.get("error") else "ok",
                    **normalize_usage(metadata),
                }
            )
    return records


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = request_records(rows)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record.get("role")),
                str(record.get("provider")),
                str(record.get("requested_model")),
            )
        ].append(record)
    groups = []
    for (role, provider, requested_model), group_values in sorted(grouped.items()):

        def token_sum(
            name: str, records: tuple[dict[str, Any], ...] = tuple(group_values)
        ) -> int:
            return sum(
                int(value[name]) for value in records if value.get(name) is not None
            )

        groups.append(
            {
                "role": role,
                "provider": provider,
                "requested_model": requested_model,
                "resolved_models": sorted(
                    {
                        str(value["resolved_model"])
                        for value in group_values
                        if value.get("resolved_model")
                    }
                ),
                "interfaces": sorted(
                    {
                        str(value["interface"])
                        for value in group_values
                        if value.get("interface")
                    }
                ),
                "access_dates": sorted(
                    {
                        str(value["access_date"])
                        for value in group_values
                        if value.get("access_date")
                    }
                ),
                "request_count": len(group_values),
                "requests_with_usage": sum(
                    value.get("total_tokens") is not None for value in group_values
                ),
                "input_tokens": token_sum("input_tokens"),
                "output_tokens": token_sum("output_tokens"),
                "reasoning_tokens": token_sum("reasoning_tokens"),
                "total_tokens": token_sum("total_tokens"),
            }
        )
    return {
        "schema_version": "1.0",
        "policy": (
            "Token accounting uses provider-returned metadata. Monetary cost is not "
            "inferred without a separately frozen provider pricing snapshot."
        ),
        "request_count": len(records),
        "requests_with_usage": sum(
            record.get("total_tokens") is not None for record in records
        ),
        "groups": groups,
        "request_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
