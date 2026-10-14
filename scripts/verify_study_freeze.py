#!/usr/bin/env python3
"""Verify the frozen Conspire-Bench V3 automated-evidence snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "study_freeze_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def verify(manifest_path: Path, *, root: Path = ROOT) -> list[str]:
    manifest = _read_object(manifest_path)
    errors: list[str] = []
    if manifest.get("schema_version") != "1.0":
        errors.append("study freeze schema_version must be 1.0")
    if manifest.get("status") != "automated_evidence_frozen_pending_human_validation":
        errors.append("study freeze has an unexpected status")

    artifacts = manifest.get("frozen_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("study freeze has no frozen_artifacts")
        return errors
    seen: set[str] = set()
    for artifact in artifacts:
        relative = str((artifact or {}).get("path") or "")
        expected = str((artifact or {}).get("sha256") or "")
        if not relative or relative in seen:
            errors.append(f"invalid or duplicate frozen path: {relative!r}")
            continue
        seen.add(relative)
        path = root / relative
        if not path.is_file():
            errors.append(f"missing frozen artifact: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            errors.append(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {observed}"
            )

    design = manifest.get("design") or {}
    completion_path = (
        root / "results/judge_v3_full_20260920/judge_v3_full.completed.json"
    )
    if completion_path.is_file():
        completion = _read_object(completion_path)
        expectations = {
            "row_count": design.get("canonical_responses"),
            "model_count": design.get("target_models"),
            "motif_count": design.get("motifs"),
            "judge_count": design.get("judge_families"),
            "successful_judgments": design.get("canonical_judgments"),
        }
        for field, expected in expectations.items():
            if completion.get(field) != expected:
                errors.append(
                    f"full completion {field}={completion.get(field)!r}; expected {expected!r}"
                )

    paraphrase_path = (
        root / "results/paraphrase_robustness_v1_20261012/paraphrase_v1.completed.json"
    )
    if paraphrase_path.is_file():
        completion = _read_object(paraphrase_path)
        expectations = {
            "row_count": design.get("paraphrase_responses"),
            "target_model_count": design.get("paraphrase_target_models"),
            "judge_count": design.get("judge_families"),
            "successful_judgments": design.get("paraphrase_judgments"),
        }
        for field, expected in expectations.items():
            if completion.get(field) != expected:
                errors.append(
                    "paraphrase completion "
                    f"{field}={completion.get(field)!r}; expected {expected!r}"
                )

    human_plan = _read_object(root / "configs/human_annotation_plan_v3.json")
    formal = (human_plan.get("expert_panel") or {}).get("formal_sample") or {}
    expected_human_sample = {
        "representative_count": 45,
        "measurement_challenge_enriched_count": 15,
        "total_item_count": 60,
        "ratings_per_item": "all_eligible_experts",
    }
    observed_human_sample = {
        field: formal.get(field) for field in expected_human_sample
    }
    if observed_human_sample != expected_human_sample:
        errors.append(
            "human plan does not match the prespecified 60-item all-expert sample"
        )
    if (human_plan.get("student_panel") or {}).get("enabled") is not False:
        errors.append("student comparison must remain disabled in the primary revision")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    errors = verify(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Study freeze verified: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
