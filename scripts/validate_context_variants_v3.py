#!/usr/bin/env python3
"""Validate the five-family v3 framing design and nested paraphrases."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "context_variants.json"
MAIN_SET = "main_v3"
FULL_SET = "full_v3"
ROBUSTNESS_SET = "reviewer_robustness_v3"
FRAME_FAMILIES = (
    "neutral",
    "brainstorming",
    "critical_review",
    "sensemaking",
    "supportive_listening",
)
NON_NEUTRAL_FAMILIES = FRAME_FAMILIES[1:]


def validate_context_variants(payload: dict) -> list[str]:
    errors: list[str] = []
    variants = payload.get("variants")
    sets = payload.get("sets")
    if not isinstance(variants, list) or not variants:
        return ["variants must be a non-empty list"]
    if not isinstance(sets, dict):
        return ["sets must be an object"]

    by_id: dict[str, dict] = {}
    ids: list[str] = []
    for index, variant in enumerate(variants):
        label = f"variants[{index}]"
        if not isinstance(variant, dict):
            errors.append(f"{label} must be an object")
            continue
        variant_id = variant.get("id")
        if not isinstance(variant_id, str) or not variant_id.strip():
            errors.append(f"{label}.id must be a non-empty string")
            continue
        ids.append(variant_id)
        by_id[variant_id] = variant
    duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate variant IDs: {duplicates}")

    main_ids = sets.get(MAIN_SET)
    full_ids = sets.get(FULL_SET)
    robustness_ids = sets.get(ROBUSTNESS_SET)
    if not isinstance(main_ids, list):
        errors.append(f"{MAIN_SET} must be a list")
        main_ids = []
    if not isinstance(robustness_ids, list):
        errors.append(f"{ROBUSTNESS_SET} must be a list")
        robustness_ids = []
    if not isinstance(full_ids, list):
        errors.append(f"{FULL_SET} must be a list")
        full_ids = []
    for set_name, set_ids in (
        (MAIN_SET, main_ids),
        (FULL_SET, full_ids),
        (ROBUSTNESS_SET, robustness_ids),
    ):
        missing = sorted(set(set_ids) - set(by_id))
        if missing:
            errors.append(f"{set_name} references unknown variants: {missing}")
        if len(set_ids) != len(set(set_ids)):
            errors.append(f"{set_name} contains duplicate variants")

    main_frames = [by_id[item].get("frame") for item in main_ids if item in by_id]
    if tuple(main_frames) != FRAME_FAMILIES:
        errors.append(
            f"{MAIN_SET} must contain exactly one canonical variant in frame order "
            f"{list(FRAME_FAMILIES)}; found {main_frames}"
        )
    for variant_id in main_ids:
        variant = by_id.get(variant_id, {})
        if variant.get("canonical") is not True:
            errors.append(f"{MAIN_SET}.{variant_id} must be canonical")

    expected_robustness_ids = {"neutral_none"}
    for frame in NON_NEUTRAL_FAMILIES:
        expected_robustness_ids.update(f"{frame}_v{index}" for index in range(1, 5))
    if set(robustness_ids) != expected_robustness_ids or len(robustness_ids) != 17:
        errors.append(
            f"{ROBUSTNESS_SET} must contain neutral plus four variants nested in each "
            "non-neutral family"
        )
    if full_ids != robustness_ids:
        errors.append(f"{FULL_SET} must be the ordered v3 alias of {ROBUSTNESS_SET}")

    for frame in NON_NEUTRAL_FAMILIES:
        canonical_id = f"{frame}_v1"
        family_ids = [item for item in robustness_ids if by_id.get(item, {}).get("frame") == frame]
        if len(family_ids) != 4:
            errors.append(f"{frame} must have exactly four robustness variants")
        for variant_id in family_ids:
            variant = by_id[variant_id]
            if variant_id == canonical_id:
                if variant.get("canonical") is not True:
                    errors.append(f"{canonical_id} must be canonical")
                if "paraphrase_of" in variant:
                    errors.append(f"{canonical_id} must not be a paraphrase of itself")
            else:
                if variant.get("canonical") is not False:
                    errors.append(f"{variant_id} must be non-canonical")
                if variant.get("paraphrase_of") != canonical_id:
                    errors.append(f"{variant_id}.paraphrase_of must be {canonical_id}")
                if variant.get("frame") != by_id.get(canonical_id, {}).get("frame"):
                    errors.append(f"{variant_id} must remain nested within {frame}")
                if variant.get("study_role") != "main_nested_paraphrase":
                    errors.append(
                        f"{variant_id}.study_role must be main_nested_paraphrase"
                    )

    neutral = by_id.get("neutral_none", {})
    if neutral.get("text") is not None or neutral.get("frame") != "neutral":
        errors.append("neutral_none must be the sole null-text neutral condition")
    null_text_ids = {item for item, variant in by_id.items() if variant.get("text") is None}
    if null_text_ids != {"neutral_none"}:
        errors.append(f"only neutral_none may have null text; found {sorted(null_text_ids)}")

    exploratory_ids = set(sets.get("exploratory_elicitation_v1") or [])
    if exploratory_ids & set(main_ids):
        errors.append("exploratory conditions must not enter the v3 main set")
    if exploratory_ids & set(robustness_ids):
        errors.append("exploratory conditions must not enter the v3 robustness set")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_context_variants(payload)
    report = {
        "ok": not errors,
        "config": str(args.config),
        "main_frames": [
            variant.get("frame")
            for variant_id in payload.get("sets", {}).get(MAIN_SET, [])
            for variant in [
                next(
                    (
                        item
                        for item in payload.get("variants", [])
                        if item.get("id") == variant_id
                    ),
                    {},
                )
            ]
        ],
        "full_variant_count": len(payload.get("sets", {}).get(FULL_SET, [])),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
