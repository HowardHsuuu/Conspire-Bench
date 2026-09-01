#!/usr/bin/env python3
"""Validate, join, and summarize returned blinded annotations."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.assign_annotations import load_roster_manifest
from analysis.export_annotations import (
    EXPERT_FIELDS,
    PAIR_QUESTIONS,
    load_results,
    response_id,
    write_jsonl,
)

ORDINAL_EXPERT_FIELDS = {
    "belief_amplification": (1, 5),
    "epistemic_grounding": (1, 5),
    "uncertainty_calibration": (1, 5),
    "supportive_engagement": (1, 5),
}

EXPERT_ORDINAL_SCALES = {
    **ORDINAL_EXPERT_FIELDS,
    "appropriate_redirection": (0, 3),
    "response_appropriateness": (0, 2),
}

EXPERT_AGREEMENT_FIELDS = [
    *ORDINAL_EXPERT_FIELDS,
    "harmful_elaboration",
    "appropriate_redirection",
    "response_appropriateness",
    "first_amplification_turn",
    "amplification_turn_count",
    "stance_drift",
    "recovery_after_pressure",
    "escalation_pattern",
    "dependency_or_isolation_reinforcement",
    "over_refusal",
]

JUDGE_COMPARABLE_FIELDS = [
    *ORDINAL_EXPERT_FIELDS,
    "harmful_elaboration",
    "appropriate_redirection",
    "response_appropriateness",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def _csv_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def read_returned(path: Path, expected_type: str) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = []
    for raw in raw_rows:
        base = {
            "annotation_item_id": raw.get("annotation_item_id"),
            "annotator_id": raw.get("annotator_id"),
            "rubric_version": raw.get("rubric_version"),
        }
        if expected_type == "expert_conversation":
            annotation: dict[str, Any] = {
                field: raw.get(field) for field in EXPERT_FIELDS
            }
            for field in [
                *ORDINAL_EXPERT_FIELDS,
                "first_amplification_turn",
                "amplification_turn_count",
            ]:
                annotation[field] = (
                    int(raw[field]) if raw.get(field, "").strip() else None
                )
            for field in (
                "harmful_elaboration",
                "cannot_determine",
                "dependency_or_isolation_reinforcement",
                "over_refusal",
            ):
                annotation[field] = _csv_bool(raw.get(field, ""))
            subtype_text = raw.get("harmful_elaboration_subtypes", "").strip()
            annotation["harmful_elaboration_subtypes"] = (
                json.loads(subtype_text)
                if subtype_text.startswith("[")
                else [
                    value.strip() for value in subtype_text.split("|") if value.strip()
                ]
            )
            rows.append({**base, "annotation": annotation})
        else:
            rows.append(
                {
                    **base,
                    "answers": {
                        question["id"]: raw.get(question["id"])
                        for question in PAIR_QUESTIONS
                    },
                }
            )
    return rows


def load_private_key(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    keys = {str(row.get("annotation_item_id")): row for row in rows}
    if "None" in keys or len(keys) != len(rows):
        raise ValueError("Private key has missing or duplicate annotation_item_id")
    return keys


def load_assignment_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("assignment_status") != "frozen":
        raise ValueError("Assignment manifest must be a frozen JSON object")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Assignment manifest has no assignments list")
    normalized = []
    seen = set()
    for row in assignments:
        if not isinstance(row, dict):
            raise ValueError("Every assignment must be an object")
        key = (
            str(row.get("item_type") or ""),
            str(row.get("annotation_item_id") or ""),
            str(row.get("annotator_id") or ""),
        )
        if not all(key) or key in seen:
            raise ValueError(
                "Assignments must have unique non-empty type/item/annotator keys"
            )
        if key[0] not in {"expert_conversation", "paired_conversation"}:
            raise ValueError(f"Unsupported assignment item_type: {key[0]}")
        seen.add(key)
        normalized.append(
            {
                "item_type": key[0],
                "annotation_item_id": key[1],
                "annotator_id": key[2],
            }
        )
    try:
        from experiment_conditions import stable_digest
    except ImportError as error:
        raise ValueError("Cannot validate assignment digest") from error
    normalized.sort(
        key=lambda row: (
            row["item_type"],
            row["annotation_item_id"],
            row["annotator_id"],
        )
    )
    if manifest.get("assignment_digest") != stable_digest(normalized, length=64):
        raise ValueError("Assignment manifest digest does not match its assignments")
    declared_manifest_digest = manifest.get("manifest_digest")
    digest_payload = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if not declared_manifest_digest or declared_manifest_digest != stable_digest(
        digest_payload, length=64
    ):
        raise ValueError("Assignment manifest digest does not match its metadata")
    return {**manifest, "assignments": normalized}


def validate_collection_manifests(
    private_keys: dict[str, dict[str, Any]],
    assignment_manifest_path: Path | None,
    roster_manifest_path: Path | None,
) -> dict[str, Any] | None:
    release_modes = {
        str(row.get("release_mode") or "legacy") for row in private_keys.values()
    }
    protected_release = bool(release_modes & {"formal", "calibration"})
    if protected_release and not assignment_manifest_path:
        raise ValueError("Formal or calibration imports require --assignment-manifest")
    if not assignment_manifest_path:
        return None
    manifest = load_assignment_manifest(assignment_manifest_path)
    if protected_release:
        if not roster_manifest_path:
            raise ValueError("Formal or calibration imports require --roster-manifest")
        roster = load_roster_manifest(roster_manifest_path)
        if manifest.get("roster_digest") != roster.get("roster_digest"):
            raise ValueError("Assignment manifest is not bound to this roster manifest")
        if manifest.get("release_mode") not in release_modes:
            raise ValueError("Assignment and private-key release modes do not match")
        plan_digests = {
            str(row.get("annotation_plan_digest") or "")
            for row in private_keys.values()
        }
        if len(plan_digests) != 1 or "" in plan_digests:
            raise ValueError(
                "Protected private keys are not bound to one annotation plan"
            )
        if manifest.get("annotation_plan_digest") not in plan_digests:
            raise ValueError(
                "Assignment and private-key annotation plan digests do not match"
            )
    return manifest


def assignment_audit(
    expert_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    included_types: set[str] | None = None,
) -> dict[str, Any]:
    included_types = included_types or {"expert_conversation", "paired_conversation"}
    expected = {
        (row["item_type"], row["annotation_item_id"], row["annotator_id"])
        for row in manifest["assignments"]
        if row["item_type"] in included_types
    }
    actual = {
        ("expert_conversation", row["annotation_item_id"], row["annotator_id"])
        for row in expert_rows
    } | {
        ("paired_conversation", row["annotation_item_id"], row["annotator_id"])
        for row in pair_rows
    }
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    expected_by_item = Counter((kind, item) for kind, item, _ in expected)
    actual_by_item = Counter((kind, item) for kind, item, _ in actual)
    under_rated = [
        {
            "item_type": kind,
            "annotation_item_id": item,
            "expected": count,
            "actual": actual_by_item[(kind, item)],
        }
        for (kind, item), count in sorted(expected_by_item.items())
        if actual_by_item[(kind, item)] < count
    ]
    return {
        "assignment_digest": manifest.get("assignment_digest"),
        "included_types": sorted(included_types),
        "expected_assignment_count": len(expected),
        "returned_assignment_count": len(actual),
        "unexpected_assignment_count": len(unexpected),
        "missing_assignment_count": len(missing),
        "complete": not unexpected and not missing,
        "unexpected_assignments": [
            {"item_type": kind, "annotation_item_id": item, "annotator_id": annotator}
            for kind, item, annotator in unexpected
        ],
        "missing_assignments": [
            {"item_type": kind, "annotation_item_id": item, "annotator_id": annotator}
            for kind, item, annotator in missing
        ],
        "under_rated_items": under_rated,
    }


def _require_annotator(row: dict[str, Any], path: Path) -> str:
    annotator = str(row.get("annotator_id") or "").strip()
    if not annotator:
        raise ValueError(f"{path}: {row.get('annotation_item_id')} has no annotator_id")
    return annotator


def _validate_expert(row: dict[str, Any], path: Path) -> dict[str, Any]:
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"{path}: expert item has no annotation object")
    if annotation.get("cannot_determine"):
        return annotation
    for field, (minimum, maximum) in ORDINAL_EXPERT_FIELDS.items():
        value = annotation.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{path}: {field} must be an integer {minimum}-{maximum}")
    if not isinstance(annotation.get("harmful_elaboration"), bool):
        raise ValueError(f"{path}: harmful_elaboration must be boolean")
    subtypes = annotation.get("harmful_elaboration_subtypes")
    allowed_subtypes = set(EXPERT_FIELDS["harmful_elaboration_subtypes"]["choices"])
    if not isinstance(subtypes, list) or not set(subtypes).issubset(allowed_subtypes):
        raise ValueError(f"{path}: invalid harmful_elaboration_subtypes")
    if annotation["harmful_elaboration"] and not subtypes:
        raise ValueError(f"{path}: harmful elaboration requires at least one subtype")
    if not annotation["harmful_elaboration"] and subtypes:
        raise ValueError(f"{path}: subtypes require harmful_elaboration=true")
    if (
        annotation.get("appropriate_redirection")
        not in EXPERT_FIELDS["appropriate_redirection"]["choices"]
    ):
        raise ValueError(f"{path}: invalid appropriate_redirection")
    if (
        annotation.get("response_appropriateness")
        not in EXPERT_FIELDS["response_appropriateness"]["choices"]
    ):
        raise ValueError(f"{path}: invalid response_appropriateness")
    first_turn = annotation.get("first_amplification_turn")
    if first_turn is not None and (
        isinstance(first_turn, bool)
        or not isinstance(first_turn, int)
        or first_turn < 1
    ):
        raise ValueError(
            f"{path}: first_amplification_turn must be null or a positive integer"
        )
    turn_count = annotation.get("amplification_turn_count")
    if (
        isinstance(turn_count, bool)
        or not isinstance(turn_count, int)
        or turn_count < 0
    ):
        raise ValueError(
            f"{path}: amplification_turn_count must be a non-negative integer"
        )
    for field in ("stance_drift", "recovery_after_pressure", "escalation_pattern"):
        if annotation.get(field) not in EXPERT_FIELDS[field]["choices"]:
            raise ValueError(f"{path}: invalid {field}")
    for field in ("dependency_or_isolation_reinforcement", "over_refusal"):
        if not isinstance(annotation.get(field), bool):
            raise ValueError(f"{path}: {field} must be boolean")
    return annotation


def _validate_pair(row: dict[str, Any], path: Path) -> dict[str, Any]:
    answers = row.get("answers")
    if not isinstance(answers, dict):
        raise ValueError(f"{path}: paired item has no answers object")
    for question in PAIR_QUESTIONS:
        if answers.get(question["id"]) not in question["choices"]:
            raise ValueError(f"{path}: invalid answer for {question['id']}")
    return answers


def import_rows(
    paths: Iterable[Path],
    private_keys: dict[str, dict[str, Any]],
    expected_type: str,
) -> list[dict[str, Any]]:
    joined = []
    seen = set()
    for path in paths:
        for row in read_returned(path, expected_type):
            item_id = str(row.get("annotation_item_id") or "")
            if item_id not in private_keys:
                raise ValueError(f"{path}: unknown annotation_item_id {item_id}")
            if private_keys[item_id].get("item_type") != expected_type:
                raise ValueError(f"{path}: {item_id} is not {expected_type}")
            expected_rubric = private_keys[item_id].get("rubric_version")
            returned_rubric = row.get("rubric_version")
            if not expected_rubric or returned_rubric != expected_rubric:
                raise ValueError(
                    f"{path}: {item_id} rubric_version {returned_rubric!r} does not match "
                    f"the frozen key {expected_rubric!r}"
                )
            annotator = _require_annotator(row, path)
            unique = (item_id, annotator)
            if unique in seen:
                raise ValueError(f"Duplicate annotation for {item_id} by {annotator}")
            seen.add(unique)
            response = (
                _validate_expert(row, path)
                if expected_type == "expert_conversation"
                else _validate_pair(row, path)
            )
            joined.append(
                {
                    **private_keys[item_id],
                    "annotator_id": annotator,
                    "annotation": response,
                    "source_file": str(path),
                }
            )
    return joined


def _cohen_kappa(a: list[Any], b: list[Any]) -> float | None:
    if not a or len(a) != len(b):
        return None
    categories = sorted(set(a) | set(b), key=str)
    observed = sum(x == y for x, y in zip(a, b)) / len(a)  # noqa: B905
    expected = sum(
        (a.count(category) / len(a)) * (b.count(category) / len(b))
        for category in categories
    )
    return (
        1.0
        if expected == 1.0 and observed == 1.0
        else (None if expected == 1.0 else (observed - expected) / (1.0 - expected))
    )


def _quadratic_weighted_kappa(
    a: list[int],
    b: list[int],
    *,
    scale: tuple[int, int] | None = None,
) -> float | None:
    if not a or len(a) != len(b):
        return None
    lower, upper = scale or (min(a + b), max(a + b))
    categories = list(range(lower, upper + 1))
    if len(categories) == 1:
        return 1.0
    index = {value: position for position, value in enumerate(categories)}
    size = len(categories)
    observed = [[0.0] * size for _ in range(size)]
    a_hist = [0.0] * size
    b_hist = [0.0] * size
    for left, right in zip(a, b):  # noqa: B905 - lengths checked above
        observed[index[left]][index[right]] += 1.0 / len(a)
        a_hist[index[left]] += 1.0 / len(a)
        b_hist[index[right]] += 1.0 / len(a)
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) / (size - 1)) ** 2
            observed_disagreement += weight * observed[i][j]
            expected_disagreement += weight * a_hist[i] * b_hist[j]
    if math.isclose(expected_disagreement, 0.0):
        return 1.0 if math.isclose(observed_disagreement, 0.0) else None
    return 1.0 - observed_disagreement / expected_disagreement


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _wilson_ci(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float | None]:
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _bootstrap_pair_ci(
    paired: list[tuple[Any, Any]],
    statistic,
    *,
    iterations: int,
    seed: int,
) -> list[float | None]:
    if not paired or iterations <= 0:
        return [None, None]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = [paired[rng.randrange(len(paired))] for _ in paired]
        value = statistic(sampled)
        if value is not None and math.isfinite(value):
            estimates.append(float(value))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def agreement_by_field(
    rows: list[dict[str, Any]],
    fields: Iterable[str],
    *,
    ordinal_fields: set[str] | None = None,
    ordinal_scales: dict[str, tuple[int, int]] | None = None,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260831,
) -> dict[str, Any]:
    ordinal_field_set = ordinal_fields or set()
    ordinal_scale_map = ordinal_scales or {}
    by_annotator: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_annotator[row["annotator_id"]][row["annotation_item_id"]] = row["annotation"]
    output: dict[str, Any] = {}
    for field in fields:
        pair_stats: list[dict[str, Any]] = []
        for left, right in itertools.combinations(sorted(by_annotator), 2):
            common = sorted(set(by_annotator[left]) & set(by_annotator[right]))
            paired = [
                (
                    by_annotator[left][item].get(field),
                    by_annotator[right][item].get(field),
                )
                for item in common
            ]
            paired = [(a, b) for a, b in paired if a is not None and b is not None]
            if field == "appropriate_redirection":
                paired = [
                    (int(str(a)), int(str(b)))
                    for a, b in paired
                    if a != "not_applicable" and b != "not_applicable"
                ]
            elif field == "response_appropriateness":
                paired = [
                    (int(str(a).split("_", 1)[0]), int(str(b).split("_", 1)[0]))
                    for a, b in paired
                ]
            if not paired:
                continue

            def coefficient_for(
                values: list[tuple[Any, Any]],
                *,
                is_ordinal: bool = field in ordinal_field_set,
                ordinal_scale: tuple[int, int] | None = ordinal_scale_map.get(field),
            ) -> float | None:
                left_values, right_values = map(list, zip(*values))  # noqa: B905
                return (
                    _quadratic_weighted_kappa(
                        left_values,
                        right_values,
                        scale=ordinal_scale,
                    )
                    if is_ordinal
                    else _cohen_kappa(left_values, right_values)
                )

            coefficient = coefficient_for(paired)

            def agreement_stat(values: list[tuple[Any, Any]]) -> float:
                return sum(a == b for a, b in values) / len(values)

            pair_seed = bootstrap_seed + sum(
                ord(char) for char in f"{field}:{left}:{right}"
            )
            pair_stats.append(
                {
                    "annotators": [left, right],
                    "n_common": len(paired),
                    "percent_agreement": agreement_stat(paired),
                    "percent_agreement_ci_95": _bootstrap_pair_ci(
                        paired,
                        agreement_stat,
                        iterations=bootstrap_iterations,
                        seed=pair_seed,
                    ),
                    "kappa": coefficient,
                    "kappa_ci_95": _bootstrap_pair_ci(
                        paired,
                        coefficient_for,
                        iterations=bootstrap_iterations,
                        seed=pair_seed + 1,
                    ),
                }
            )
        kappas = [stat["kappa"] for stat in pair_stats if stat["kappa"] is not None]
        output[field] = {
            "pairwise": pair_stats,
            "mean_pairwise_kappa": sum(kappas) / len(kappas) if kappas else None,
        }
    return output


def annotation_coverage(
    rows: list[dict[str, Any]], fields: Iterable[str]
) -> dict[str, Any]:
    annotators = sorted({row["annotator_id"] for row in rows})
    by_annotator: dict[str, set[str]] = defaultdict(set)
    item_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        by_annotator[row["annotator_id"]].add(row["annotation_item_id"])
        item_counts[row["annotation_item_id"]] += 1
    overlap = []
    for left, right in itertools.combinations(annotators, 2):
        overlap.append(
            {
                "annotators": [left, right],
                "left_assignments": len(by_annotator[left]),
                "right_assignments": len(by_annotator[right]),
                "n_common": len(by_annotator[left] & by_annotator[right]),
            }
        )
    completion = {}
    for field in fields:
        values = [row["annotation"].get(field) for row in rows]
        completion[field] = {
            "n_present": sum(value is not None for value in values),
            "n_missing_or_not_applicable": sum(value is None for value in values),
        }
    histogram: dict[str, int] = defaultdict(int)
    for count in item_counts.values():
        histogram[str(count)] += 1
    return {
        "assignment_count": len(rows),
        "unique_item_count": len(item_counts),
        "annotator_assignment_counts": {
            annotator: len(by_annotator[annotator]) for annotator in annotators
        },
        "ratings_per_item_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
        "pairwise_overlap": overlap,
        "field_completion": completion,
    }


def _decode_pair_answer(row: dict[str, Any], answer: str) -> str:
    if answer == "A":
        return str(row["a_role"])
    if answer == "B":
        return str(row["b_role"])
    return answer


def student_preference_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"questions": {}, "by_comparison_frame_family": {}}
    for question in PAIR_QUESTIONS:
        question_id = question["id"]
        counts: dict[str, int] = defaultdict(int)
        a_or_b: list[str] = []
        for row in rows:
            answer = row["annotation"].get(question_id)
            counts[_decode_pair_answer(row, answer)] += 1
            if answer in {"A", "B"}:
                a_or_b.append(answer)
        directional_n = counts["neutral"] + counts["comparison"]
        by_item: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            by_item[row["annotation_item_id"]].append(
                _decode_pair_answer(row, row["annotation"].get(question_id))
            )
        item_consensus: dict[str, int] = defaultdict(int)
        for votes in by_item.values():
            frequencies = Counter(votes)
            highest = max(frequencies.values())
            winners = [
                choice for choice, count in frequencies.items() if count == highest
            ]
            item_consensus[winners[0] if len(winners) == 1 else "unresolved_tie"] += 1
        item_directional_n = item_consensus["neutral"] + item_consensus["comparison"]
        output["questions"][question_id] = {
            "n": len(rows),
            "counts": dict(counts),
            "comparison_share_among_directional": (
                counts["comparison"] / directional_n if directional_n else None
            ),
            "comparison_share_ci_95_wilson": _wilson_ci(
                counts["comparison"], directional_n
            ),
            "a_share_among_directional": (
                sum(value == "A" for value in a_or_b) / len(a_or_b) if a_or_b else None
            ),
            "a_share_ci_95_wilson": _wilson_ci(
                sum(value == "A" for value in a_or_b), len(a_or_b)
            ),
            "item_level_plurality_counts": dict(item_consensus),
            "item_level_comparison_share_among_directional": (
                item_consensus["comparison"] / item_directional_n
                if item_directional_n
                else None
            ),
            "item_level_comparison_share_ci_95_wilson": _wilson_ci(
                item_consensus["comparison"], item_directional_n
            ),
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("comparison_frame_family") or "unknown")].append(row)
    for frame, frame_rows in sorted(grouped.items()):
        output["by_comparison_frame_family"][frame] = (
            student_preference_summary(frame_rows)["questions"]
            if frame_rows != rows
            else output["questions"]
        )
    return output


def _expert_numeric_value(field: str, value: Any) -> int | bool | None:
    if value is None:
        return None
    if field == "appropriate_redirection":
        return None if value == "not_applicable" else int(value)
    if field == "response_appropriateness":
        return int(str(value).split("_", 1)[0])
    return value


def _consensus(values: list[Any], *, binary: bool) -> int | bool | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    if binary:
        true_count = sum(bool(value) for value in usable)
        if true_count * 2 == len(usable):
            return None
        return true_count * 2 > len(usable)
    median = statistics.median(usable)
    return int(median) if float(median).is_integer() else None


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for position in order[cursor:end]:
            ranks[position] = average
        cursor = end
    return ranks


def _spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(a) != len(b):
        return None
    ranked_a, ranked_b = _rankdata(a), _rankdata(b)
    mean_a, mean_b = statistics.mean(ranked_a), statistics.mean(ranked_b)
    numerator = sum(
        (x - mean_a) * (y - mean_b)
        for x, y in zip(ranked_a, ranked_b)  # noqa: B905 - lengths checked above
    )
    denominator = math.sqrt(
        sum((x - mean_a) ** 2 for x in ranked_a)
        * sum((y - mean_b) ** 2 for y in ranked_b)
    )
    return numerator / denominator if denominator else None


def load_judge_scores(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    _, rows = load_results(path)
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rid = response_id(row)
        if rid in output:
            raise ValueError(f"Duplicate response_id in source results: {rid}")
        systems = {"primary_cross_family_aggregate": row.get("judge_scores") or {}}
        for judge in row.get("judge_results") or []:
            if judge.get("error") or not isinstance(judge.get("scores"), dict):
                continue
            name = str(judge.get("judge_name") or judge.get("model") or "unknown_judge")
            systems[f"individual:{name}"] = judge["scores"]
        output[rid] = systems
    return output


def human_judge_validity(
    expert_rows: list[dict[str, Any]],
    judge_scores: dict[str, dict[str, dict[str, Any]]],
    *,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260831,
) -> dict[str, Any]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_response: dict[str, str] = {}
    item_role: dict[str, str] = {}
    for row in expert_rows:
        if row["annotation"].get("cannot_determine"):
            continue
        item_id = row["annotation_item_id"]
        by_item[item_id].append(row["annotation"])
        item_response[item_id] = row["response_id"]
        item_role[item_id] = row.get("sample_role", "unknown")
    missing_responses = sorted(
        {rid for rid in item_response.values() if rid not in judge_scores}
    )
    if missing_responses:
        raise ValueError(
            "Source results do not match the annotation key; missing response_id values: "
            + ", ".join(missing_responses[:5])
        )

    def calculate(selected_roles: set[str] | None) -> dict[str, Any]:
        comparisons: dict[tuple[str, str], list[tuple[Any, Any]]] = defaultdict(list)
        for item_id, item_annotations in by_item.items():
            if selected_roles is not None and item_role[item_id] not in selected_roles:
                continue
            systems = judge_scores.get(item_response[item_id], {})
            for field in JUDGE_COMPARABLE_FIELDS:
                human = _consensus(
                    [
                        _expert_numeric_value(field, annotation.get(field))
                        for annotation in item_annotations
                    ],
                    binary=field == "harmful_elaboration",
                )
                if human is None:
                    continue
                for system, scores in systems.items():
                    judge = scores.get(field)
                    if judge is not None:
                        comparisons[(system, field)].append((human, judge))
        report: dict[str, Any] = defaultdict(dict)
        for (system, field), paired in sorted(comparisons.items()):
            binary = field == "harmful_elaboration"
            scale = ORDINAL_EXPERT_FIELDS.get(field)
            if field == "appropriate_redirection":
                scale = (0, 3)
            elif field == "response_appropriateness":
                scale = (0, 2)

            def coefficient(
                values: list[tuple[Any, Any]],
                *,
                binary_metric: bool = binary,
                metric_scale: tuple[int, int] | None = scale,
            ) -> float | None:
                if binary_metric:
                    return _cohen_kappa([a for a, _ in values], [b for _, b in values])
                return _quadratic_weighted_kappa(
                    [int(a) for a, _ in values],
                    [int(b) for _, b in values],
                    scale=metric_scale,
                )

            def exact(values: list[tuple[Any, Any]]) -> float:
                return sum(a == b for a, b in values) / len(values)

            def mae(values: list[tuple[Any, Any]]) -> float:
                return sum(abs(float(a) - float(b)) for a, b in values) / len(values)

            seed = bootstrap_seed + sum(ord(char) for char in f"{system}:{field}")
            report[system][field] = {
                "n_items": len(paired),
                "exact_agreement": exact(paired),
                "exact_agreement_ci_95": _bootstrap_pair_ci(
                    paired, exact, iterations=bootstrap_iterations, seed=seed
                ),
                "kappa": coefficient(paired),
                "kappa_ci_95": _bootstrap_pair_ci(
                    paired, coefficient, iterations=bootstrap_iterations, seed=seed + 1
                ),
                "mean_absolute_error": mae(paired),
                "mean_absolute_error_ci_95": _bootstrap_pair_ci(
                    paired, mae, iterations=bootstrap_iterations, seed=seed + 2
                ),
                "spearman": _spearman(
                    [float(a) for a, _ in paired], [float(b) for _, b in paired]
                ),
            }
        return dict(report)

    roles = sorted(set(item_role.values()))
    return {
        "consensus_rule": (
            "Item-level median for ordinal fields only when it is an integer; majority for "
            "harmful_elaboration. Half-step medians, binary ties, and not-applicable values are "
            "excluded pending adjudication."
        ),
        "all": calculate(None),
        "by_sample_role": {role: calculate({role}) for role in roles},
    }


def build_summary(
    expert_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 20260831,
) -> dict[str, Any]:
    pair_fields = [question["id"] for question in PAIR_QUESTIONS]
    ordinal_fields = {
        *EXPERT_ORDINAL_SCALES,
        "first_amplification_turn",
        "amplification_turn_count",
    }
    sample_roles = sorted(
        {str(row.get("sample_role") or "unknown") for row in expert_rows}
    )
    return {
        "schema_version": "2.0",
        "expert_annotation_count": len(expert_rows),
        "student_annotation_count": len(pair_rows),
        "expert_annotators": sorted({row["annotator_id"] for row in expert_rows}),
        "student_annotators": sorted({row["annotator_id"] for row in pair_rows}),
        "expert_coverage": annotation_coverage(expert_rows, EXPERT_AGREEMENT_FIELDS),
        "student_coverage": annotation_coverage(pair_rows, pair_fields),
        "expert_cannot_determine_count": sum(
            bool(row["annotation"].get("cannot_determine")) for row in expert_rows
        ),
        "expert_agreement": agreement_by_field(
            expert_rows,
            EXPERT_AGREEMENT_FIELDS,
            ordinal_fields=ordinal_fields,
            ordinal_scales=EXPERT_ORDINAL_SCALES,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "expert_agreement_by_sample_role": {
            role: agreement_by_field(
                [
                    row
                    for row in expert_rows
                    if str(row.get("sample_role") or "unknown") == role
                ],
                EXPERT_AGREEMENT_FIELDS,
                ordinal_fields=ordinal_fields,
                ordinal_scales=EXPERT_ORDINAL_SCALES,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
            for role in sample_roles
        },
        "student_pair_agreement": agreement_by_field(
            pair_rows,
            pair_fields,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "student_preferences": student_preference_summary(pair_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--expert", type=Path, action="append", default=[])
    parser.add_argument("--student", type=Path, action="append", default=[])
    parser.add_argument(
        "--assignment-manifest",
        type=Path,
        help="Frozen ledger from assign_annotations.py; returned ratings are checked against it.",
    )
    parser.add_argument(
        "--roster-manifest",
        type=Path,
        help="Frozen eligibility roster; required for formal or calibration assignments.",
    )
    parser.add_argument(
        "--allow-incomplete-assignments",
        action="store_true",
        help="Permit missing planned ratings for interim analysis; unexpected ratings still fail.",
    )
    parser.add_argument(
        "--source-results",
        type=Path,
        help="Frozen benchmark results used to compute expert-human versus judge validity.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keys = load_private_key(args.private_key)
    assignment_manifest = validate_collection_manifests(
        keys, args.assignment_manifest, args.roster_manifest
    )
    expert = import_rows(args.expert, keys, "expert_conversation")
    student = import_rows(args.student, keys, "paired_conversation")
    if not expert and not student:
        raise ValueError("Provide at least one --expert or --student annotation file")
    summary = build_summary(
        expert,
        student,
        bootstrap_iterations=max(0, args.bootstrap_iterations),
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.assignment_manifest:
        if assignment_manifest is None:
            raise ValueError("Assignment manifest could not be validated")
        included_types = set()
        if args.expert:
            included_types.add("expert_conversation")
        if args.student:
            included_types.add("paired_conversation")
        audit = assignment_audit(
            expert,
            student,
            assignment_manifest,
            included_types=included_types,
        )
        if audit["unexpected_assignment_count"]:
            raise ValueError(
                f"Returned files contain {audit['unexpected_assignment_count']} unassigned ratings"
            )
        if audit["missing_assignment_count"] and not args.allow_incomplete_assignments:
            raise ValueError(
                f"Returned files are missing {audit['missing_assignment_count']} planned ratings; "
                "use --allow-incomplete-assignments only for interim analysis"
            )
        summary["assignment_audit"] = audit
    if args.source_results:
        summary["human_judge_validity"] = human_judge_validity(
            expert,
            load_judge_scores(args.source_results),
            bootstrap_iterations=max(0, args.bootstrap_iterations),
            bootstrap_seed=args.bootstrap_seed,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joined_path = args.out_dir / "annotations_joined.private.jsonl"
    summary_path = args.out_dir / "agreement_summary.json"
    write_jsonl(joined_path, [*expert, *student])
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(joined_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
