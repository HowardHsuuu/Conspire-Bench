from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


ALLOWED_CATEGORIES = {
    "aliens_ufo",
    "modern_health",
    "government_control",
    "ai_consciousness",
    "secret_societies",
    "religious_apocalyptic",
    "technology_control",
    "mixed_theories",
}

ALLOWED_SCENARIO_TYPES = {
    "multi_turn_progression",
    "single_turn_complete_logic",
    "complete_logic_then_resistance",
}

METADATA_FIELDS = {
    "risk_level",
    "belief_domain",
    "pressure_type",
    "grounding_trap",
    "expected_safe_behavior",
}


@dataclass
class DatasetValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    scenario_count: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    scenario_types: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_dataset(
    dataset: Dict[str, Any],
    *,
    strict_metadata: bool = False,
) -> DatasetValidationReport:
    report = DatasetValidationReport()

    scenarios = dataset.get("scenarios")
    if not isinstance(scenarios, list):
        report.errors.append("Top-level 'scenarios' must be a list")
        return report

    report.scenario_count = len(scenarios)
    seen_ids = set()

    for index, scenario in enumerate(scenarios):
        label = scenario.get("id", f"<index:{index}>") if isinstance(scenario, dict) else f"<index:{index}>"
        if not isinstance(scenario, dict):
            report.errors.append(f"{label}: scenario must be an object")
            continue

        _validate_required_string(report, scenario, label, "id")
        _validate_required_string(report, scenario, label, "type")
        _validate_required_string(report, scenario, label, "category")

        scenario_id = scenario.get("id")
        if scenario_id in seen_ids:
            report.errors.append(f"{label}: duplicate scenario id")
        seen_ids.add(scenario_id)

        scenario_type = scenario.get("type")
        category = scenario.get("category")

        if scenario_type not in ALLOWED_SCENARIO_TYPES:
            report.errors.append(f"{label}: unsupported type '{scenario_type}'")
        else:
            report.scenario_types[scenario_type] = report.scenario_types.get(scenario_type, 0) + 1

        if category not in ALLOWED_CATEGORIES:
            report.errors.append(f"{label}: unsupported category '{category}'")
        else:
            report.categories[category] = report.categories.get(category, 0) + 1

        if scenario_type == "single_turn_complete_logic":
            _validate_required_string(report, scenario, label, "user")
        elif scenario_type in {"multi_turn_progression", "complete_logic_then_resistance"}:
            _validate_conversation(report, scenario, label)

        if "subcategory" in scenario and not isinstance(scenario["subcategory"], str):
            report.errors.append(f"{label}: 'subcategory' must be a string when present")

        missing_metadata = sorted(field for field in METADATA_FIELDS if field not in scenario)
        if missing_metadata:
            message = f"{label}: missing expansion metadata fields: {', '.join(missing_metadata)}"
            if strict_metadata:
                report.errors.append(message)
            else:
                report.warnings.append(message)
        else:
            _validate_metadata_fields(report, scenario, label)

    _validate_metadata_counts(dataset, report)
    return report


def _validate_required_string(
    report: DatasetValidationReport,
    scenario: Dict[str, Any],
    label: str,
    field_name: str,
):
    value = scenario.get(field_name)
    if not isinstance(value, str) or not value.strip():
        report.errors.append(f"{label}: '{field_name}' must be a non-empty string")


def _validate_conversation(
    report: DatasetValidationReport,
    scenario: Dict[str, Any],
    label: str,
):
    conversation = scenario.get("conversation")
    if not isinstance(conversation, list) or not conversation:
        report.errors.append(f"{label}: 'conversation' must be a non-empty list")
        return

    for turn_index, turn in enumerate(conversation):
        turn_label = f"{label}.conversation[{turn_index}]"
        if not isinstance(turn, dict):
            report.errors.append(f"{turn_label}: turn must be an object")
            continue
        user_text = turn.get("user")
        if not isinstance(user_text, str) or not user_text.strip():
            report.errors.append(f"{turn_label}: 'user' must be a non-empty string")
        if "turn" in turn and not isinstance(turn["turn"], int):
            report.errors.append(f"{turn_label}: 'turn' must be an integer when present")


def _validate_metadata_fields(
    report: DatasetValidationReport,
    scenario: Dict[str, Any],
    label: str,
):
    for field_name in ("risk_level", "belief_domain", "pressure_type", "grounding_trap"):
        if not isinstance(scenario.get(field_name), str) or not scenario[field_name].strip():
            report.errors.append(f"{label}: '{field_name}' must be a non-empty string")

    expected = scenario.get("expected_safe_behavior")
    if not isinstance(expected, list) or not expected:
        report.errors.append(f"{label}: 'expected_safe_behavior' must be a non-empty list")
    elif not all(isinstance(item, str) and item.strip() for item in expected):
        report.errors.append(f"{label}: 'expected_safe_behavior' items must be non-empty strings")


def _validate_metadata_counts(dataset: Dict[str, Any], report: DatasetValidationReport):
    metadata = dataset.get("metadata", {})
    if not isinstance(metadata, dict):
        report.warnings.append("Top-level 'metadata' should be an object")
        return

    total = metadata.get("total_scenarios")
    if isinstance(total, int) and total != report.scenario_count:
        report.warnings.append(
            f"metadata.total_scenarios={total} but found {report.scenario_count} scenarios"
        )

    category_counts = metadata.get("categories")
    if isinstance(category_counts, dict):
        for category, observed in sorted(report.categories.items()):
            expected = category_counts.get(category)
            if isinstance(expected, int) and expected != observed:
                report.warnings.append(
                    f"metadata.categories.{category}={expected} but found {observed}"
                )

    type_counts = metadata.get("scenario_types")
    if isinstance(type_counts, dict):
        expected_by_type = {
            "multi_turn_progression": type_counts.get("multi_turn_progression"),
            "single_turn_complete_logic": type_counts.get("single_turn_complete_logic"),
            "complete_logic_then_resistance": type_counts.get("complete_logic_then_resistance"),
        }
        for scenario_type, observed in sorted(report.scenario_types.items()):
            expected = expected_by_type.get(scenario_type)
            if isinstance(expected, int) and expected != observed:
                report.warnings.append(
                    f"metadata.scenario_types.{scenario_type}={expected} but found {observed}"
                )


def format_validation_report(
    report: DatasetValidationReport,
    *,
    max_warnings: Optional[int] = None,
) -> str:
    lines = [
        "Dataset validation",
        f"- status: {'ok' if report.ok else 'failed'}",
        f"- scenarios: {report.scenario_count}",
    ]

    if report.categories:
        categories = ", ".join(
            f"{category}={count}" for category, count in sorted(report.categories.items())
        )
        lines.append(f"- categories: {categories}")

    if report.scenario_types:
        scenario_types = ", ".join(
            f"{scenario_type}={count}" for scenario_type, count in sorted(report.scenario_types.items())
        )
        lines.append(f"- types: {scenario_types}")

    if report.errors:
        lines.append("Errors:")
        lines.extend(f"- {error}" for error in report.errors)

    if report.warnings:
        lines.append("Warnings:")
        visible_warnings = report.warnings
        if max_warnings is not None:
            visible_warnings = report.warnings[:max_warnings]
        lines.extend(f"- {warning}" for warning in visible_warnings)
        if max_warnings is not None and len(report.warnings) > max_warnings:
            lines.append(f"- ... {len(report.warnings) - max_warnings} more warnings")

    return "\n".join(lines)
