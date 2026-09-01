"""Annotation-readiness checks for materialized V3 scenarios."""

from __future__ import annotations

from typing import Any


def annotation_readiness_errors(scenario: dict[str, Any]) -> list[str]:
    errors = []
    if scenario.get("fact_check_status") not in {
        "source_reviewed",
        "verified",
        "not_applicable",
    }:
        errors.append(
            "fact_check_status must be source_reviewed, verified, or not_applicable"
        )
    if scenario.get("review_status") not in {
        "ai_author_reviewed",
        "approved",
        "expert_approved",
    }:
        errors.append(
            "review_status must be ai_author_reviewed, approved, or expert_approved"
        )
    return errors
