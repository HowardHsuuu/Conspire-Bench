"""Shared public types and credential resolution for Conspire-Bench."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"


class ScenarioType(Enum):
    MULTI_TURN = "multi_turn_progression"
    SINGLE_TURN = "single_turn_complete_logic"
    RESISTANCE = "complete_logic_then_resistance"


API_KEY_ENV_VARS = {
    ModelProvider.OPENAI.value: "OPENAI_API_KEY",
    ModelProvider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
    ModelProvider.GEMINI.value: "GEMINI_API_KEY",
}


class ModelText(str):
    """String-compatible model output carrying non-prompt request metadata."""

    metadata: dict[str, Any]

    def __new__(cls, value: str, metadata: dict[str, Any] | None = None) -> ModelText:
        instance = super().__new__(cls, value)
        instance.metadata = metadata or {}
        return instance


def serializable_metadata(value: Any) -> Any:
    """Convert SDK metadata objects into JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def is_placeholder_api_key(api_key: str | None) -> bool:
    if not api_key:
        return True

    normalized = api_key.strip().lower()
    return (
        normalized.startswith("your-")
        or normalized in {"", "none", "null", "changeme", "replace-me"}
        or "api-key-here" in normalized
    )


def resolve_api_key(
    config: dict[str, Any], provider: ModelProvider | str
) -> str | None:
    """Resolve a provider key without logging or persisting its value."""

    provider_value = provider.value if isinstance(provider, ModelProvider) else provider
    configured_key = config.get("api_keys", {}).get(provider_value)

    if configured_key:
        configured_key = str(configured_key).strip()
        if configured_key.startswith("env:"):
            return os.environ.get(configured_key[4:].strip())
        if configured_key.startswith("$"):
            return os.environ.get(configured_key[1:].strip())
        if not is_placeholder_api_key(configured_key):
            return configured_key

    env_var = API_KEY_ENV_VARS.get(provider_value)
    return os.environ.get(env_var) if env_var else None


@dataclass
class TestResult:
    scenario_id: str
    model_name: str
    judge_name: str
    scenario_type: str
    category: str
    scenario_subcategory: str | None
    conversation_log: list[dict[str, Any]]
    judge_scores: dict[str, Any]
    judge_reasoning: dict[str, Any]
    judge_results: list[dict[str, Any]]
    judge_disagreement: dict[str, float]
    total_safety_score: float | None
    timestamp: str
    context_setting: str | None = None
    context_label: str | None = None
    error: str | None = None
    schema_version: str = "2.0"
    condition_id: str | None = None
    response_id: str | None = None
    frame_family: str | None = None
    prompt_variant_id: str | None = None
    prompt_schema_version: str | None = None
    generation_seed: int | None = None
    replicate_id: int = 0
    generation_config: dict[str, Any] | None = None
    generation_interface: str | None = None
    access_date: str | None = None
    primary_judge_names: list[str] | None = None
    same_family_excluded: bool = False
    scenario_metadata: dict[str, Any] | None = None


@dataclass
class JudgeMetrics:
    conspiracy_amplification: float | None = None
    reality_grounding: float | None = None
    harmful_elaboration: bool | None = None
    appropriate_uncertainty: float | None = None
    overall_safety: float | None = None
    belief_amplification: float | None = None
    epistemic_grounding: float | None = None
    uncertainty_calibration: float | None = None
    supportive_engagement: float | None = None
    appropriate_redirection: float | None = None
    response_appropriateness: float | None = None
    harmful_elaboration_subtype: str | None = None
    reasoning: dict[str, str] = field(default_factory=dict)
    rubric_version: str = "1.0"

    def scores(self) -> dict[str, Any]:
        values = {
            "conspiracy_amplification": self.conspiracy_amplification,
            "reality_grounding": self.reality_grounding,
            "harmful_elaboration": self.harmful_elaboration,
            "appropriate_uncertainty": self.appropriate_uncertainty,
            "overall_safety": self.overall_safety,
            "belief_amplification": self.belief_amplification,
            "epistemic_grounding": self.epistemic_grounding,
            "uncertainty_calibration": self.uncertainty_calibration,
            "supportive_engagement": self.supportive_engagement,
            "appropriate_redirection": self.appropriate_redirection,
            "response_appropriateness": self.response_appropriateness,
            "harmful_elaboration_subtype": self.harmful_elaboration_subtype,
        }
        return {key: value for key, value in values.items() if value is not None}
