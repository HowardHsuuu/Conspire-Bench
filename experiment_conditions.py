from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_CONTEXT_VARIANTS_PATH = Path("configs/context_variants.json")


@dataclass(frozen=True)
class ContextCondition:
    variant_id: str
    frame: str
    text: Optional[str]
    canonical: bool = False
    study_role: str = "unspecified"
    prompt_schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.variant_id

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def adhoc_context_condition(label: str, text: Optional[str]) -> ContextCondition:
    """Create a non-registered condition for an exploratory CLI run."""
    frame = "neutral" if label == "none" else label
    return ContextCondition(
        variant_id=label,
        frame=frame,
        text=text,
        canonical=True,
        study_role="exploratory_cli",
        prompt_schema_version="adhoc-1.0",
    )


def normalize_context_condition(value: Any) -> ContextCondition:
    if isinstance(value, ContextCondition):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return adhoc_context_condition(str(value[0]), value[1])
    if isinstance(value, dict):
        return ContextCondition(
            variant_id=str(
                value.get("variant_id") or value.get("id") or value.get("label")
            ),
            frame=str(
                value.get("frame") or value.get("frame_family") or value.get("label")
            ),
            text=value.get("text", value.get("setting")),
            canonical=bool(value.get("canonical", False)),
            study_role=str(value.get("study_role", "unspecified")),
            prompt_schema_version=str(value.get("prompt_schema_version", "1.0")),
            metadata=dict(value.get("metadata") or {}),
        )
    raise TypeError(f"Unsupported context condition: {value!r}")


def normalize_context_conditions(values: Iterable[Any]) -> list[ContextCondition]:
    return [normalize_context_condition(value) for value in values]


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Context-variant config must be a JSON object: {path}")
    return payload


def load_context_conditions(
    path: str | Path = DEFAULT_CONTEXT_VARIANTS_PATH,
) -> dict[str, ContextCondition]:
    payload = _read_json(path)
    schema_version = str(payload.get("schema_version", "1.0"))
    variants = payload.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError(
            "Context-variant config must contain a non-empty 'variants' list"
        )

    conditions: dict[str, ContextCondition] = {}
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"variants[{index}] must be an object")
        variant_id = variant.get("id")
        frame = variant.get("frame")
        if not isinstance(variant_id, str) or not variant_id.strip():
            raise ValueError(f"variants[{index}].id must be a non-empty string")
        if variant_id in conditions:
            raise ValueError(f"Duplicate context variant id: {variant_id}")
        if not isinstance(frame, str) or not frame.strip():
            raise ValueError(f"variants[{index}].frame must be a non-empty string")
        text = variant.get("text")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError(
                f"variants[{index}].text must be null or a non-empty string"
            )

        known = {"id", "frame", "text", "canonical", "study_role"}
        metadata = {key: value for key, value in variant.items() if key not in known}
        conditions[variant_id] = ContextCondition(
            variant_id=variant_id,
            frame=frame,
            text=text,
            canonical=bool(variant.get("canonical", False)),
            study_role=str(variant.get("study_role", "unspecified")),
            prompt_schema_version=schema_version,
            metadata=metadata,
        )
    return conditions


def load_context_set(
    set_name: str,
    path: str | Path = DEFAULT_CONTEXT_VARIANTS_PATH,
) -> list[ContextCondition]:
    payload = _read_json(path)
    sets = payload.get("sets")
    if not isinstance(sets, dict) or set_name not in sets:
        available = ", ".join(sorted(sets)) if isinstance(sets, dict) else "none"
        raise ValueError(
            f"Unknown context set '{set_name}'. Available sets: {available}"
        )
    variant_ids = sets[set_name]
    if not isinstance(variant_ids, list) or not variant_ids:
        raise ValueError(f"Context set '{set_name}' must be a non-empty list")
    conditions = load_context_conditions(path)
    missing = [variant_id for variant_id in variant_ids if variant_id not in conditions]
    if missing:
        raise ValueError(
            f"Context set '{set_name}' references unknown variants: {', '.join(missing)}"
        )
    return [conditions[variant_id] for variant_id in variant_ids]


def select_context_conditions(
    variant_ids: Iterable[str],
    path: str | Path = DEFAULT_CONTEXT_VARIANTS_PATH,
) -> list[ContextCondition]:
    conditions = load_context_conditions(path)
    selected = []
    for variant_id in variant_ids:
        if variant_id not in conditions:
            raise ValueError(f"Unknown context variant: {variant_id}")
        selected.append(conditions[variant_id])
    return selected


def stable_digest(payload: Any, *, length: int = 16) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def condition_id(
    *,
    scenario_id: str,
    model_name: str,
    context: ContextCondition,
    generation_seed: Optional[int] = None,
    replicate_id: int = 0,
    generation_config: Optional[dict[str, Any]] = None,
) -> str:
    digest = stable_digest(
        {
            "scenario_id": scenario_id,
            "model_name": model_name,
            "prompt_variant_id": context.variant_id,
            "generation_seed": generation_seed,
            "replicate_id": replicate_id,
            "generation_config": generation_config or {},
        }
    )
    return f"cond_{digest}"


def response_id(**kwargs: Any) -> str:
    return condition_id(**kwargs).replace("cond_", "resp_", 1)
