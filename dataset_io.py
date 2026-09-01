"""Dataset loading for the materialized Conspire-Bench V3 JSON release."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_benchmark_dataset(path: str | Path) -> dict[str, Any]:
    """Load one materialized benchmark dataset.

    V3 publishes executable scenarios directly; authoring catalogs are validated
    separately and are never expanded implicitly at experiment runtime.
    """

    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset must be a JSON object: {dataset_path}")
    if not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"Dataset must contain a scenarios list: {dataset_path}")
    return payload
