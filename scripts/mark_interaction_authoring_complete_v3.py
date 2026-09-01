#!/usr/bin/env python3
"""Record validated v3 interaction authoring in the frozen motif manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "primary_motif_manifest_v3.json"
DEFAULT_CATALOG = ROOT / "configs" / "interaction_catalog_v3.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    catalog_ids = [item["motif_id"] for item in catalog.get("motifs", [])]
    manifest_ids = [item["motif_id"] for item in manifest.get("motifs", [])]
    if catalog_ids != manifest_ids or len(catalog_ids) != 51:
        raise ValueError("validated catalog must exactly match the 51-item manifest")
    if args.check:
        ok = (
            all(
                item.get("authoring_status") == "authored_validated"
                for item in manifest.get("motifs", [])
            )
            and manifest.get("interaction_catalog")
            == "configs/interaction_catalog_v3.json"
        )
        print(json.dumps({"ok": ok, "motif_count": len(manifest_ids)}, indent=2))
        return 0 if ok else 1
    for item in manifest["motifs"]:
        item["authoring_status"] = "authored_validated"
    manifest["interaction_catalog"] = "configs/interaction_catalog_v3.json"
    manifest["interaction_authoring"] = {
        "status": "authored_validated",
        "motif_count": 51,
        "scenario_count": 153,
        "structures_per_motif": 3,
        "validated_at": "2026-09-01",
        "validator": "scripts/validate_interaction_catalog_v3.py",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ok": True, "motif_count": 51}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
