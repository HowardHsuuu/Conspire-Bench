#!/usr/bin/env python3
"""Report whether the public V3 code contract is ready for external evidence runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.frame_effect_stats import build_v3_coequal_report
from dataset_validation import validate_dataset
from scripts.validate_analysis_plan_v3 import validate as validate_analysis_plan
from scripts.validate_context_variants_v3 import validate_context_variants
from scripts.validate_experiment_configs_v3 import validate as validate_experiments
from scripts.validate_motif_narratives_v3 import validate_records
from scripts.validate_motif_quality_review_v3 import validate_review
from scripts.validate_primary_motif_manifest import validate_manifest


def _read(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _analysis_runtime_contract_errors() -> list[str]:
    scores = {
        "belief_amplification": 1,
        "epistemic_grounding": 4,
        "uncertainty_calibration": 4,
        "harmful_elaboration": False,
        "supportive_engagement": 4,
        "appropriate_redirection": 2,
        "response_appropriateness": 0,
    }
    variants = [
        ("neutral", "neutral_none"),
        *[
            (frame, f"{frame}_v{index}")
            for frame in (
                "brainstorming",
                "critical_review",
                "sensemaking",
                "supportive_listening",
            )
            for index in range(1, 5)
        ],
    ]
    rows = [
        {
            "model_name": "contract/model",
            "scenario_id": "contract_scenario",
            "scenario_type": "single_turn_complete_logic",
            "frame_family": frame,
            "prompt_variant_id": variant,
            "generation_seed": 1,
            "replicate_id": 0,
            "judge_scores": scores,
            "error": None,
        }
        for frame, variant in variants
    ]
    try:
        report = build_v3_coequal_report(rows, draws=10, seed=1, canonical_only=False)
    except Exception as error:
        return [f"V3 coequal analysis failed its synthetic contract: {error}"]
    errors: list[str] = []
    if report.get("estimand_count") != 28:
        errors.append("V3 analysis must emit exactly 28 coequal estimands")
    contrasts = [
        contrast
        for outcome in report.get("outcome_reports", [])
        for contrast in outcome.get("contrasts", [])
    ]
    if any(
        "fdr_bh_adjusted_p_value" not in (contrast.get("motif_level_sign_test") or {})
        for contrast in contrasts
    ):
        errors.append("Every V3 estimand must carry the common BH-FDR adjustment")
    if any(
        outcome.get("wording_aggregation")
        != "equal_mean_of_four_variants_nested_within_frame_family"
        for outcome in report.get("outcome_reports", [])
    ):
        errors.append("Full V3 analysis must average four wordings within frame family")
    sensitivity_estimands = sum(
        len(outcome.get("contrasts", []))
        for outcome in (report.get("overlap_cluster_sensitivity") or {}).get(
            "outcome_reports", []
        )
    )
    if sensitivity_estimands != 28:
        errors.append("Overlap-cluster sensitivity must repeat all 28 effect summaries")
    return errors


def build_report() -> dict[str, Any]:
    manifest = _read("configs/primary_motif_manifest_v3.json")
    narratives = _read("configs/motif_narratives_v3.json")
    quality = _read("configs/motif_quality_review_v3.json")
    dataset = _read("Conspire-Bench-v3.json")
    contexts = _read("configs/context_variants.json")
    local_config = _read("configs/experiment_v3_local_full.json")
    api_config = _read("configs/experiment_v3_api_full.json")
    analysis_plan = _read("configs/analysis_plan_v3.json")
    human_plan = _read("configs/human_annotation_plan_v2.json")

    checks: dict[str, list[str]] = {
        "motif_manifest": validate_manifest(manifest),
        "circulation_grounding": validate_records(narratives, manifest),
        "motif_quality_boundaries": validate_review(quality, manifest, narratives),
        "dataset": validate_dataset(dataset, strict_metadata=True).errors,
        "framing_and_nested_paraphrases": validate_context_variants(contexts),
        "model_and_judge_matrices": validate_experiments(local_config, api_config),
        "analysis_contract": validate_analysis_plan(analysis_plan),
    }

    required_analysis_tools = (
        "analysis/frame_effect_stats.py",
        "analysis/paraphrase_robustness.py",
        "analysis/judge_family_sensitivity.py",
        "analysis/rubric_consistency_audit.py",
        "analysis/api_usage_report.py",
        "analysis/export_results.py",
    )
    checks["analysis_implementations"] = [
        f"missing public analysis tool: {path}"
        for path in required_analysis_tools
        if not (ROOT / path).is_file()
    ]
    checks["coequal_statistics_runtime"] = _analysis_runtime_contract_errors()

    required_annotation_tools = (
        "analysis/export_rubric_feedback.py",
        "analysis/export_calibration.py",
        "analysis/freeze_response_pool.py",
        "analysis/export_annotations.py",
        "analysis/assign_annotations.py",
        "analysis/import_annotations.py",
        "analysis/scenario_review.py",
        "analysis/freeze_human_annotation_plan.py",
        "analysis/freeze_analysis_plan.py",
    )
    checks["annotation_workflow"] = [
        f"missing annotation workflow tool: {path}"
        for path in required_annotation_tools
        if not (ROOT / path).is_file()
    ]

    contract_errors = [
        f"{check}: {error}" for check, errors in checks.items() for error in errors
    ]
    return {
        "release_state": (
            "code_ready_pending_external_evidence"
            if not contract_errors
            else "code_contract_failed"
        ),
        "code_ready": not contract_errors,
        "design": {
            "motifs": 51,
            "scenarios": 153,
            "interaction_structures": 3,
            "frame_families": 5,
            "wording_conditions": 17,
            "rubric_outcomes": 7,
            "coequal_estimands": 28,
        },
        "checks": {
            name: {"ok": not errors, "errors": errors}
            for name, errors in checks.items()
        },
        "contract_errors": contract_errors,
        "pending_live_evidence": [
            "Generate the open-source target response matrix and local diagnostic judges.",
            "Preflight current API model IDs, then run the API pilot and full target matrix.",
            "Run the final cross-provider judge panel on the frozen response pool.",
            "Produce result-dependent frame effects, paraphrase robustness, confidence intervals, judge sensitivity, usage, tables, and figures.",
        ],
        "pending_human_evidence": [
            "Complete independent scenario QA if it will support the manuscript's construction-validity claim.",
            "Obtain supervisor or ethics-process approval for consent, compensation, data retention, and annotator welfare.",
            "Run expert rubric content validation and timed calibration, then freeze rubric, workload, and analysis plans.",
            "Collect the frozen expert conversation ratings and student paired-frame ratings, then import and analyze them.",
        ],
        "human_plan_status": human_plan.get("status"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; the report is always printed to stdout.",
    )
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["code_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
