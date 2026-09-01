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

from analysis.api_usage_report import build_report as build_usage_report
from analysis.export_results import model_variant_summary_rows
from analysis.frame_effect_stats import build_v3_coequal_report
from analysis.judge_family_sensitivity import (
    build_report as build_judge_sensitivity_report,
)
from analysis.make_figures import effect_rows as figure_effect_rows
from analysis.paraphrase_robustness import build_report as build_paraphrase_report
from analysis.rubric_consistency_audit import (
    build_report as build_consistency_report,
)
from analysis.scenario_review import load_artifacts as load_scenario_review_artifacts
from analysis.scenario_review import review_items as build_scenario_review_items
from dataset_validation import validate_dataset
from scripts.validate_analysis_plan_v3 import validate as validate_analysis_plan
from scripts.validate_context_variants_v3 import validate_context_variants
from scripts.validate_experiment_configs_v3 import validate as validate_experiments
from scripts.validate_motif_narratives_v3 import validate_records
from scripts.validate_motif_quality_review_v3 import validate_review
from scripts.validate_primary_motif_manifest import validate_manifest

PUBLIC_MARKDOWN = {
    "README.md",
    "docs/annotation.md",
    "docs/dataset.md",
    "docs/evaluation.md",
    "docs/rubric.md",
}

OBSOLETE_PUBLIC_PATHS = (
    "legacy",
    "CONSPIRE-Bench.json",
    "adversarial_testing.py",
    "scenario_expansion.py",
    "scenario_metadata.py",
    "analysis/select_scenario_subset.py",
    "results/20260515_163319",
    "scripts/run_v2_stage.py",
    "docs/local_models.md",
    "docs/runpod_5090_setup.md",
    "annotation_ui/README.md",
    "annotation/rubric_validity_v2",
    "configs/analysis_plan_v2.json",
    "configs/experiment_v2_api_full.json",
    "configs/experiment_v2_api_pilot.json",
    "configs/experiment_v2_api_robustness.json",
    "configs/scenario_expansion_v2.json",
    "configs/scenario_metadata_v2.json",
    "configs/scenario_source_packets_v2.json",
)


def _public_tree_errors() -> list[str]:
    errors = [
        f"obsolete public path must be absent: {path}"
        for path in OBSOLETE_PUBLIC_PATHS
        if (ROOT / path).exists()
    ]
    markdown = {
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*.md")
        if not ({".git", ".local", "results"} & set(path.parts))
    }
    if markdown != PUBLIC_MARKDOWN:
        errors.append(
            "public Markdown set must be exactly: " + ", ".join(sorted(PUBLIC_MARKDOWN))
        )

    forbidden_runtime_fields = (
        "total_safety_score",
        "overall_safety_score",
        "conspiracy_amplification",
        "reality_grounding",
        "appropriate_uncertainty",
    )
    runtime_files = (
        "bench_runner.py",
        "benchmark_types.py",
        "result_reporting.py",
        "analysis/export_results.py",
    )
    for path in runtime_files:
        source = (ROOT / path).read_text(encoding="utf-8")
        for field in forbidden_runtime_fields:
            if field in source:
                errors.append(f"{path} retains obsolete runtime field {field}")
    return errors


def _read(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _analysis_runtime_contract_errors() -> list[str]:
    scores: dict[str, Any] = {
        "belief_amplification": 1,
        "epistemic_grounding": 4,
        "uncertainty_calibration": 4,
        "harmful_elaboration": False,
        "harmful_elaboration_subtype": "none",
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
            "response_id": f"contract_{variant}",
            "conversation_log": [
                {
                    "role": "assistant",
                    "content": "Synthetic contract response.",
                    "response_metadata": {
                        "provider": "openai",
                        "requested_model": "contract-target",
                        "resolved_model": "contract-target-resolved",
                        "interface": "responses",
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    },
                }
            ],
            "judge_results": [
                {
                    "judge_name": "same-family-contract-judge",
                    "provider": "openai",
                    "model": "same-family-judge",
                    "same_family_as_target": True,
                    "scores": scores,
                    "error": None,
                    "response_metadata": {
                        "provider": "openai",
                        "requested_model": "same-family-judge",
                        "resolved_model": "same-family-judge-resolved",
                        "interface": "responses",
                        "usage": {"input_tokens": 20, "output_tokens": 8},
                    },
                },
                {
                    "judge_name": "cross-family-contract-judge",
                    "provider": "anthropic",
                    "model": "cross-family-judge",
                    "same_family_as_target": False,
                    "scores": scores,
                    "error": None,
                    "response_metadata": {
                        "provider": "anthropic",
                        "requested_model": "cross-family-judge",
                        "resolved_model": "cross-family-judge-resolved",
                        "interface": "messages_api",
                        "usage": {"input_tokens": 20, "output_tokens": 8},
                    },
                },
            ],
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
    try:
        if len(figure_effect_rows(report)) != 28:
            errors.append("V3 figure extraction must preserve all 28 estimands")
        table_rows = model_variant_summary_rows(rows)
        if len(table_rows) != 17:
            errors.append("V3 table export must preserve all 17 wording conditions")
        elif any("sd_belief_amplification" not in row for row in table_rows):
            errors.append("V3 descriptive tables must include standard deviations")
        paraphrase = build_paraphrase_report(rows, "belief_amplification")
        if len(paraphrase.get("family_summary", [])) != 4:
            errors.append(
                "Paraphrase analysis must summarize four non-neutral families"
            )
        sensitivity = build_judge_sensitivity_report(rows)
        if not sensitivity.get("paired_comparisons"):
            errors.append("Judge-family sensitivity must emit paired comparisons")
        usage = build_usage_report(rows)
        if usage.get("request_count") != len(rows) * 3:
            errors.append("Usage report must retain target and per-judge requests")
        consistency = build_consistency_report(rows)
        if not str(consistency.get("policy") or "").startswith("Diagnostic only"):
            errors.append("Rubric consistency checks must remain post-hoc diagnostics")
    except Exception as error:
        errors.append(f"V3 downstream analysis pipeline failed: {error}")
    return errors


def _reviewer_revision_coverage() -> list[dict[str, Any]]:
    """Public traceability from the ARR concerns to V3 evidence boundaries."""

    return [
        {
            "issue": "benchmark_size_and_diversity",
            "status": "implemented_pending_live_evidence",
            "evidence": [
                "Conspire-Bench-v3.json: 51 motifs, 153 scenarios, 24 categories",
                "configs/primary_motif_manifest_v3.json",
            ],
        },
        {
            "issue": "scenario_construction_quality_control",
            "status": "implemented_plus_pending_independent_human_qa",
            "evidence": [
                "configs/motif_narratives_v3.json",
                "configs/motif_quality_review_v3.json",
                "configs/interaction_identity_policy_v3.json",
                "analysis/scenario_review.py",
            ],
        },
        {
            "issue": "human_validation_and_human_judge_agreement",
            "status": "workflow_implemented_pending_human_evidence",
            "evidence": [
                "configs/human_annotation_plan_v2.json",
                "docs/annotation.md",
                "analysis/import_annotations.py",
                "analysis/import_rubric_feedback.py",
            ],
        },
        {
            "issue": "judge_reliability_size_and_family_bias",
            "status": "implemented_pending_live_and_human_evidence",
            "evidence": [
                "configs/experiment_v3_api_full.json: three provider-diverse judges",
                "analysis/judge_family_sensitivity.py",
                "cross-family primary aggregation in bench_runner.py",
            ],
        },
        {
            "issue": "single_prompt_frame_dependence_and_critical_frame_ambiguity",
            "status": "implemented_pending_live_evidence",
            "evidence": [
                "configs/context_variants.json: 17 conditions",
                "analysis/paraphrase_robustness.py",
                "analysis/frame_effect_stats.py: within-family wording aggregation",
            ],
        },
        {
            "issue": "larger_open_and_deployed_models",
            "status": "matrices_implemented_pending_live_evidence",
            "evidence": [
                "configs/experiment_v3_local_full.json: 13 open models",
                "configs/experiment_v3_api_full.json: 9 API models",
                "scripts/preflight_api_models.py",
            ],
        },
        {
            "issue": "practical_role_motivation_and_adjacent_benchmark_positioning",
            "status": "documented",
            "evidence": ["README.md", "docs/rubric.md"],
        },
        {
            "issue": "construction_to_evaluation_running_example",
            "status": "pipeline_documented_pending_live_response_example",
            "evidence": ["docs/dataset.md: Running example"],
        },
        {
            "issue": "ambiguous_overall_safety_composite",
            "status": "resolved",
            "evidence": [
                "docs/rubric.md: seven coequal outcomes",
                "judge_rubric.py",
                "result_reporting.py",
            ],
        },
        {
            "issue": "prompt_time_consistency_corrections",
            "status": "resolved_as_post_hoc_diagnostics",
            "evidence": ["analysis/rubric_consistency_audit.py"],
        },
        {
            "issue": "variability_confidence_intervals_and_result_figures",
            "status": "implemented_pending_live_evidence",
            "evidence": [
                "analysis/export_results.py",
                "analysis/frame_effect_stats.py",
                "analysis/make_figures.py",
            ],
        },
    ]


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
        "public_tree_hygiene": _public_tree_errors(),
    }

    required_analysis_tools = (
        "analysis/frame_effect_stats.py",
        "analysis/paraphrase_robustness.py",
        "analysis/judge_family_sensitivity.py",
        "analysis/rubric_consistency_audit.py",
        "analysis/api_usage_report.py",
        "analysis/export_results.py",
        "analysis/make_figures.py",
    )
    checks["analysis_implementations"] = [
        f"missing public analysis tool: {path}"
        for path in required_analysis_tools
        if not (ROOT / path).is_file()
    ]
    checks["coequal_statistics_runtime"] = _analysis_runtime_contract_errors()

    try:
        scenario_review_count = len(
            build_scenario_review_items(load_scenario_review_artifacts())
        )
        scenario_review_errors = (
            []
            if scenario_review_count == 51
            else ["V3 independent scenario-review packet must contain 51 motifs"]
        )
    except Exception as error:
        scenario_review_errors = [f"V3 scenario-review workflow failed: {error}"]
    checks["independent_scenario_review_workflow"] = scenario_review_errors

    required_annotation_tools = (
        "analysis/export_rubric_feedback.py",
        "analysis/export_calibration.py",
        "analysis/freeze_response_pool.py",
        "analysis/export_annotations.py",
        "analysis/assign_annotations.py",
        "analysis/import_annotations.py",
        "analysis/scenario_review.py",
        "analysis/freeze_human_annotation_plan.py",
        "analysis/freeze_analysis_plan_v3.py",
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
        "reviewer_revision_coverage": _reviewer_revision_coverage(),
        "unresolved_code_or_design_issues": []
        if not contract_errors
        else contract_errors,
        "pending_live_evidence": [
            "Generate the open-source target response matrix and local diagnostic judges.",
            "Preflight current API model IDs, then run the API pilot and full target matrix.",
            "Run the final cross-provider judge panel on the frozen response pool.",
            "Produce result-dependent frame effects, paraphrase robustness, confidence intervals, judge sensitivity, usage, tables, and figures.",
            "Select a blinded running transcript and update result-dependent manuscript tables, figures, and claims without reviving an overall-safety composite.",
        ],
        "pending_human_evidence": [
            "Complete independent V3 scenario QA for the construction-validity evidence and final analysis freeze.",
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
