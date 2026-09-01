from __future__ import annotations

import copy
import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.frame_effect_stats import (
    FRAME_DESIGNS,
    build_v3_coequal_report,
)
from analysis.frame_effect_stats import (
    build_report as build_frame_report,
)
from analysis.make_figures import effect_rows, render_frame_effect_figure
from experiment_conditions import load_context_set
from scripts.audit_release_readiness import build_report as build_readiness_report
from scripts.build_interaction_catalog_v3 import build_catalog
from scripts.render_motif_circulation_audit_v3 import (
    render_document as render_circulation_audit,
)
from scripts.render_motif_pool_coverage_v3 import (
    render_document as render_pool_coverage,
)
from scripts.validate_context_variants_v3 import validate_context_variants
from scripts.validate_experiment_configs_v3 import validate as validate_experiments
from scripts.validate_interaction_catalog_v3 import validate as validate_interactions
from scripts.validate_motif_narratives_v3 import validate_records
from scripts.validate_motif_quality_review_v3 import validate_review
from scripts.validate_motif_selection_recommendation_v3 import (
    validate_recommendation,
)
from scripts.validate_primary_motif_manifest import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MARKDOWN = {
    "README.md",
    "annotation/rubric_validity_v2/README.md",
    "annotation_ui/README.md",
    "docs/annotation_protocol.md",
    "docs/dataset.md",
    "docs/evaluation.md",
    "docs/local_models.md",
    "docs/rubric.md",
    "docs/runpod_5090_setup.md",
}


class V3MotifPipelineTest(unittest.TestCase):
    def test_public_markdown_is_bounded_and_local_links_resolve(self) -> None:
        actual = {
            str(path.relative_to(ROOT))
            for path in ROOT.rglob("*.md")
            if ".local" not in path.parts and "legacy" not in path.parts
        }
        self.assertEqual(actual, PUBLIC_MARKDOWN)

        missing_links: list[str] = []
        for relative in sorted(PUBLIC_MARKDOWN):
            document = ROOT / relative
            for target in re.findall(
                r"\[[^\]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8")
            ):
                if "://" in target or target.startswith(("#", "mailto:")):
                    continue
                clean_target = target.split("#", 1)[0].strip("<>")
                if clean_target and not (document.parent / clean_target).exists():
                    missing_links.append(f"{relative} -> {target}")
        self.assertEqual(missing_links, [])

    def test_v3_experiment_matrices_preserve_scale_and_provider_coverage(self) -> None:
        local_config = self._load("configs/experiment_v3_local_full.json")
        api_config = self._load("configs/experiment_v3_api_full.json")
        self.assertEqual(validate_experiments(local_config, api_config), [])

        invalid = copy.deepcopy(local_config)
        invalid["models"][0].pop("parameter_scale_b")
        errors = validate_experiments(invalid, api_config)
        self.assertTrue(any("parameter_scale_b" in error for error in errors))

    def test_release_readiness_contract_is_code_complete(self) -> None:
        report = build_readiness_report()

        self.assertTrue(report["code_ready"], report["contract_errors"])
        self.assertEqual(
            report["release_state"], "code_ready_pending_external_evidence"
        )
        self.assertTrue(report["pending_live_evidence"])
        self.assertTrue(report["pending_human_evidence"])
        self.assertEqual(report["unresolved_code_or_design_issues"], [])
        coverage = report["reviewer_revision_coverage"]
        self.assertEqual(len(coverage), 11)
        self.assertTrue(
            all(item["status"] not in {"missing", "unimplemented"} for item in coverage)
        )

    def test_builder_uses_only_manifest_primary_motifs(self) -> None:
        narratives = {
            "records": [
                self._record("selected"),
                self._record("eligible_but_not_selected"),
            ]
        }
        manifest = {
            "selection_state": "frozen_primary",
            "motifs": [
                {
                    "motif_id": "selected",
                    "display_name": "Selected motif",
                    "category": "test",
                }
            ],
            "additional_eligible_candidates": [
                {
                    "motif_id": "eligible_but_not_selected",
                    "display_name": "Other motif",
                    "category": "test",
                }
            ],
        }

        catalog = build_catalog(narratives, manifest)

        self.assertEqual(catalog["motif_count"], 1)
        self.assertEqual(catalog["scenario_count"], 3)
        self.assertEqual([item["motif_id"] for item in catalog["motifs"]], ["selected"])

    def test_builder_reports_selected_motif_without_narrative(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "selected motifs without narrative records"
        ):
            build_catalog(
                {"records": []},
                {
                    "selection_state": "frozen_primary",
                    "motifs": [
                        {
                            "motif_id": "missing",
                            "display_name": "Missing",
                            "category": "test",
                        }
                    ],
                },
            )

    def test_real_recommendation_partitions_the_candidate_pool(self) -> None:
        recommendation = self._load("configs/motif_selection_recommendation_v3.json")
        manifest = self._load("configs/primary_motif_manifest_v3.json")
        quality = self._load("configs/motif_quality_review_v3.json")

        self.assertEqual(validate_recommendation(recommendation, manifest, quality), [])

    def test_real_complete_story_quality_layers_validate(self) -> None:
        manifest = self._load("configs/primary_motif_manifest_v3.json")
        narratives = self._load("configs/motif_narratives_v3.json")
        quality = self._load("configs/motif_quality_review_v3.json")
        current_catalog = self._load("configs/scenario_expansion_v2.json")

        self.assertEqual(validate_manifest(manifest, current_catalog), [])
        self.assertEqual(validate_records(narratives, manifest), [])
        self.assertEqual(validate_review(quality, manifest, narratives), [])

    def test_manifest_anchor_cannot_be_a_merely_close_source(self) -> None:
        manifest = self._load("configs/primary_motif_manifest_v3.json")
        narratives = self._load("configs/motif_narratives_v3.json")
        mutated = copy.deepcopy(manifest)
        haarp = next(
            motif
            for motif in mutated["motifs"]
            if motif["motif_id"] == "haarp_earthquake"
        )
        haarp["evidence_date"] = "2025-05-30"
        haarp["evidence_url"] = "https://fullfact.org/world/haarp-earthquake-myanmar/"

        errors = validate_records(narratives, mutated)

        self.assertTrue(
            any(
                "haarp_earthquake: manifest evidence must be" in error
                for error in errors
            ),
            errors,
        )

    def test_quality_review_names_an_eligible_complete_story_anchor(self) -> None:
        manifest = self._load("configs/primary_motif_manifest_v3.json")
        narratives = self._load("configs/motif_narratives_v3.json")
        quality = self._load("configs/motif_quality_review_v3.json")
        mutated = copy.deepcopy(quality)
        digital_id = next(
            record
            for record in mutated["records"]
            if record["motif_id"] == "digital_id_total_control"
        )
        digital_id["complete_story_anchor_url"] = (
            "https://www.aap.com.au/factcheck/"
            "jordan-peterson-gets-it-wrong-on-chinas-social-credit-system/"
        )

        errors = validate_review(mutated, manifest, narratives)

        self.assertTrue(
            any(
                "digital_id_total_control.complete_story_anchor_url must identify"
                in error
                for error in errors
            ),
            errors,
        )

    def test_circulation_audit_renderer_uses_current_records(self) -> None:
        rendered = render_circulation_audit(
            self._load("configs/primary_motif_manifest_v3.json"),
            self._load("configs/motif_narratives_v3.json"),
            self._load("configs/motif_quality_review_v3.json"),
            self._load("configs/motif_selection_recommendation_v3.json"),
        )

        self.assertIn("Candidate pool: 51 complete", rendered)
        self.assertIn("Selection state: `frozen_primary`", rendered)
        self.assertIn("All 51 eligible, nonduplicate motifs", rendered)

    def test_pool_coverage_renderer_uses_current_records(self) -> None:
        rendered = render_pool_coverage(
            self._load("configs/primary_motif_manifest_v3.json"),
            self._load("configs/motif_quality_review_v3.json"),
            self._load("configs/motif_selection_recommendation_v3.json"),
            self._load("configs/context_variants.json"),
        )

        self.assertIn("51 eligible complete narratives", rendered)
        self.assertIn("| Frozen main | 51 | 153 | 765 | 2601 |", rendered)

    def test_v3_frames_are_five_families_with_nested_paraphrases(self) -> None:
        config_path = ROOT / "configs/context_variants.json"
        payload = self._load("configs/context_variants.json")

        self.assertEqual(validate_context_variants(payload), [])
        main = load_context_set("main_v3", config_path)
        full = load_context_set("full_v3", config_path)
        robustness = load_context_set("reviewer_robustness_v3", config_path)
        self.assertEqual(
            [condition.frame for condition in main],
            [
                "neutral",
                "brainstorming",
                "critical_review",
                "sensemaking",
                "supportive_listening",
            ],
        )
        self.assertEqual(len(robustness), 17)
        self.assertEqual(full, robustness)

    def test_v3_frame_statistics_include_four_neutral_contrasts(self) -> None:
        rows = []
        for frame in ("neutral", *FRAME_DESIGNS["v3"]):
            rows.append(
                {
                    "model_name": "provider/model",
                    "scenario_id": "scenario_1",
                    "scenario_type": "single_turn_complete_logic",
                    "frame_family": frame,
                    "prompt_variant_id": (
                        "neutral_none" if frame == "neutral" else f"{frame}_v1"
                    ),
                    "generation_seed": 7,
                    "replicate_id": 0,
                    "judge_scores": {
                        "belief_amplification": 1,
                        "harmful_elaboration": False,
                    },
                    "error": None,
                }
            )

        report = build_frame_report(rows, draws=20, seed=1, frames=FRAME_DESIGNS["v3"])

        self.assertEqual(
            [contrast["frame"] for contrast in report["contrasts"]],
            list(FRAME_DESIGNS["v3"]),
        )
        self.assertTrue(
            all(
                contrast["effect"]["matched_dyads"] == 1
                for contrast in report["contrasts"]
            )
        )

    def test_v3_coequal_report_applies_one_28_test_fdr_family(self) -> None:
        rows = []
        for frame in ("neutral", *FRAME_DESIGNS["v3"]):
            rows.append(
                {
                    "model_name": "provider/model",
                    "scenario_id": "scenario_1",
                    "scenario_type": "single_turn_complete_logic",
                    "frame_family": frame,
                    "prompt_variant_id": (
                        "neutral_none" if frame == "neutral" else f"{frame}_v1"
                    ),
                    "generation_seed": 7,
                    "replicate_id": 0,
                    "judge_scores": {
                        "belief_amplification": 1,
                        "epistemic_grounding": 4,
                        "uncertainty_calibration": 4,
                        "harmful_elaboration": False,
                        "supportive_engagement": 4,
                        "appropriate_redirection": 2,
                        "response_appropriateness": 0,
                    },
                    "error": None,
                }
            )

        report = build_v3_coequal_report(rows, draws=20, seed=1)

        self.assertEqual(report["estimand_count"], 28)
        contrasts = [
            contrast
            for outcome_report in report["outcome_reports"]
            for contrast in outcome_report["contrasts"]
        ]
        self.assertTrue(
            all(
                "fdr_bh_adjusted_p_value" in contrast["motif_level_sign_test"]
                for contrast in contrasts
            )
        )
        by_outcome = {
            outcome_report["metric"]: outcome_report
            for outcome_report in report["outcome_reports"]
        }
        self.assertIn(
            "appropriate_redirection_applicability",
            by_outcome["appropriate_redirection"]["contrasts"][0],
        )
        self.assertIn(
            "subtype_frequencies",
            by_outcome["harmful_elaboration"]["contrasts"][0]["harmful_elaboration"],
        )
        sensitivity_contrasts = [
            contrast
            for outcome_report in report["overlap_cluster_sensitivity"][
                "outcome_reports"
            ]
            for contrast in outcome_report["contrasts"]
        ]
        self.assertEqual(len(sensitivity_contrasts), 28)
        self.assertEqual(len(effect_rows(report)), 28)
        with TemporaryDirectory() as temporary:
            outputs = render_frame_effect_figure(report, Path(temporary))
            self.assertEqual({path.suffix for path in outputs}, {".pdf", ".png"})
            self.assertTrue(all(path.stat().st_size > 0 for path in outputs))

    def test_full_wording_analysis_averages_variants_within_frame(self) -> None:
        rows = [
            {
                "model_name": "provider/model",
                "scenario_id": "scenario_1",
                "scenario_type": "single_turn_complete_logic",
                "frame_family": "neutral",
                "prompt_variant_id": "neutral_none",
                "generation_seed": 7,
                "replicate_id": 0,
                "judge_scores": {
                    "belief_amplification": 1,
                    "harmful_elaboration": False,
                },
                "error": None,
            }
        ]
        for index, score in enumerate((1, 2, 3, 4), start=1):
            rows.append(
                {
                    "model_name": "provider/model",
                    "scenario_id": "scenario_1",
                    "scenario_type": "single_turn_complete_logic",
                    "frame_family": "brainstorming",
                    "prompt_variant_id": f"brainstorming_v{index}",
                    "generation_seed": 7,
                    "replicate_id": 0,
                    "judge_scores": {
                        "belief_amplification": score,
                        "harmful_elaboration": False,
                    },
                    "error": None,
                }
            )

        report = build_frame_report(
            rows,
            draws=20,
            seed=1,
            canonical_only=False,
            frames=("brainstorming",),
        )

        self.assertEqual(
            report["contrasts"][0]["effect"]["mean_paired_difference"], 1.5
        )
        self.assertEqual(
            report["wording_aggregation"],
            "equal_mean_of_four_variants_nested_within_frame_family",
        )

    def test_real_v3_interaction_catalog_has_all_matched_structures(self) -> None:
        errors = validate_interactions(
            self._load("configs/interaction_catalog_v3.json"),
            self._load("Conspire-Bench-v3.json"),
            self._load("configs/primary_motif_manifest_v3.json"),
            self._load("configs/motif_quality_review_v3.json"),
            self._load("configs/motif_narratives_v3.json"),
            self._load("configs/interaction_identity_policy_v3.json"),
        )
        self.assertEqual(errors, [])

    @staticmethod
    def _record(motif_id: str) -> dict:
        return {
            "motif_id": motif_id,
            "narrative_summary": "A complete hidden-coordination account circulated.",
            "alleged_actor": "hidden actors",
            "alleged_action": "coordinate an action",
            "alleged_goal": "achieve a concealed goal",
            "alleged_mechanism": "a hidden mechanism",
            "alleged_concealment": "public explanations conceal coordination",
            "authoring_scope": "Use fictional details.",
        }

    @staticmethod
    def _load(relative_path: str) -> dict:
        return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
