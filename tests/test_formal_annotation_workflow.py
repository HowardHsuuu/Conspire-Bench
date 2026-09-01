import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from analysis.export_annotations import export_annotation_package
from analysis.assign_annotations import assign_package
from analysis.freeze_annotator_roster import freeze_roster
from analysis.freeze_human_annotation_plan import freeze_human_plan
from analysis.freeze_response_pool import freeze_manifest as freeze_response_pool
from analysis.human_annotation_plan import human_annotation_plan_digest
from analysis.import_annotations import load_private_key, validate_collection_manifests
from experiment_conditions import stable_digest


ROOT = Path(__file__).resolve().parents[1]


def synthetic_results(*, scenario_count: int = 24, model_count: int = 3) -> dict:
    frames = (
        ("neutral", "neutral_none"),
        ("brainstorming", "brainstorming_v1"),
        ("critical_review", "critical_review_v1"),
    )
    rows = []
    for scenario_index in range(scenario_count):
        scenario_id = f"scenario_{scenario_index:03d}"
        scenario_type = ("single_turn", "multi_turn", "resistance")[
            scenario_index % 3
        ]
        category = f"category_{scenario_index % 6}"
        for model_index in range(model_count):
            model = f"provider/model_{model_index}"
            for frame, variant in frames:
                response_id = f"resp_{scenario_index}_{model_index}_{variant}"
                rows.append({
                    "response_id": response_id,
                    "condition_id": f"cond_{scenario_index}_{model_index}_{variant}",
                    "scenario_id": scenario_id,
                    "scenario_type": scenario_type,
                    "category": category,
                    "model_name": model,
                    "frame_family": frame,
                    "prompt_variant_id": variant,
                    "context_label": variant,
                    "prompt_schema_version": "2.0",
                    "generation_config": {"temperature": 0.0},
                    "generation_interface": "responses",
                    "access_date": "2026-09-01",
                    "scenario_metadata": {
                        "fact_check_status": "source_reviewed",
                        "review_status": "ai_author_reviewed",
                    },
                    "conversation_log": [
                        {"role": "user", "content": f"Claim {scenario_index}"},
                        {
                            "role": "assistant",
                            "content": f"Response {response_id}",
                            "response_metadata": {
                                "requested_model": model,
                                "interface": "responses",
                            },
                        },
                    ],
                    "judge_scores": {
                        "belief_amplification": 1 + scenario_index % 5,
                    },
                    "judge_disagreement": {
                        "belief_amplification": (scenario_index + model_index) % 4,
                    },
                    "judge_results": [{"rubric_version": "2.0"}],
                    "error": None,
                })
    return {
        "metadata": {
            "total_scenarios": scenario_count,
            "models": [f"provider/model_{index}" for index in range(model_count)],
            "filters": {"contexts": [variant for _, variant in frames]},
        },
        "detailed_results": rows,
    }


class FormalAnnotationWorkflowTests(unittest.TestCase):
    def _artifacts(self, directory: Path, payload: dict) -> tuple[Path, Path, Path, str]:
        result_path = directory / "results.json"
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        response_freeze_path = directory / "response_freeze.json"
        response_freeze_path.write_text(
            json.dumps(freeze_response_pool(result_path)), encoding="utf-8"
        )

        plan = json.loads(
            (ROOT / "configs" / "human_annotation_plan_v2.json").read_text(
                encoding="utf-8"
            )
        )
        frozen_plan = freeze_human_plan(
            plan,
            approved_by=["pi_fixture"],
            expert_minutes_median=3.5,
            student_minutes_median=1.2,
            change_summary="No count changes in formal integration fixture.",
            frozen_at="2026-09-01T00:00:00Z",
        )
        plan_path = directory / "human_annotation_plan.frozen.json"
        plan_path.write_text(json.dumps(frozen_plan), encoding="utf-8")

        excluded = [payload["detailed_results"][0]["response_id"]]
        exclusion_path = directory / "calibration_exclusion.json"
        exclusion_path.write_text(json.dumps({
            "schema_version": "1.0",
            "status": "frozen_calibration_exclusion",
            "must_exclude_from_formal_annotation": True,
            "response_ids": excluded,
            "response_ids_digest": stable_digest(excluded, length=64),
        }), encoding="utf-8")
        return (
            result_path,
            response_freeze_path,
            exclusion_path,
            human_annotation_plan_digest(frozen_plan),
        )

    def test_formal_export_exactly_matches_frozen_workload_and_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            directory = Path(tmp)
            result_path, response_freeze, exclusion, plan_digest = self._artifacts(
                directory, synthetic_results()
            )
            plan_path = directory / "human_annotation_plan.frozen.json"
            output_dir = directory / "formal"
            export_annotation_package(
                result_path,
                output_dir,
                representative_count=54,
                disagreement_count=18,
                pair_count=108,
                seed=24,
                release_mode="formal",
                freeze_manifest_path=response_freeze,
                calibration_exclusion_manifest_path=exclusion,
                annotation_plan_path=plan_path,
            )

            manifest = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(manifest["expert_item_count"], 72)
            self.assertEqual(
                manifest["expert_sample_roles"],
                {"representative": 54, "judge_disagreement_enriched": 18},
            )
            self.assertEqual(manifest["student_pair_item_count"], 108)
            self.assertEqual(manifest["annotation_plan_digest"], plan_digest)

            keys = [
                json.loads(line)
                for line in (output_dir / "annotation_key.private.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(keys), 180)
            self.assertEqual({row["annotation_plan_digest"] for row in keys}, {plan_digest})
            expert_roles = Counter(
                row["sample_role"]
                for row in keys
                if row["item_type"] == "expert_conversation"
            )
            self.assertEqual(
                expert_roles,
                {"representative": 54, "judge_disagreement_enriched": 18},
            )

            roster_csv = directory / "roster.csv"
            header = (
                "annotator_id,role,rubric_version,expertise_verified,"
                "content_validity_complete,calibration_complete,training_complete,"
                "qualification_score,qualification_max,qualification_pass_threshold,"
                "exclusion_reason,eligibility_note\n"
            )
            expert_rows = [
                f"expert_{index:02d},expert,2.0,true,true,true,false,,,,,verified\n"
                for index in range(1, 4)
            ]
            student_rows = [
                f"student_{index:02d},student,2.0,false,false,false,true,9,10,0.8,,passed\n"
                for index in range(1, 10)
            ]
            roster_csv.write_text(
                header + "".join([*expert_rows, *student_rows]), encoding="utf-8"
            )
            roster_manifest_path = directory / "roster_manifest.json"
            freeze_roster(roster_csv, roster_manifest_path)
            assignment_dir = directory / "assigned"
            assign_package(
                expert_path=output_dir / "expert_items.jsonl",
                student_path=output_dir / "student_pair_items.jsonl",
                output_dir=assignment_dir,
                expert_ids=[f"expert_{index:02d}" for index in range(1, 4)],
                student_ids=[f"student_{index:02d}" for index in range(1, 10)],
                expert_ratings_per_item=2,
                student_ratings_per_item=3,
                seed=20260831,
                release_mode="formal",
                roster_manifest_path=roster_manifest_path,
                annotation_plan_path=plan_path,
            )
            assignment_manifest_path = (
                assignment_dir / "assignment_manifest.private.json"
            )
            assignment_manifest = json.loads(assignment_manifest_path.read_text())
            self.assertEqual(
                assignment_manifest["assignment_counts_by_type"],
                {"expert_conversation": 144, "paired_conversation": 324},
            )
            self.assertEqual(
                {
                    assignment_manifest["assignment_counts_by_annotator"][
                        f"expert_{index:02d}"
                    ]
                    for index in range(1, 4)
                },
                {48},
            )
            self.assertEqual(
                {
                    assignment_manifest["assignment_counts_by_annotator"][
                        f"student_{index:02d}"
                    ]
                    for index in range(1, 10)
                },
                {36},
            )
            validated_assignment = validate_collection_manifests(
                load_private_key(output_dir / "annotation_key.private.jsonl"),
                assignment_manifest_path,
                roster_manifest_path,
            )
            self.assertEqual(
                validated_assignment["annotation_plan_digest"], plan_digest
            )

            with self.assertRaisesRegex(ValueError, "do not match the frozen"):
                export_annotation_package(
                    result_path,
                    directory / "drifted",
                    representative_count=53,
                    disagreement_count=18,
                    pair_count=108,
                    seed=24,
                    release_mode="formal",
                    freeze_manifest_path=response_freeze,
                    calibration_exclusion_manifest_path=exclusion,
                    annotation_plan_path=plan_path,
                )

    def test_formal_export_rejects_pool_that_cannot_supply_all_pairs(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            directory = Path(tmp)
            result_path, response_freeze, exclusion, _ = self._artifacts(
                directory, synthetic_results(scenario_count=10)
            )
            with self.assertRaisesRegex(ValueError, "complete prespecified student pair"):
                export_annotation_package(
                    result_path,
                    directory / "formal",
                    representative_count=54,
                    disagreement_count=18,
                    pair_count=108,
                    seed=24,
                    release_mode="formal",
                    freeze_manifest_path=response_freeze,
                    calibration_exclusion_manifest_path=exclusion,
                    annotation_plan_path=(
                        directory / "human_annotation_plan.frozen.json"
                    ),
                )


if __name__ == "__main__":
    unittest.main()
