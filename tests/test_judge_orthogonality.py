"""Tests for the matched grounding/uncertainty calibration suite."""

import json
import unittest
from pathlib import Path

from analysis.analyze_judge_orthogonality import summarize
from analysis.prepare_judge_orthogonality_suite import CELLS, SUITE_VERSION, build_suite


class JudgeOrthogonalityTests(unittest.TestCase):
    def test_suite_is_balanced_and_matched(self):
        dataset = json.loads(Path("Conspire-Bench-v3.json").read_text())
        suite = build_suite(dataset, base_count=8, seed=20260922)
        rows = suite["detailed_results"]

        self.assertEqual(len(rows), 32)
        self.assertEqual(suite["metadata"]["suite_version"], SUITE_VERSION)
        self.assertEqual(len({row["response_id"] for row in rows}), 32)
        groups = {}
        for row in rows:
            groups.setdefault(row["scenario_id"], []).append(row)
        self.assertEqual(len(groups), 8)
        for group in groups.values():
            self.assertEqual(
                {row["context_label"] for row in group},
                {cell[0] for cell in CELLS},
            )
            self.assertEqual(
                len({row["conversation_log"][0]["content"] for row in group}), 1
            )
            high_low = next(
                row
                for row in group
                if row["context_label"] == "grounding_high_uncertainty_low"
            )["conversation_log"][1]["content"]
            low_high = next(
                row
                for row in group
                if row["context_label"] == "grounding_low_uncertainty_high"
            )["conversation_log"][1]["content"]
            self.assertIn("No further assessment is provided", high_low)
            self.assertNotIn("source provenance", low_high)
            self.assertNotIn("independent corroboration", low_high)

    def test_analysis_recovers_selective_effects(self):
        rows = []
        for base in ("s1", "s2"):
            for grounding in ("high", "low"):
                for uncertainty in ("high", "low"):
                    rows.append(
                        {
                            "response_id": f"{base}-{grounding}-{uncertainty}",
                            "scenario_metadata": {
                                "orthogonality_cell": {
                                    "grounding": grounding,
                                    "uncertainty": uncertainty,
                                }
                            },
                            "calibration_sampling": {"base_scenario_id": base},
                            "judge_results": [
                                {
                                    "judge_model_family": "test",
                                    "error": None,
                                    "scores": {
                                        "epistemic_grounding": (
                                            5 if grounding == "high" else 1
                                        ),
                                        "uncertainty_calibration": (
                                            5 if uncertainty == "high" else 1
                                        ),
                                    },
                                }
                            ],
                        }
                    )

        report = summarize(rows)

        self.assertTrue(report["all_judges_pass"])
        judge = report["judges"][0]
        self.assertEqual(judge["grounding_manipulation"]["mean_intended_effect"], 4)
        self.assertEqual(judge["uncertainty_manipulation"]["mean_intended_effect"], 4)
        self.assertEqual(
            judge["grounding_manipulation"]["mean_cross_effect_from_uncertainty"],
            0,
        )
        self.assertEqual(
            judge["uncertainty_manipulation"]["mean_cross_effect_from_grounding"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
