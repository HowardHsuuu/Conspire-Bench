"""Validation checks for the frozen reviewer-sized paraphrase study."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from scripts.run_vllm_paraphrase_robustness import (
    FORMAL_PROMPT_VARIANT,
    ensure_idle_gpu,
    validate_design,
    validate_final_panel,
    validate_generation_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class ParaphraseRobustnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.design = load("configs/paraphrase_robustness_v1.json")
        self.config = load("configs/experiment_v3_paraphrase_robustness.json")
        self.dataset = load("Conspire-Bench-v3.json")
        self.registry = load("configs/context_variants.json")

    def test_frozen_design_is_complete_and_outcome_blind(self) -> None:
        plan = validate_design(self.design, self.config, self.dataset, self.registry)
        self.assertTrue(self.design["selection"]["outcome_blind"])
        self.assertEqual(self.design["selection"]["outcomes_used_for_selection"], [])
        self.assertEqual(plan["motifs"], 12)
        self.assertEqual(plan["scenarios"], 36)
        self.assertEqual(plan["context_variants"], 17)
        self.assertEqual(plan["responses"], 2448)
        self.assertEqual(plan["successful_judgments"], 9792)
        categories = {
            scenario["category"]
            for scenario in self.dataset["scenarios"]
            if scenario["id"] in plan["scenario_ids"]
        }
        self.assertEqual(len(categories), 12)

    def test_design_rejects_context_drift(self) -> None:
        changed = deepcopy(self.design)
        changed["contexts"]["variant_ids"][-1] = "supportive_listening_v3"
        with self.assertRaisesRegex(ValueError, "frozen context set"):
            validate_design(changed, self.config, self.dataset, self.registry)

    def test_generation_validator_requires_exact_cartesian_product(self) -> None:
        plan = validate_design(self.design, self.config, self.dataset, self.registry)
        rows = []
        for model in self.design["target_models"]:
            for scenario in plan["scenario_ids"]:
                for variant in plan["variant_ids"]:
                    identity = f"{model}|{scenario}|{variant}"
                    rows.append(
                        {
                            "response_id": f"response:{identity}",
                            "condition_id": f"condition:{identity}",
                            "model_name": model,
                            "scenario_id": scenario,
                            "prompt_variant_id": variant,
                            "generation_seed": 20261012,
                            "replicate_id": 0,
                            "generation_complete": True,
                            "error": None,
                            "conversation_log": [
                                {"role": "assistant", "content": "response"}
                            ],
                        }
                    )
        report = validate_generation_rows(
            rows,
            model_names=set(self.design["target_models"]),
            scenario_ids=set(plan["scenario_ids"]),
            variant_ids=set(plan["variant_ids"]),
            seed=20261012,
        )
        self.assertEqual(report["row_count"], 2448)
        rows[-1]["generation_seed"] = 0
        with self.assertRaisesRegex(ValueError, "generation seed mismatch"):
            validate_generation_rows(
                rows,
                model_names=set(self.design["target_models"]),
                scenario_ids=set(plan["scenario_ids"]),
                variant_ids=set(plan["variant_ids"]),
                seed=20261012,
            )

    def test_final_panel_requires_every_frozen_judge(self) -> None:
        results = [
            {
                "judge_name": name,
                "judge_prompt_variant": FORMAL_PROMPT_VARIANT,
                "error": None,
                "scores": {"belief_amplification": 1},
            }
            for name in self.design["judge_models"]
        ]
        report = validate_final_panel(
            [{"response_id": "r1", "judge_results": results}],
            self.config["judges"],
            prompt_variant=FORMAL_PROMPT_VARIANT,
        )
        self.assertEqual(report["successful_judgments"], 4)
        results.pop()
        with self.assertRaisesRegex(ValueError, "incomplete judge panel"):
            validate_final_panel(
                [{"response_id": "r1", "judge_results": results}],
                self.config["judges"],
                prompt_variant=FORMAL_PROMPT_VARIANT,
            )

    def test_gb10_na_process_placeholder_is_treated_as_idle(self) -> None:
        with mock.patch("scripts.run_vllm_paraphrase_robustness.subprocess.run") as run:
            run.return_value.stdout = "[N/A], [N/A], [N/A]\n"
            ensure_idle_gpu()


if __name__ == "__main__":
    unittest.main()
