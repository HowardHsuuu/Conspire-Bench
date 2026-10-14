import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_vllm_gemini_cross_family_judges import (
    FORMAL_PROMPT_VARIANT,
    validate_final,
    validate_judges,
    validate_source,
)


def judge(family: str, model: str) -> dict:
    return {
        "provider": "huggingface",
        "model": model,
        "model_family": family,
        "judge_prompt_variant": FORMAL_PROMPT_VARIANT,
    }


JUDGES = [
    judge("qwen", "Qwen/Qwen2.5-32B-Instruct"),
    judge("llama", "meta-llama/Llama-3.1-8B-Instruct"),
    judge("gemma", "google/gemma-3-27b-it"),
    judge("gpt_oss", "openai/gpt-oss-120b"),
]


class GeminiCrossFamilyJudgesTest(unittest.TestCase):
    def make_rows(self) -> list[dict]:
        rows = []
        names = [f"{item['provider']}/{item['model']}" for item in JUDGES]
        for scenario_index in range(36):
            for variant_index in range(5):
                response_id = f"response-{scenario_index}-{variant_index}"
                cross = [
                    {
                        "judge_name": name,
                        "judge_prompt_variant": FORMAL_PROMPT_VARIANT,
                        "same_family_as_target": False,
                        "scores": {"belief_amplification": 1},
                        "error": None,
                    }
                    for name in names
                ]
                rows.append(
                    {
                        "response_id": response_id,
                        "scenario_id": f"scenario-{scenario_index}",
                        "scenario_metadata": {
                            "motif_id": f"motif-{scenario_index // 3}"
                        },
                        "prompt_variant_id": f"variant-{variant_index}",
                        "model_name": "gemini/gemini-3.8-flash",
                        "model_family": "gemini",
                        "generation_complete": True,
                        "conversation_log": [{"role": "assistant", "content": "ok"}],
                        "evaluation_complete": True,
                        "error": None,
                        "judge_scores": {"belief_amplification": 1},
                        "primary_judge_names": names,
                        "judge_results": [
                            {
                                "judge_name": "gemini-self",
                                "same_family_as_target": True,
                                "scores": {"belief_amplification": 1},
                                "error": None,
                            },
                            *cross,
                        ],
                    }
                )
        return rows

    def test_frozen_source_and_completed_panel_contract(self) -> None:
        rows = self.make_rows()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps({"detailed_results": rows}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source = validate_source(path, expected_sha256=digest)
            final = validate_final(path, validate_judges({"judges": JUDGES}))
        self.assertEqual(source["row_count"], 180)
        self.assertEqual(source["motif_count"], 12)
        self.assertEqual(final["successful_cross_family_judgments"], 720)
        self.assertEqual(final["retained_same_family_judgments"], 180)

    def test_source_hash_is_enforced(self) -> None:
        rows = self.make_rows()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps({"detailed_results": rows}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_source(path, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
