"""Tests for the resumable formal V3 judge-matrix launcher."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_vllm_judge_full_matrix as full


class JudgeFullMatrixTests(unittest.TestCase):
    def _bundle(self, path: Path, *, include_judges: bool = False) -> None:
        rows = []
        for index in range(2):
            results = []
            if include_judges:
                for family in ("qwen", "llama"):
                    results.append(
                        {
                            "judge_name": f"huggingface/{family}/judge",
                            "judge_prompt_variant": full.FORMAL_PROMPT_VARIANT,
                            "scores": {"epistemic_grounding": 3},
                            "error": None,
                        }
                    )
            rows.append(
                {
                    "response_id": f"response-{index}",
                    "model_name": f"model-{index}",
                    "scenario_metadata": {"motif_id": f"motif-{index}"},
                    "conversation_log": [{"role": "assistant", "content": "ok"}],
                    "generation_complete": True,
                    "judge_results": results,
                }
            )
        path.write_text(json.dumps({"detailed_results": rows}), encoding="utf-8")

    def test_validates_frozen_source_and_final_panel(self):
        judges = [
            {"provider": "huggingface", "model": "qwen/judge"},
            {"provider": "huggingface", "model": "llama/judge"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.json"
            final = Path(temporary) / "final.json"
            self._bundle(source)
            self._bundle(final, include_judges=True)
            with (
                patch.object(full, "EXPECTED_ROWS", 2),
                patch.object(full, "EXPECTED_MODELS", 2),
                patch.object(full, "EXPECTED_MOTIFS", 2),
                patch.object(full, "EXPECTED_JUDGES", 2),
            ):
                self.assertEqual(full.validate_frozen_source(source)["row_count"], 2)
                report = full.validate_final_panel(
                    final, judges, prompt_variant=full.FORMAL_PROMPT_VARIANT
                )
                self.assertEqual(report["successful_judgments"], 4)

            self.assertEqual(len(full.sha256_file(source)), 64)

    def test_rejects_missing_formal_judge(self):
        judges = [
            {"provider": "huggingface", "model": "qwen/judge"},
            {"provider": "huggingface", "model": "llama/judge"},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "final.json"
            self._bundle(final, include_judges=True)
            payload = json.loads(final.read_text(encoding="utf-8"))
            payload["detailed_results"][0]["judge_results"].pop()
            final.write_text(json.dumps(payload), encoding="utf-8")
            with (
                patch.object(full, "EXPECTED_ROWS", 2),
                patch.object(full, "EXPECTED_JUDGES", 2),
                self.assertRaisesRegex(RuntimeError, "Incomplete formal judge panel"),
            ):
                full.validate_final_panel(
                    final, judges, prompt_variant=full.FORMAL_PROMPT_VARIANT
                )


if __name__ == "__main__":
    unittest.main()
