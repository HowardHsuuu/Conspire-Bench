import asyncio
import unittest

from bench_runner import ConspireBenchmarkRunner
from benchmark_types import JudgeMetrics, ModelText
from scripts.run_judge_calibration import incomplete_result_reason


def make_runner_without_init() -> ConspireBenchmarkRunner:
    runner = ConspireBenchmarkRunner.__new__(ConspireBenchmarkRunner)
    runner.config = {}
    return runner


class JudgeRecoveryTests(unittest.TestCase):
    def test_empty_timeout_message_is_normalized(self):
        runner = make_runner_without_init()
        runner.config = {"evaluation": {"max_retries": 1, "timeout": 12}}

        async def fail():
            raise TimeoutError()

        with self.assertRaisesRegex(
            TimeoutError, "judge test timed out after 12 seconds"
        ):
            asyncio.run(
                runner._with_retries(
                    fail,
                    "judge test",
                    timeout_seconds=12,
                )
            )

    def test_formal_open_compatible_judge_gets_full_composite_timeout(self):
        runner = make_runner_without_init()
        runner.config = {"evaluation": {"timeout": 300}}
        captured = []

        async def response(*args, **kwargs):
            return ModelText('{"valid": true}')

        async def capture_retries(operation, operation_name, *, timeout_seconds):
            captured.append(timeout_seconds)
            return await operation()

        runner._get_model_response = response
        runner._parse_judge_response = lambda *args, **kwargs: JudgeMetrics(
            belief_amplification=1
        )
        runner._with_retries = capture_retries
        result = asyncio.run(
            runner._evaluate_with_judge_config(
                {"id": "scenario-1", "category": "test", "type": "single_turn"},
                [{"role": "assistant", "content": "cached"}],
                {
                    "provider": "huggingface",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "inference_backend": "openai_compatible",
                    "judge_prompt_variant": "v3_behavioral_anchors_evidence_first",
                },
            )
        )

        self.assertIsNone(result["error"])
        self.assertEqual(captured, [1800.0])

    def test_empty_error_is_not_reported_as_success_and_tail_is_saved(self):
        runner = make_runner_without_init()
        statuses = []
        checkpoints = []
        runner._scenario_by_id = lambda scenario_id: {"id": scenario_id}
        runner._write_status_row = lambda *args, **kwargs: statuses.append(
            (args, kwargs)
        )
        runner._save_results = lambda results, filename: checkpoints.append(
            (results, filename)
        )

        async def fail_with_legacy_empty_error(*args, **kwargs):
            return {
                "judge_name": "judge-a",
                "judge_run_id": runner._judge_run_id(
                    {"provider": "huggingface", "model": "judge-a"}
                ),
                "scores": {},
                "reasoning": {},
                "error": "",
            }

        runner._evaluate_with_judge_config = fail_with_legacy_empty_error
        rows = [
            {
                "scenario_id": "scenario-1",
                "model_name": "target-a",
                "conversation_log": [{"role": "assistant", "content": "cached"}],
                "judge_results": [],
            }
        ]

        asyncio.run(
            runner._run_judge_for_results(
                all_results=rows,
                judge_config={"provider": "huggingface", "model": "judge-a"},
                parallel_judgements=1,
                save_intermediate=True,
                save_intermediate_every=500,
                output_file="stage.json",
                status_file="stage.tsv",
            )
        )

        self.assertEqual(statuses[0][0][4], "judge_error")
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0][1], "temp_stage.json")

    def test_calibration_failure_reason_is_precise(self):
        expected = {
            "judge_run_id": "judge-current",
            "judge_prompt_variant": "v3",
            "scores": {"epistemic_grounding": 4},
            "error": None,
        }
        self.assertIsNone(
            incomplete_result_reason(
                expected,
                judge_run_id="judge-current",
                prompt_variant="v3",
            )
        )
        self.assertEqual(
            incomplete_result_reason(
                {**expected, "error": "", "scores": {}},
                judge_run_id="judge-current",
                prompt_variant="v3",
            ),
            "judge error (empty message)",
        )
        self.assertEqual(
            incomplete_result_reason(
                {**expected, "scores": {}},
                judge_run_id="judge-current",
                prompt_variant="v3",
            ),
            "missing scores",
        )
        self.assertEqual(
            incomplete_result_reason(
                {**expected, "judge_run_id": "judge-old"},
                judge_run_id="judge-current",
                prompt_variant="v3",
            ),
            "judge_run_id mismatch",
        )


if __name__ == "__main__":
    unittest.main()
