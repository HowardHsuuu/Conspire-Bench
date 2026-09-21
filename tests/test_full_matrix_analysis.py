"""Numerical checks for the motif-clustered post-run analysis."""

import unittest

import numpy as np

from analysis.analyze_full_matrix import (
    FAMILIES,
    HARM,
    METRICS,
    REDIR,
    Inference,
    aggregate,
    family_affinity,
    kappa_from_confusion,
    model_scaling,
)


class FullMatrixAnalysisTests(unittest.TestCase):
    def test_other_family_or_and_missing_redirection(self):
        x = np.ones((2, 1, 1, 1, 4, 7))
        x[..., REDIR] = np.nan
        x[..., HARM] = 0
        x[..., 0, HARM] = 1
        a = aggregate(x, ["qwen"])
        self.assertTrue((a[..., HARM] == 0).all())
        self.assertTrue(np.isnan(a[..., REDIR]).all())
        x[..., 1, HARM] = 1
        self.assertTrue((aggregate(x, ["qwen"])[..., HARM] == 1).all())
        self.assertTrue(
            np.allclose(aggregate(x, ["qwen"], harm_rule="mean")[..., HARM], 1 / 3)
        )

    def test_motif_weighting_does_not_count_missing_as_zero(self):
        d = np.full((2, 3, 7), np.nan)
        d[0, :, 0] = 2
        d[1, 0, 0] = 0
        result = Inference(2, 1000, 7).summarize(d)
        self.assertEqual(result[0]["mean_difference"], 1)
        self.assertEqual(result[0]["matched_observations"], 4)
        self.assertEqual(result[0]["motif_clusters"], 2)
        self.assertIsNone(result[REDIR]["mean_difference"])

    def test_linear_kappa_known_cases(self):
        levels = np.arange(2)
        self.assertAlmostEqual(float(kappa_from_confusion([[5, 0], [0, 5]], levels)), 1)
        self.assertAlmostEqual(float(kappa_from_confusion([[5, 5], [5, 5]], levels)), 0)
        self.assertAlmostEqual(
            float(kappa_from_confusion([[0, 5], [5, 0]], levels)), -1
        )
        self.assertTrue(np.isnan(kappa_from_confusion([[10, 0], [0, 0]], levels)))

    def test_adjusted_affinity_removes_additive_severity(self):
        x = np.zeros((6, 4, 1, 1, 4, len(METRICS)))
        expected = [0.7, -0.2, 0.4, 0.0]
        for target in range(4):
            for judge in range(4):
                x[:, target, :, :, judge, :] = 10 + target * 0.3 - judge * 1.2
                if target == judge:
                    x[:, target, :, :, judge, :] += expected[judge]
        report = family_affinity(x, FAMILIES, Inference(6, 100, 7))
        for r in report["adjusted"]:
            self.assertAlmostEqual(
                r["mean_difference"], expected[FAMILIES.index(r["judge"])], places=10
            )

    def test_scaling_preserves_motif_axis_with_model_subset(self):
        models = ["huggingface/Qwen/Qwen2.5-1B", "huggingface/Qwen/Qwen2.5-2B"]
        config = {
            "models": [
                {
                    "provider": "huggingface",
                    "model": m.split("/", 1)[1],
                    "parameter_scale_b": 2**i,
                }
                for i, m in enumerate(models)
            ]
        }
        x = np.zeros((7, 2, 3, 5, 7))
        x[:, 1] = 2
        x[:, 1, :, 1] += 1
        records = model_scaling(x, models, config, Inference(7, 100, 3))
        for r in records:
            expected = 1 if r["estimand"].startswith("brainstorm") else 2.2
            self.assertAlmostEqual(r["mean_difference"], expected)


if __name__ == "__main__":
    unittest.main()
