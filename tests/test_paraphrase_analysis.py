"""Unit tests for the formal paraphrase robustness analysis."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.analyze_full_matrix import FAMILIES, Inference, aggregate
from analysis.analyze_paraphrase_study import (
    METRICS,
    judge_sensitivity,
    strict_flip_summary,
    summarize_scalar,
)


class ParaphraseAnalysisTests(unittest.TestCase):
    def test_strict_flip_requires_both_polarities(self) -> None:
        deltas = np.array(
            [
                [[[1.0, 2.0, 0.0, 1.0]]],
                [[[-1.0, 1.0, 0.0, 1.0]]],
                [[[np.nan, 1.0, -1.0, 1.0]]],
            ]
        )
        report = strict_flip_summary(deltas)
        self.assertEqual(report["complete_cells"], 2)
        self.assertEqual(report["strict_polarity_flip_count"], 1)
        self.assertEqual(report["strict_polarity_flip_rate"], 0.5)

    def test_scalar_summary_clusters_on_first_axis(self) -> None:
        differences = np.array(
            [
                [[1.0, 3.0]],
                [[-1.0, 1.0]],
            ]
        )
        report = summarize_scalar(differences, Inference(2, 100, 7))
        self.assertEqual(report["matched_observations"], 4)
        self.assertEqual(report["motif_clusters"], 2)
        self.assertEqual(report["mean_difference"], 1.0)
        self.assertEqual(report["positive_motifs"], 1)
        self.assertEqual(report["tie_motifs"], 1)

    def test_judge_sensitivity_preserves_fully_missing_redirection(self) -> None:
        scores = np.ones((2, 4, 1, 17, 4, len(METRICS)), dtype=float)
        scores[..., METRICS.index("harmful_elaboration")] = 0
        scores[..., METRICS.index("appropriate_redirection")] = np.nan
        primary = aggregate(scores, list(FAMILIES))
        report = judge_sensitivity(
            scores,
            primary,
            list(FAMILIES),
            Inference(2, 20, 7),
        )
        redirection = [
            row
            for row in report["same_family_affinity"]["adjusted"]
            if row["metric"] == "appropriate_redirection"
        ]
        self.assertEqual(len(redirection), 4)
        self.assertTrue(all(row["favorable_shift"] is None for row in redirection))


if __name__ == "__main__":
    unittest.main()
