"""Paired frame-effect statistics for Conspire-Bench results.

This script reproduces the paired bootstrap, sign-test, and McNemar summaries
reported in the paper appendix. It uses only the merged JSON results file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable


FRAMES = ("brainstorming", "critical_review")
STRUCTURES = (
    "single_turn_complete_logic",
    "multi_turn_progression",
    "complete_logic_then_resistance",
)


def exact_binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) / (2**n) for i in range(n + 1)]
    observed = probs[k]
    return min(1.0, sum(p for p in probs if p <= observed + 1e-15))


def bootstrap_ci(values: list[float], draws: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(draws):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def overall(row: dict) -> float:
    return float(row["judge_scores"]["overall_safety"])


def harm(row: dict) -> float:
    return float(bool(row["judge_scores"]["harmful_elaboration"]))


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data["detailed_results"]


def paired_differences(
    triples: dict[tuple[str, str], dict[str, dict]],
    frame: str,
    metric: Callable[[dict], float],
    structure: str | None = None,
) -> list[float]:
    values = []
    for frames in triples.values():
        if structure and frames["none"]["scenario_type"] != structure:
            continue
        values.append(metric(frames[frame]) - metric(frames["none"]))
    return values


def sign_test(values: list[float]) -> tuple[int, int, int, float]:
    pos = sum(v > 0 for v in values)
    neg = sum(v < 0 for v in values)
    ties = sum(v == 0 for v in values)
    return pos, neg, ties, exact_binom_two_sided(min(pos, neg), pos + neg)


def mcnemar(
    triples: dict[tuple[str, str], dict[str, dict]],
    frame: str,
    structure: str | None = None,
) -> tuple[int, int, float]:
    none_to_harm = 0
    harm_to_none = 0
    for frames in triples.values():
        if structure and frames["none"]["scenario_type"] != structure:
            continue
        neutral = bool(harm(frames["none"]))
        framed = bool(harm(frames[frame]))
        if not neutral and framed:
            none_to_harm += 1
        elif neutral and not framed:
            harm_to_none += 1
    return none_to_harm, harm_to_none, exact_binom_two_sided(
        min(none_to_harm, harm_to_none), none_to_harm + harm_to_none
    )


def summarize(values: list[float], draws: int, seed: int) -> tuple[float, float, float]:
    mean = sum(values) / len(values)
    low, high = bootstrap_ci(values, draws=draws, seed=seed)
    return mean, low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results",
        nargs="?",
        default="results/20260515_163319/merged_local_seed24_contexts_6models.json",
        type=Path,
    )
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    triples: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in load_rows(args.results):
        triples[(row["model_name"], row["scenario_id"])][row["context_label"]] = row

    print(f"matched triples: {len(triples)}")
    for frame in FRAMES:
        print(f"\n{frame} vs neutral")
        for name, metric in (("overall", overall), ("harm", harm)):
            values = paired_differences(triples, frame, metric)
            mean, low, high = summarize(values, args.draws, seed=7)
            print(f"  {name}: mean={mean:.3f} 95% CI=[{low:.3f}, {high:.3f}]")
            if name == "overall":
                pos, neg, ties, p = sign_test(values)
                print(f"    sign: pos={pos} neg={neg} ties={ties} p={p:.4g}")
            else:
                b, c, p = mcnemar(triples, frame)
                print(f"    McNemar: none->harm={b} harm->none={c} p={p:.4g}")

    print("\nby conversation structure")
    for frame in FRAMES:
        print(f"\n{frame}")
        for structure in STRUCTURES:
            d_overall = paired_differences(triples, frame, overall, structure)
            d_harm = paired_differences(triples, frame, harm, structure)
            mo, lo, hi = summarize(d_overall, args.draws, seed=11)
            mh, lh, hh = summarize(d_harm, args.draws, seed=13)
            b, c, p = mcnemar(triples, frame, structure)
            print(
                f"  {structure}: overall={mo:.3f} [{lo:.3f}, {hi:.3f}] "
                f"harm={mh:.3f} [{lh:.3f}, {hh:.3f}] "
                f"discordant={b}/{c} p={p:.4g}"
            )


if __name__ == "__main__":
    main()
