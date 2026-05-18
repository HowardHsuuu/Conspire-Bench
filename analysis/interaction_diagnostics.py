"""Moderator analysis for Conspire-Bench frame effects.

The main paper uses paired frame-effect tests. This script asks whether those
paired frame effects vary by conversation structure, model, model family,
parameter count, or topic domain. Parameter-count tests should be interpreted
descriptively because the benchmark has only six target models and one scenario
instantiation per domain-by-structure cell.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats


RESULTS = Path("results/20260515_163319/merged_local_seed24_contexts_6models.json")
OUTDIR = Path("analysis")

MODEL_META = {
    "huggingface/meta-llama/Llama-3.2-3B-Instruct": ("Llama", 3.0),
    "huggingface/meta-llama/Llama-3.1-8B-Instruct": ("Llama", 8.0),
    "huggingface/Qwen/Qwen2.5-0.5B-Instruct": ("Qwen", 0.5),
    "huggingface/Qwen/Qwen2.5-7B-Instruct": ("Qwen", 7.0),
    "huggingface/google/gemma-3-1b-it": ("Gemma", 1.0),
    "huggingface/google/gemma-4-E2B-it": ("Gemma", 2.0),
}

FRAMES = ("brainstorming", "critical_review")
METRICS = ("overall_delta", "harm_delta")
MODERATORS = ("scenario_type", "model_name", "model_family", "category")


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data if isinstance(data, list) else data["detailed_results"]


def score(row: dict) -> float:
    return float(row["judge_scores"]["overall_safety"])


def harm(row: dict) -> float:
    return float(bool(row["judge_scores"]["harmful_elaboration"]))


def build_delta_frame(rows: list[dict]) -> pd.DataFrame:
    triples: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        triples[(row["model_name"], row["scenario_id"])][row["context_label"]] = row

    records = []
    for (model, scenario), frames in sorted(triples.items()):
        if "none" not in frames:
            continue
        family, params_b = MODEL_META[model]
        neutral = frames["none"]
        for frame in FRAMES:
            if frame not in frames:
                continue
            framed = frames[frame]
            records.append(
                {
                    "pair_id": f"{model}::{scenario}",
                    "frame": frame,
                    "scenario_id": scenario,
                    "scenario_type": neutral["scenario_type"],
                    "category": neutral["category"],
                    "model_name": model,
                    "model_short": model.split("/")[-1],
                    "model_family": family,
                    "params_b": params_b,
                    "log_params_b": math.log(params_b),
                    "overall_delta": score(framed) - score(neutral),
                    "harm_delta": harm(framed) - harm(neutral),
                    "neutral_overall": score(neutral),
                    "framed_overall": score(framed),
                    "neutral_harm": harm(neutral),
                    "framed_harm": harm(framed),
                }
            )
    return pd.DataFrame.from_records(records)


def anova_f(values: np.ndarray, groups: np.ndarray) -> float:
    unique = [values[groups == g] for g in sorted(set(groups))]
    if len(unique) < 2:
        return 0.0
    return float(stats.f_oneway(*unique).statistic)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = float(np.mean(values))
    ss_total = float(np.sum((values - grand) ** 2))
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for group in sorted(set(groups)):
        vals = values[groups == group]
        ss_between += len(vals) * float((np.mean(vals) - grand) ** 2)
    return ss_between / ss_total


def permutation_anova(
    values: np.ndarray, groups: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float]:
    observed = anova_f(values, groups)
    rng = random.Random(seed)
    labels = list(groups)
    count = 0
    for _ in range(draws):
        rng.shuffle(labels)
        if anova_f(values, np.array(labels)) >= observed - 1e-12:
            count += 1
    p = (count + 1) / (draws + 1)
    return observed, p, eta_squared(values, groups)


def moderator_tests(df: pd.DataFrame, draws: int) -> pd.DataFrame:
    rows = []
    for frame in FRAMES:
        subset = df[df["frame"] == frame].copy()
        for metric in METRICS:
            values = subset[metric].to_numpy(dtype=float)
            for moderator in MODERATORS:
                groups = subset[moderator].astype(str).to_numpy()
                f_stat, p_perm, eta2 = permutation_anova(
                    values, groups, draws=draws, seed=17
                )
                rows.append(
                    {
                        "frame": frame,
                        "metric": metric,
                        "moderator": moderator,
                        "levels": subset[moderator].nunique(),
                        "f_stat": f_stat,
                        "perm_p": p_perm,
                        "eta2": eta2,
                    }
                )
    return pd.DataFrame(rows)


def model_size_trends(df: pd.DataFrame) -> pd.DataFrame:
    model_means = (
        df.groupby(["frame", "model_name", "model_short", "model_family", "params_b"])
        [list(METRICS)]
        .mean()
        .reset_index()
    )
    rows = []
    for frame in FRAMES:
        subset = model_means[model_means["frame"] == frame]
        x = np.log(subset["params_b"].to_numpy(dtype=float))
        for metric in METRICS:
            y = subset[metric].to_numpy(dtype=float)
            pearson = stats.pearsonr(x, y)
            spearman = stats.spearmanr(x, y)
            rows.append(
                {
                    "frame": frame,
                    "metric": metric,
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                    "n_models": len(subset),
                }
            )
    return model_means, pd.DataFrame(rows)


def omnibus_interaction_anova(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        formula = (
            f"{metric} ~ C(frame) * C(scenario_type) "
            "+ C(frame) * C(model_family) "
            "+ C(frame) * log_params_b "
            "+ C(frame) * C(category)"
        )
        model = smf.ols(formula, data=df).fit()
        table = anova_lm(model, typ=2).reset_index(names="term")
        table["metric"] = metric
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def format_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def write_markdown(
    out: Path,
    moderator_df: pd.DataFrame,
    model_means: pd.DataFrame,
    size_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
) -> None:
    lines = [
        "# Moderator Analysis of Frame Effects",
        "",
        "Moderator analysis over paired frame deltas. P-values for categorical",
        "moderators are label-permutation ANOVA tests on paired deltas; model-size",
        "trends use six model-level mean deltas and should be read descriptively.",
        "",
        "## Permutation ANOVA by Moderator",
        "",
        moderator_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Model Mean Deltas",
        "",
        model_means.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Model Size Trends",
        "",
        size_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Omnibus Delta ANOVA",
        "",
        "OLS ANOVA over paired deltas with frame-by-moderator terms.",
        "",
        omnibus_df.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Short Summary",
        "",
    ]

    for frame in FRAMES:
        lines.append(f"- `{frame}`:")
        for metric in METRICS:
            sub = moderator_df[
                (moderator_df["frame"] == frame) & (moderator_df["metric"] == metric)
            ].sort_values("perm_p")
            best = sub.iloc[0]
            lines.append(
                f"  - strongest categorical heterogeneity for `{metric}` is "
                f"`{best.moderator}` (permutation p={best.perm_p:.3f}, "
                f"eta2={best.eta2:.3f})."
            )
        st = size_df[size_df["frame"] == frame]
        for _, row in st.iterrows():
            lines.append(
                f"  - model-size trend for `{row.metric}`: Spearman "
                f"rho={row.spearman_r:.3f}, p={row.spearman_p:.3f} over six models."
            )
    out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = build_delta_frame(load_rows(args.results))
    moderator_df = moderator_tests(df, draws=args.draws)
    model_means, size_df = model_size_trends(df)
    omnibus_df = omnibus_interaction_anova(df)

    df.to_csv(args.outdir / "frame_delta_rows.csv", index=False)
    moderator_df.to_csv(args.outdir / "interaction_moderator_tests.csv", index=False)
    model_means.to_csv(args.outdir / "model_frame_delta_means.csv", index=False)
    size_df.to_csv(args.outdir / "model_size_trends.csv", index=False)
    omnibus_df.to_csv(args.outdir / "omnibus_interaction_anova.csv", index=False)
    write_markdown(
        args.outdir / "interaction_diagnostics.md",
        moderator_df=moderator_df,
        model_means=model_means,
        size_df=size_df,
        omnibus_df=omnibus_df,
    )

    print((args.outdir / "interaction_diagnostics.md").as_posix())


if __name__ == "__main__":
    main()
