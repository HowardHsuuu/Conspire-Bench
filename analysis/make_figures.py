#!/usr/bin/env python3
"""Generate Conspire-Bench result figures from merged JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FRAMES = ["none", "brainstorming", "critical_review"]
FRAME_LABELS = {
    "none": "Neutral",
    "brainstorming": "Brainstorming",
    "critical_review": "Critical review",
}
FRAME_LABELS_COMPACT = {
    "none": "Neutral",
    "brainstorming": "Brainstorming",
    "critical_review": "Critical\nreview",
}
STRUCTURES = [
    "single_turn_complete_logic",
    "multi_turn_progression",
    "complete_logic_then_resistance",
]
STRUCTURE_LABELS = {
    "single_turn_complete_logic": "Single-turn",
    "multi_turn_progression": "Multi-turn",
    "complete_logic_then_resistance": "Resistance",
}


def short_model(name: str) -> str:
    return (
        name.split("/")[-1]
        .replace("-Instruct", "")
        .replace("-it", "")
        .replace("gemma-", "Gemma-")
    )


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data if isinstance(data, list) else data.get("detailed_results", [])
    for row in rows:
        row["_agg"] = aggregate_row(row)
    return rows


def aggregate_row(row: dict) -> dict:
    judges = [
        judge
        for judge in row.get("judge_results", [])
        if not judge.get("error") and judge.get("scores")
    ]
    out: dict[str, float | bool] = {}
    for dim in [
        "conspiracy_amplification",
        "reality_grounding",
        "appropriate_uncertainty",
        "overall_safety",
    ]:
        vals = []
        for judge in judges:
            scores = judge.get("scores", {})
            val = scores.get(dim)
            if dim == "overall_safety" and val is None:
                val = judge.get("overall_safety")
            if isinstance(val, (int, float)):
                vals.append(float(val))
        if vals:
            out[dim] = float(np.mean(vals))

    harms = []
    for judge in judges:
        val = judge.get("scores", {}).get("harmful_elaboration")
        if isinstance(val, bool):
            harms.append(val)
    if harms:
        out["harmful_elaboration"] = any(harms)

    if "overall_safety" not in out and isinstance(row.get("total_safety_score"), (int, float)):
        out["overall_safety"] = float(row["total_safety_score"])
    return out


def grouped_mean(rows: list[dict], field: str, **filters: str) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            val = row["_agg"].get(field)
            if isinstance(val, (int, float)):
                vals.append(float(val))
    return float(np.mean(vals))


def grouped_harm(rows: list[dict], **filters: str) -> float:
    vals = []
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            val = row["_agg"].get("harmful_elaboration")
            if isinstance(val, bool):
                vals.append(val)
    return 100.0 * sum(vals) / len(vals)


def save_both(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_structure_heatmap(rows: list[dict], outdir: Path) -> None:
    overall = np.array(
        [
            [grouped_mean(rows, "overall_safety", context_label=f, scenario_type=s) for f in FRAMES]
            for s in STRUCTURES
        ]
    )
    harm = np.array(
        [
            [grouped_harm(rows, context_label=f, scenario_type=s) for f in FRAMES]
            for s in STRUCTURES
        ]
    )

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.55), constrained_layout=True)
    panels = [
        (axes[0], overall, "Overall safety", "YlGnBu", 2.7, 3.8, "{:.2f}"),
        (axes[1], harm, "Harmful elaboration (%)", "OrRd", 0, 90, "{:.0f}"),
    ]
    for ax, data, title, cmap, vmin, vmax, fmt in panels:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(FRAMES)), [FRAME_LABELS_COMPACT[f] for f in FRAMES])
        ax.set_yticks(range(len(STRUCTURES)), [STRUCTURE_LABELS[s] for s in STRUCTURES])
        ax.tick_params(labelsize=8.2)
        ax.set_title(title, fontsize=9.8, weight="bold")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.055, pad=0.03)
        cbar.ax.tick_params(labelsize=7.5)
    save_both(fig, outdir, "figA1_structure_heatmap")


def plot_model_trajectories(rows: list[dict], outdir: Path) -> None:
    models = sorted({row["model_name"] for row in rows}, key=short_model)
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(3.45, 3.35))
    x = np.arange(len(FRAMES))
    for i, model in enumerate(models):
        ys = [
            grouped_mean(rows, "overall_safety", model_name=model, context_label=frame)
            for frame in FRAMES
        ]
        ax.plot(
            x,
            ys,
            marker="o",
            lw=1.55,
            ms=3.6,
            color=colors[i % len(colors)],
            label=short_model(model),
        )
    ax.set_xticks(x, [FRAME_LABELS_COMPACT[f] for f in FRAMES])
    ax.set_ylabel("Overall safety")
    ax.set_ylim(2.7, 4.15)
    ax.set_xlim(-0.08, 2.08)
    ax.set_title("Model trajectories", loc="left", fontsize=9.8, weight="bold")
    ax.tick_params(labelsize=8.2)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=7.1,
        handlelength=1.55,
        columnspacing=0.75,
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save_both(fig, outdir, "figA2_model_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/20260515_163319/merged_local_seed24_contexts_6models.json"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("ConspireBench_paper/latex/figures"),
    )
    args = parser.parse_args()
    rows = load_rows(args.results)
    plot_structure_heatmap(rows, args.outdir)
    plot_model_trajectories(rows, args.outdir)


if __name__ == "__main__":
    main()
