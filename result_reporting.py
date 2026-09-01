"""Human-readable reporting for completed benchmark result bundles."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any


def print_results_summary(results: dict[str, Any], is_adversarial: bool) -> None:
    """Print a compact summary without collapsing V2 outcomes into one score."""

    print("\nRESULTS SUMMARY")
    print("=" * 50)
    for model, stats in results.get("summary", {}).items():
        print(f"\nModel: {model}")
        if "overall_safety_score" in stats:
            print(
                "   Legacy Overall Safety Score: "
                f"{stats['overall_safety_score']:.2f}/5.0"
            )
        for metric, value in stats.get("metric_means", {}).items():
            print(f"   {metric}: {value:.2f}")
        if is_adversarial:
            print(
                "   Persona Effectiveness: "
                f"{stats.get('persona_effectiveness', 0):.2f}/5.0"
            )
            print(
                f"   Conversations Completed: {stats.get('conversations_completed', 0)}"
            )
            print(f"   Conversations Failed: {stats.get('conversations_failed', 0)}")
        else:
            print(f"   Scenarios Completed: {stats.get('scenarios_completed', 0)}")
            print(f"   Scenarios Failed: {stats.get('scenarios_failed', 0)}")


def write_analysis(results: dict[str, Any], output_path: str | Path) -> None:
    """Write a descriptive run summary; inferential analyses live in ``analysis/``."""

    detailed = results.get("detailed_results") or []
    summary = results.get("summary") or {}
    failed = [row for row in detailed if row.get("error")]
    completed = [row for row in detailed if not row.get("error")]
    failure_categories = Counter(
        str(row.get("category") or row.get("user_persona") or "unknown")
        for row in failed
    )

    lines = [
        "Conspire-Bench run summary",
        "=" * 40,
        "",
        f"Result rows: {len(detailed)}",
        f"Completed rows: {len(completed)}",
        f"Failed rows: {len(failed)}",
        f"Models represented: {len(summary)}",
        "",
        "Per-model descriptive outcomes:",
    ]
    for model, stats in summary.items():
        lines.append(f"- {model}")
        for metric, value in stats.get("metric_means", {}).items():
            lines.append(f"  - {metric}: {value:.4f}")
        if "overall_safety_score" in stats:
            lines.append(
                "  - legacy_overall_safety: "
                f"{stats['overall_safety_score']:.4f} (rubric v1 only)"
            )
    if not summary:
        lines.append("- No completed model summaries.")

    lines.extend(["", "Failures by category/persona:"])
    if failure_categories:
        lines.extend(
            f"- {label}: {count}" for label, count in failure_categories.most_common()
        )
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "This file is descriptive only. Use the prespecified scripts in analysis/ ",
            "for paired frame effects, uncertainty intervals, paraphrase robustness, ",
            "judge-family sensitivity, and human-validation analyses.",
            "",
        ]
    )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
