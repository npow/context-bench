"""Markdown comparison report."""

from __future__ import annotations

from context_bench.results import EvalResult


def to_markdown(result: EvalResult) -> str:
    """Generate a markdown comparison table from an EvalResult."""
    if not result.summary:
        return "No results to report."

    systems = list(result.summary.keys())
    if not systems:
        return "No systems in results."

    # Collect all metric names across systems
    all_metrics: list[str] = []
    for sys_metrics in result.summary.values():
        for m in sys_metrics:
            if m not in all_metrics:
                all_metrics.append(m)

    if not all_metrics:
        return "No metrics computed."

    lines: list[str] = []
    lines.append("# Evaluation Results\n")

    # Summary table
    header = "| System | " + " | ".join(all_metrics) + " |"
    sep = "|--------|" + "|".join("-" * (len(m) + 2) for m in all_metrics) + "|"
    lines.append(header)
    lines.append(sep)

    for system in systems:
        values = []
        for metric in all_metrics:
            v = result.summary[system].get(metric)
            if v is None:
                values.append("—")
            elif isinstance(v, float) and v != float("inf"):
                values.append(f"{v:.4f}")
            else:
                values.append(str(v))
        lines.append(f"| {system} | " + " | ".join(values) + " |")

    lines.append("")

    # Timing
    if result.timing:
        lines.append("## Timing\n")
        for system, elapsed in result.timing.items():
            lines.append(f"- **{system}**: {elapsed:.2f}s")
        lines.append("")

    # Config
    if result.config:
        lines.append(f"*{result.config.get('num_examples', '?')} examples evaluated*")

    return "\n".join(lines)
