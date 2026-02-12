"""Summary table generation for benchmark results."""

from pathlib import Path
from typing import Any


def extract_summary_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from an evaluation results dict.

    Args:
        results: Evaluation results from synth_evaluate().

    Returns:
        Dict with total_score, max_score, percentage, full, partial, wrong counts.
    """
    summary = results.get("summary", {})
    return {
        "total_score": summary.get("auto_scored_total_score", 0),
        "max_score": summary.get("auto_scored_max_score", 0),
        "percentage": summary.get("auto_scored_percentage", 0),
        "full_credit": summary.get("full_credit_count", 0),
        "partial_credit": summary.get("partial_credit_count", 0),
        "wrong": summary.get("wrong_count", 0),
    }


def generate_summary_table(
    run_dir: Path,
    config_results: list[dict[str, Any]],
) -> str:
    """Generate a Markdown summary table for a benchmark run.

    Args:
        run_dir: Path to the timestamped run directory.
        config_results: List of dicts, each with keys:
            - config: BenchmarkConfig
            - metrics: dict from extract_summary_metrics()
            - elapsed: float seconds

    Returns:
        Markdown string with results table.
    """
    lines = [
        f"# Benchmark Results — {run_dir.name}",
        "",
        (
            "| Config | Chunk | Top-K | Reranker | LLM "
            "| Score | % | Full/Partial/Wrong | Time | Details |"
        ),
        (
            "|--------|-------|-------|----------|-----"
            "|-------|---|--------------------|------|---------|"
        ),
    ]

    for entry in config_results:
        cfg = entry["config"]
        m = entry["metrics"]
        elapsed = entry["elapsed"]

        config_link = f"[{cfg.name}]({cfg.name}/config.yaml)"
        chunk = f"{cfg.chunk_size}/{cfg.chunk_overlap}"
        reranker = "Yes" if cfg.use_reranker else "No"
        score = f"{m['total_score']}/{m['max_score']}"
        pct = f"{m['percentage']:.1f}"
        breakdown = f"{m['full_credit']}/{m['partial_credit']}/{m['wrong']}"
        time_str = f"{elapsed:.0f}s"
        details_link = f"[results]({cfg.name}/results.json)"

        lines.append(
            f"| {config_link} | {chunk} | {cfg.top_k} | {reranker} "
            f"| {cfg.llm_model} | {score} | {pct} | {breakdown} "
            f"| {time_str} | {details_link} |"
        )

    lines.append("")
    return "\n".join(lines)
