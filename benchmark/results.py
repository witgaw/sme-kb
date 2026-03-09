"""Summary table generation for benchmark results."""

from pathlib import Path
from typing import Any

# Display-friendly category names
_CATEGORY_LABELS = {
    "exact_match_questions": "Exact match",
    "multi_document_synthesis_questions": "Multi-doc synthesis",
    "negative_questions": "Negative",
    "qualitative_questions": "Qualitative",
    "temporal_filter_questions": "Temporal filter",
    "ocr_questions": "OCR",
    "multi_hop_ocr_questions": "Multi-hop OCR",
}


def extract_summary_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Extract key metrics from an evaluation results dict.

    Args:
        results: Evaluation results from synth_evaluate().

    Returns:
        Dict with total_score, max_score, percentage, full, partial, wrong counts,
        and per_category breakdown.
    """
    summary = results.get("summary", {})

    # Per-category scores from auto_scored entries
    per_category: dict[str, dict[str, float]] = {}
    for entry in results.get("auto_scored", []):
        cat = entry.get("category", "unknown")
        score = entry.get("score", 0.0)
        if cat not in per_category:
            per_category[cat] = {"score": 0.0, "count": 0}
        per_category[cat]["score"] += score
        per_category[cat]["count"] += 1

    # Temporal results (separate scoring path)
    temporal_pass = summary.get("temporal_pass", 0)
    temporal_total = summary.get("temporal_total", 0)
    if temporal_total:
        per_category["temporal_filter_questions"] = {
            "score": temporal_pass,
            "count": temporal_total,
        }

    return {
        "total_score": summary.get("auto_scored_total_score", 0),
        "max_score": summary.get("auto_scored_max_score", 0),
        "percentage": summary.get("auto_scored_percentage", 0),
        "full_credit": summary.get("full_credit_count", 0),
        "partial_credit": summary.get("partial_credit_count", 0),
        "wrong": summary.get("wrong_count", 0),
        "per_category": per_category,
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
        Markdown string with overall results table and per-category breakdown.
    """
    lines = [
        f"# Benchmark Results — {run_dir.name}",
        "",
        "## Overall",
        "",
        (
            "| Config | Chunk | Top-K | Reranker | LLM | OCR"
            " | Score | % | Full/Partial/Wrong | Time | Details |"
        ),
        (
            "|--------|-------|-------|----------|-----|-----"
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
        ocr = cfg.ocr_model if cfg.ocr_enabled else "—"
        score = f"{m['total_score']}/{m['max_score']}"
        pct = f"{m['percentage']:.1f}"
        breakdown = f"{m['full_credit']}/{m['partial_credit']}/{m['wrong']}"
        time_str = f"{elapsed:.0f}s" if elapsed else "-"
        details_link = f"[results]({cfg.name}/results.json)"

        lines.append(
            f"| {config_link} | {chunk} | {cfg.top_k} | {reranker} | {cfg.llm_model}"
            f" | {ocr} | {score} | {pct} | {breakdown} | {time_str} | {details_link} |"
        )

    # Per-category breakdown — collect all categories seen across all configs
    all_cats: list[str] = []
    for entry in config_results:
        for cat in entry["metrics"].get("per_category", {}):
            if cat not in all_cats:
                all_cats.append(cat)

    if all_cats:
        lines += ["", "## By category", ""]
        config_names = [e["config"].name for e in config_results]
        header = "| Category | " + " | ".join(config_names) + " |"
        sep = "|----------|" + "|".join("---" for _ in config_names) + "|"
        lines += [header, sep]

        for cat in all_cats:
            label = _CATEGORY_LABELS.get(cat, cat)
            cells = []
            for entry in config_results:
                cat_data = entry["metrics"].get("per_category", {}).get(cat)
                if cat_data:
                    s = cat_data["score"]
                    n = cat_data["count"]
                    pct = 100 * s / n if n else 0
                    cells.append(f"{s:.0f}/{n} ({pct:.0f}%)")
                else:
                    cells.append("—")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)
