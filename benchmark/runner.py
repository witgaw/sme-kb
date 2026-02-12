"""Benchmark runner — orchestrates multi-config RAG evaluation."""

import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from scripts.evaluate import evaluate as synth_evaluate

from benchmark.config_schema import BenchmarkConfig
from benchmark.results import extract_summary_metrics, generate_summary_table
from config import RAGConfig, get_config, set_config
from ingestion import DocumentIngester
from pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Silence per-request HTTP noise from httpx/httpcore (Ollama transport)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

console = Console()


class BenchmarkRunner:
    """Run multiple RAG configurations against a ground-truth Q&A dataset."""

    def __init__(
        self,
        configs_dir: Path,
        docs_dir: Path,
        ground_truth: dict[str, Any],
        output_dir: Path = Path("benchmark_results"),
        config_names: list[str] | None = None,
        force: bool = False,
    ):
        self.configs_dir = Path(configs_dir)
        self.docs_dir = Path(docs_dir)
        self.ground_truth = ground_truth
        self.output_dir = Path(output_dir)
        self.config_names = config_names
        self.force = force

    def _load_configs(self) -> list[BenchmarkConfig]:
        """Load and optionally filter BenchmarkConfig objects from YAML files."""
        configs: list[BenchmarkConfig] = []
        for yaml_path in sorted(self.configs_dir.glob("*.yaml")):
            cfg = BenchmarkConfig.from_yaml(str(yaml_path))
            if self.config_names and cfg.name not in self.config_names:
                continue
            configs.append(cfg)
        return configs

    def _find_latest_run_dir(self) -> Path | None:
        """Find the most recent timestamped run directory in output_dir."""
        if not self.output_dir.exists():
            return None
        candidates = [
            d for d in self.output_dir.iterdir() if d.is_dir() and _TIMESTAMP_RE.match(d.name)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.name)

    @staticmethod
    def _fingerprint_hash(fingerprint: str) -> str:
        """Short hex hash for an index fingerprint (used as directory name)."""
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:12]

    def _extract_questions(self, cfg: BenchmarkConfig) -> list[dict[str, Any]]:
        """Extract and filter questions from ground truth based on config flags."""
        all_questions: list[dict[str, Any]] = []
        for category in [
            "exact_match_questions",
            "multi_document_questions",
            "negative_questions",
            "qualitative_questions",
        ]:
            all_questions.extend(self.ground_truth.get(category, []))

        filtered = []
        for q in all_questions:
            if cfg.skip_ocr and q.get("requires_ocr"):
                continue
            if cfg.skip_db and q.get("requires_database"):
                continue
            filtered.append(q)
        return filtered

    def _apply_config(self, cfg: BenchmarkConfig, index_dir: str) -> None:
        """Set the global RAGConfig to match a BenchmarkConfig with isolated storage.

        RAGConfig uses aliases (e.g. VECTOR_DB_PATH) and ``extra="ignore"``, so
        we must pass alias names — Python field names are silently dropped.
        """
        rag_cfg = RAGConfig(
            VECTOR_DB_PATH=str(Path(index_dir) / "chroma_db"),
            METADATA_DB_PATH=str(Path(index_dir) / "metadata.db"),
            CHUNK_SIZE=cfg.chunk_size,
            CHUNK_OVERLAP=cfg.chunk_overlap,
            EMBEDDING_MODE=cfg.embedding_mode,
            EMBEDDING_MODEL=cfg.embedding_model,
            TOP_K=cfg.top_k,
            USE_RERANKER=cfg.use_reranker,
            LLM_PROVIDER=cfg.llm_provider,
            LLM_MODEL=cfg.llm_model,
            TEMPERATURE=cfg.temperature,
            MAX_TOKENS=cfg.max_tokens,
            OCR_ENABLED=cfg.ocr_enabled,
            OCR_MODEL=cfg.ocr_model,
        )
        set_config(rag_cfg)

    @staticmethod
    def _index_dir_has_data(index_dir: Path) -> bool:
        """Return True if the index directory contains a non-empty chroma_db."""
        chroma_dir = index_dir / "chroma_db"
        if not chroma_dir.exists():
            return False
        return any(chroma_dir.iterdir())

    def _print_config_header(self, cfg: BenchmarkConfig, idx: int, total: int) -> None:
        """Print a rich panel with config details before evaluation starts."""
        parts = [
            f"chunk_size [cyan]{cfg.chunk_size}[/]  chunk_overlap [cyan]{cfg.chunk_overlap}[/]",
            f"top_k [cyan]{cfg.top_k}[/]",
            f"reranker [cyan]{'on' if cfg.use_reranker else 'off'}[/]",
            f"llm [cyan]{cfg.llm_model}[/]",
            f"temp [cyan]{cfg.temperature}[/]",
        ]
        if cfg.ocr_enabled:
            parts.append(f"ocr [cyan]{cfg.ocr_model}[/]")
        details = "  ".join(parts)
        console.print(
            Panel(
                details,
                title=f"[bold magenta]{cfg.name}[/]  ({idx}/{total})",
                border_style="blue",
                padding=(0, 1),
            )
        )

    def _print_results_table(self, config_results: list[dict[str, Any]]) -> None:
        """Print a rich summary table to the console."""
        table = Table(title="Benchmark Results", border_style="blue", header_style="bold cyan")

        table.add_column("Config", style="bold")
        table.add_column("Chunk", justify="center")
        table.add_column("Top-K", justify="center")
        table.add_column("Reranker", justify="center")
        table.add_column("LLM")
        table.add_column("Score", justify="right")
        table.add_column("%", justify="right")
        table.add_column("Full/Partial/Wrong", justify="center")
        table.add_column("Time", justify="right")

        for entry in config_results:
            cfg = entry["config"]
            m = entry["metrics"]
            elapsed = entry["elapsed"]
            pct = m["percentage"]

            if pct >= 70:
                pct_style = "bold green"
            elif pct >= 40:
                pct_style = "yellow"
            else:
                pct_style = "red"

            table.add_row(
                cfg.name,
                f"{cfg.chunk_size}/{cfg.chunk_overlap}",
                str(cfg.top_k),
                "Yes" if cfg.use_reranker else "No",
                cfg.llm_model,
                f"{m['total_score']}/{m['max_score']}",
                Text(f"{pct:.1f}", style=pct_style),
                f"{m['full_credit']}/{m['partial_credit']}/{m['wrong']}",
                f"{elapsed:.0f}s" if elapsed else "-",
            )

        console.print()
        console.print(table)

    def run(self) -> Path:
        """Execute the full benchmark.

        Returns:
            Path to the run directory.
        """
        configs = self._load_configs()
        if not configs:
            raise ValueError(f"No configs found in {self.configs_dir}")

        console.print(
            f"\n[bold]Running benchmark:[/] "
            f"[cyan]{len(configs)}[/] configs from [dim]{self.configs_dir}[/]"
        )

        # Decide run directory: reuse latest or create new
        if self.force:
            run_dir = None
        else:
            run_dir = self._find_latest_run_dir()

        if run_dir is None:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = self.output_dir / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)

        indexes_dir = run_dir / "_indexes"
        indexes_dir.mkdir(exist_ok=True)

        # Group configs by index fingerprint to avoid redundant ingestion
        fp_groups: dict[str, list[BenchmarkConfig]] = defaultdict(list)
        for cfg in configs:
            fp_groups[cfg.index_fingerprint].append(cfg)

        config_results: list[dict[str, Any]] = []

        original_config = get_config()

        try:
            # Ingest once per unique fingerprint
            for fp, fp_configs in fp_groups.items():
                fp_hash = self._fingerprint_hash(fp)
                index_dir = indexes_dir / fp_hash
                index_dir.mkdir(parents=True, exist_ok=True)

                representative = fp_configs[0]
                self._apply_config(representative, str(index_dir))

                if self._index_dir_has_data(index_dir):
                    console.print(
                        f"  [dim]Index [bold]{fp_hash}[/bold] already exists, skipping ingestion[/]"
                    )
                    continue

                console.print(
                    f"  [yellow]Ingesting[/] index [bold]{fp_hash}[/]"
                    f" ({len(fp_configs)} configs share this index)"
                )
                ingester = DocumentIngester()
                ingester.ingest_directory(self.docs_dir)

            console.print()

            # Evaluate each config
            for cfg_idx, cfg in enumerate(configs, 1):
                fp_hash = self._fingerprint_hash(cfg.index_fingerprint)
                index_dir_str = str(indexes_dir / fp_hash)

                self._apply_config(cfg, index_dir_str)

                config_dir = run_dir / cfg.name
                config_dir.mkdir(parents=True, exist_ok=True)

                results_path = config_dir / "results.json"

                # Skip evaluation if results already exist
                if results_path.exists():
                    console.print(
                        f"  [dim]Skipping [bold]{cfg.name}[/bold] (results already exist)[/]"
                    )
                    with open(results_path, encoding="utf-8") as f:
                        old_results = json.load(f)
                    metrics = extract_summary_metrics(old_results)
                    config_results.append({"config": cfg, "metrics": metrics, "elapsed": 0.0})
                    continue

                # Save config
                with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(cfg.model_dump(), f, default_flow_style=False)

                questions = self._extract_questions(cfg)
                if not questions:
                    console.print(
                        f"  [yellow]No questions for [bold]{cfg.name}[/bold] after filtering[/]"
                    )
                    continue

                self._print_config_header(cfg, cfg_idx, len(configs))

                rag = RAGPipeline()
                try:
                    submissions: dict[str, str] = {}
                    t0 = time.monotonic()

                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(bar_width=30),
                        MofNCompleteColumn(),
                        TimeElapsedColumn(),
                        console=console,
                    ) as progress:
                        task = progress.add_task(cfg.name, total=len(questions))

                        for i, q in enumerate(questions, 1):
                            qid = q["id"]
                            question_text = q.get("question_pl", q.get("question_en", ""))
                            q_preview = question_text.replace("\n", " ")[:100]

                            progress.update(task, description=f"[dim]{qid}[/] {q_preview}")

                            try:
                                result = rag.query(question_text, return_sources=False)
                                answer = result.get("answer", "")
                                submissions[qid] = answer
                            except Exception:
                                logger.exception("Error querying %s for config %s", qid, cfg.name)
                                submissions[qid] = ""

                            progress.advance(task)

                            # Print completed Q&A above the progress bar
                            a_preview = submissions[qid].replace("\n", " ")[:120]
                            progress.console.print(f"  [dim]{qid}[/] [bold]{q_preview}[/]")
                            progress.console.print(f"       [green]{a_preview}[/]")

                    elapsed = time.monotonic() - t0

                    # Save submissions
                    with open(config_dir / "submissions.json", "w", encoding="utf-8") as f:
                        json.dump(submissions, f, indent=2, ensure_ascii=False)

                    # Evaluate
                    results = synth_evaluate(submissions, ground_truth=self.ground_truth)

                    # Save results
                    with open(results_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                    metrics = extract_summary_metrics(results)
                    config_results.append({"config": cfg, "metrics": metrics, "elapsed": elapsed})

                    pct = metrics["percentage"]
                    if pct >= 70:
                        score_style = "bold green"
                    elif pct >= 40:
                        score_style = "yellow"
                    else:
                        score_style = "red"
                    console.print(
                        f"  [{score_style}]{pct:.1f}%[/] "
                        f"({metrics['total_score']}/{metrics['max_score']}) "
                        f"in {elapsed:.0f}s\n"
                    )
                finally:
                    rag.close()

            # Generate summary (always rebuilt to include both old and new results)
            summary_md = generate_summary_table(run_dir, config_results)
            (run_dir / "summary.md").write_text(summary_md, encoding="utf-8")

            self._print_results_table(config_results)

            console.print(f"\n[dim]Results:[/] {run_dir}")

            return run_dir

        finally:
            set_config(original_config)
