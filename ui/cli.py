"""Command-line interface for RAG system."""

import importlib.metadata
import importlib.resources
import json
from pathlib import Path

import click

from config import get_config
from ingestion import DocumentIngester
from pipeline.rag_pipeline import RAGPipeline
from storage.metadata_db import MetadataDB
from storage.vector_db import VectorStore


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RAG System CLI - Document ingestion and querying."""
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--embedding-mode",
    default=None,
    type=click.Choice(["local", "hosted"]),
    help="Embedding mode (default: from config)",
)
@click.option("--recursive/--no-recursive", default=True, help="Scan subdirectories")
def ingest(directory: Path, embedding_mode: str | None, recursive: bool):
    """Ingest documents from a directory."""
    click.echo(f"Ingesting documents from {directory}...")

    ingester = DocumentIngester(embedding_mode=embedding_mode)

    def progress_callback(filepath: str, current: int, total: int):
        click.echo(f"[{current}/{total}] {filepath}")

    stats = ingester.ingest_directory(
        directory,
        recursive=recursive,
        progress_callback=progress_callback,
    )

    unreadable = stats.get("unreadable", 0)
    unsupported = stats.get("unsupported", 0)

    click.echo("\nIngestion complete!")
    total_found = stats["total_files"] + unsupported
    click.echo(f"  Files found: {total_found} ({stats['total_files']} supported)")
    click.echo(f"  Ingested: {stats['ingested']}")
    click.echo(f"  Skipped (duplicate): {stats['skipped_duplicate']}")
    if unreadable:
        click.echo(f"  Unreadable (no text): {unreadable}")
    if unsupported:
        click.echo(f"  Unsupported type: {unsupported}")
    click.echo(f"  Failed: {stats['failed']}")

    if unreadable:
        for filepath in stats.get("unreadable_files", []):
            click.echo(f"    [unreadable] {Path(filepath).name}")

    if unsupported:
        for filepath in stats.get("unsupported_files", []):
            p = Path(filepath)
            click.echo(f"    [unsupported] {p.name} ({p.suffix})")

    if stats["errors"]:
        click.echo("\nErrors:")
        for error in stats["errors"]:
            click.echo(f"  {error['file']}: {error['error']}")


@cli.command()
@click.argument("question")
@click.option("--model", default=None, help="LLM model name (default: from config)")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["ollama", "openrouter"]),
    help="LLM provider (default: from config)",
)
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve")
@click.option("--no-sources", is_flag=True, help="Don't show sources")
def query(
    question: str,
    model: str | None,
    provider: str | None,
    top_k: int | None,
    no_sources: bool,
):
    """Query the RAG system."""
    rag = RAGPipeline(llm_provider=provider, llm_model=model)

    try:
        result = rag.query(question, top_k=top_k, return_sources=not no_sources)

        click.echo("\n" + "=" * 80)
        click.echo("ANSWER:")
        click.echo("=" * 80)
        click.echo(result["answer"])

        if not no_sources and "sources" in result:
            click.echo("\n" + "=" * 80)
            click.echo("SOURCES:")
            click.echo("=" * 80)
            unique_sources = list(dict.fromkeys(result["sources"]))  # Preserve order
            for i, source in enumerate(unique_sources, 1):
                click.echo(f"[{i}] {source}")
    finally:
        rag.close()


@cli.command()
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve")
def retrieve(question: str, top_k: int | None):
    """Retrieve relevant chunks without generating (debug command)."""
    rag = RAGPipeline()

    try:
        chunks = rag.retrieve_only(question, top_k=top_k)

        click.echo(f"\nRetrieved {len(chunks)} chunks:\n")
        for i, chunk in enumerate(chunks, 1):
            click.echo("=" * 80)
            click.echo(f"[{i}] Source: {chunk['source']}")
            if "distance" in chunk and chunk["distance"] is not None:
                click.echo(f"    Distance: {chunk['distance']:.4f}")
            click.echo("-" * 80)
            # Truncate long content
            content = chunk["content"]
            if len(content) > 500:
                content = content[:500] + "..."
            click.echo(content)
    finally:
        rag.close()


@cli.command()
def stats():
    """Show database statistics."""
    config = get_config()

    metadata_db = MetadataDB()
    vector_db = VectorStore()

    doc_count = metadata_db.get_document_count()
    chunk_count = metadata_db.get_chunk_count()
    embedding_count = vector_db.count()

    click.echo("\nDatabase Statistics:")
    click.echo(f"  Documents: {doc_count}")
    click.echo(f"  Chunks: {chunk_count}")
    click.echo(f"  Embeddings: {embedding_count}")
    click.echo("\nStorage:")
    click.echo(f"  Vector DB: {config.vector_db_path}")
    click.echo(f"  Metadata DB: {config.metadata_db_path}")


@cli.command()
def serve():
    """Start the web UI."""
    from ui.web import launch_ui

    launch_ui()


@cli.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def clear(yes: bool):
    """Clear all data from databases."""
    if not yes:
        click.confirm("This will delete all ingested data. Continue?", abort=True)

    metadata_db = MetadataDB()
    vector_db = VectorStore()

    metadata_db.clear_all()
    vector_db.delete_collection()

    click.echo("All data cleared.")


@cli.command()
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    help="Path to ground truth JSON (default: from sme-synth-data-gen package)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for results (default: stdout)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format",
)
@click.option("--model", default=None, help="LLM model name (default: from config)")
@click.option(
    "--provider",
    default=None,
    type=click.Choice(["ollama", "openrouter"]),
    help="LLM provider (default: from config)",
)
@click.option(
    "--questions",
    "-q",
    multiple=True,
    help="Specific question IDs to evaluate (default: all)",
)
@click.option(
    "--skip-ocr/--include-ocr",
    default=True,
    help="Skip OCR-requiring questions (default: skip)",
)
@click.option(
    "--skip-db/--include-db",
    default=True,
    help="Skip database-requiring questions (default: skip)",
)
def evaluate(
    ground_truth: Path | None,
    output: Path | None,
    output_format: str,
    model: str | None,
    provider: str | None,
    questions: tuple[str, ...],
    skip_ocr: bool,
    skip_db: bool,
):
    """Run evaluation against ground truth dataset.

    Uses the sme-synth-data-gen evaluation framework.
    """
    from scripts.evaluate import evaluate as run_eval
    from scripts.evaluate import format_markdown_report

    # Load ground truth
    if ground_truth is None:
        # Try to find from package data
        try:
            gt_ref = importlib.resources.files("scripts").parent / "dataset" / "ground_truth.json"
            with importlib.resources.as_file(gt_ref) as gt_path:
                with open(gt_path, encoding="utf-8") as f:
                    gt_data = json.load(f)
        except (FileNotFoundError, TypeError):
            # Fall back to local path (if running from synth-data-gen checkout)
            local_path = Path("../sme-synth-data-gen/dataset/ground_truth.json")
            if local_path.exists():
                with open(local_path, encoding="utf-8") as f:
                    gt_data = json.load(f)
            else:
                raise click.ClickException(
                    "Ground truth file not found. Use --ground-truth to specify path."
                )
    else:
        with open(ground_truth, encoding="utf-8") as f:
            gt_data = json.load(f)

    # Extract questions to answer
    all_questions = []
    for category in ["exact_match_questions", "multi_document_questions", "negative_questions"]:
        all_questions.extend(gt_data.get(category, []))

    # Also include qualitative if present
    all_questions.extend(gt_data.get("qualitative_questions", []))

    # Filter questions
    filtered_questions = []
    for q in all_questions:
        qid = q.get("id", "")

        # Skip if specific questions requested and this isn't one
        if questions and qid not in questions:
            continue

        # Skip OCR questions if requested
        if skip_ocr and q.get("requires_ocr"):
            continue

        # Skip DB questions if requested
        if skip_db and q.get("requires_database"):
            continue

        filtered_questions.append(q)

    if not filtered_questions:
        raise click.ClickException("No questions to evaluate after filtering.")

    click.echo(f"Evaluating {len(filtered_questions)} questions...")

    # Initialize RAG pipeline
    rag = RAGPipeline(llm_provider=provider, llm_model=model)

    try:
        # Run each question through RAG
        submissions = {}
        for i, q in enumerate(filtered_questions, 1):
            qid = q["id"]
            question_text = q.get("question_pl", q.get("question_en", ""))

            click.echo(f"[{i}/{len(filtered_questions)}] {qid}: {question_text[:50]}...")

            try:
                result = rag.query(question_text, return_sources=False)
                answer = result.get("answer", "")
                submissions[qid] = answer
            except Exception as e:
                click.echo(f"  Error: {e}")
                submissions[qid] = ""

        # Run evaluation
        results = run_eval(submissions, ground_truth=gt_data)

        # Format output
        if output_format == "json":
            output_text = json.dumps(results, indent=2, ensure_ascii=False)
        else:
            rubrics = results.get("rubrics")
            output_text = format_markdown_report(results, rubrics)

        # Write output
        if output:
            output.write_text(output_text, encoding="utf-8")
            click.echo(f"\nResults written to {output}")
        else:
            click.echo("\n" + output_text)

    finally:
        rag.close()


def help_cmd():
    """Print all available uv run commands with their descriptions."""
    try:
        eps = importlib.metadata.distribution("sme-kb").entry_points
    except importlib.metadata.PackageNotFoundError:
        click.echo("Package 'sme-kb' not found. Run: uv pip install -e .", err=True)
        raise SystemExit(1)

    # Build name -> description by importing the Click object referenced by each entry point
    rows: list[tuple[str, str]] = []
    for ep in eps:
        if ep.group != "console_scripts":
            continue
        try:
            obj = ep.load()
            desc = getattr(obj, "help", None) or getattr(obj, "__doc__", None) or ""
            desc = desc.strip().splitlines()[0] if desc.strip() else ""
        except Exception:
            desc = ""
        rows.append((ep.name, desc))

    rows.sort(key=lambda r: r[0])
    max_name = max(len(r[0]) for r in rows) if rows else 0
    click.echo("Available commands (uv run <command>):\n")
    for name, desc in rows:
        click.echo(f"  {name:<{max_name}}  {desc}")


if __name__ == "__main__":
    cli()
