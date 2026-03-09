"""Command-line interface for RAG system."""

import importlib.metadata
import importlib.resources
import json
import logging
import shutil
from pathlib import Path

import click
from scripts.evaluate import evaluate as run_eval
from scripts.evaluate import format_markdown_report

from benchmark.runner import BenchmarkRunner
from config import get_config
from ingestion import DocumentIngester
from pipeline.rag_pipeline import RAGPipeline
from storage.metadata_db import MetadataDB
from storage.vector_db import VectorStore
from ui.web import launch_ui


@click.command()
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


@click.command()
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


@click.command()
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


@click.command()
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


@click.command()
def serve():
    """Start the web UI."""

    launch_ui()


@click.command()
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def clear(yes: bool):
    """Clear all data from databases."""
    if not yes:
        click.confirm("This will delete all ingested data. Continue?", abort=True)

    metadata_db = MetadataDB()
    vector_db = VectorStore()

    metadata_db.clear_all()
    vector_db.delete_collection()

    config = get_config()
    demo_docs = Path(config.metadata_db_path).parent / "demo_docs"
    if demo_docs.exists():
        shutil.rmtree(demo_docs)
        click.echo("Cleared demo_docs/")

    ocr_texts = Path(config.metadata_db_path).parent / "ocr_texts"
    if ocr_texts.exists():
        shutil.rmtree(ocr_texts)
        click.echo("Cleared ocr_texts/")

    click.echo("All data cleared.")


@click.command()
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
    for category in [
        "exact_match_questions",
        "multi_document_synthesis_questions",
        "negative_questions",
        "qualitative_questions",
        "temporal_filter_questions",
        "ocr_questions",
        "multi_hop_ocr_questions",
        # "database_questions",       # not supported yet
        # "multi_hop_db_doc_questions",  # not supported yet
    ]:
        all_questions.extend(gt_data.get(category, []))

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


@click.command()
@click.option(
    "--configs-dir",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default="benchmark_configs",
    help="Directory containing benchmark YAML configs",
)
@click.option(
    "--docs-dir",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default="demo_docs",
    help="Directory with documents to ingest",
)
@click.option(
    "--ground-truth",
    "-g",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to ground truth JSON (default: from sme-synth-data-gen package)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="benchmark_results",
    help="Directory for benchmark output",
)
@click.option(
    "--config-names",
    "-n",
    multiple=True,
    help="Only run configs with these names (default: all)",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force fresh run, ignoring existing results",
)
def benchmark(
    configs_dir: Path,
    docs_dir: Path,
    ground_truth: Path | None,
    output_dir: Path,
    config_names: tuple[str, ...],
    force: bool,
):
    """Run benchmark evaluation across multiple RAG configurations."""
    # Load ground truth
    if ground_truth is not None:
        with open(ground_truth, encoding="utf-8") as f:
            gt_data = json.load(f)
    else:
        try:
            gt_ref = importlib.resources.files("scripts").parent / "dataset" / "ground_truth.json"
            with importlib.resources.as_file(gt_ref) as gt_path:
                with open(gt_path, encoding="utf-8") as f:
                    gt_data = json.load(f)
        except (FileNotFoundError, TypeError):
            local_path = Path("../sme-synth-data-gen/dataset/ground_truth.json")
            if local_path.exists():
                with open(local_path, encoding="utf-8") as f:
                    gt_data = json.load(f)
            else:
                raise click.ClickException(
                    "Ground truth file not found. Use --ground-truth to specify path."
                )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    runner = BenchmarkRunner(
        configs_dir=configs_dir,
        docs_dir=docs_dir,
        ground_truth=gt_data,
        output_dir=output_dir,
        config_names=list(config_names) if config_names else None,
        force=force,
    )

    try:
        runner.run()
    except ValueError as e:
        raise click.ClickException(str(e))


@click.command("generate-docs")
@click.option(
    "--output-dir",
    "-o",
    default="demo_docs",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to write generated documents into.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing files.",
)
def generate_docs(output_dir: Path, force: bool):
    """Generate synthetic SME documents from the sme-synth-data-gen dataset."""
    from scripts.generate_files import (
        generate_docx,
        generate_eml,
        generate_md,
        generate_pdf_easy,
        generate_pdf_hard,
        generate_pptx,
        generate_xlsx,
        set_file_timestamps,
    )

    documents_text = importlib.resources.files("dataset").joinpath("documents.json").read_text()
    docs = json.loads(documents_text)["documents"]

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = skipped = existing = 0
    for doc in docs:
        fmt = doc.get("format", "")
        doc_type = doc["type"]
        dest = output_dir / doc["filename"]

        if dest.exists() and not force:
            existing += 1
            continue

        try:
            if fmt == "pdf":
                if doc.get("pdf_difficulty", "easy") == "easy":
                    generate_pdf_easy(doc, output_dir)
                else:
                    generate_pdf_hard(doc, output_dir)
            elif fmt == "eml" or "email" in doc_type:
                generate_eml(doc, output_dir)
            elif fmt == "md" or doc_type in ["meeting_notes", "project_kickoff"]:
                generate_md(doc, output_dir)
            elif fmt == "docx" or doc_type in [
                "report_quarterly", "report_monthly", "report_project", "proposal"
            ]:
                generate_docx(doc, output_dir)
            elif fmt == "xlsx" or "spreadsheet" in doc_type:
                generate_xlsx(doc, output_dir)
            elif fmt == "pptx" or "presentation" in doc_type:
                generate_pptx(doc, output_dir)
            else:
                click.echo(f"  Unknown format for {doc['id']}: {doc_type}/{fmt}", err=True)
                skipped += 1
                continue

            set_file_timestamps(dest, doc)
            generated += 1
        except Exception as e:
            click.echo(f"  Error generating {doc['filename']}: {e}", err=True)
            skipped += 1

    if existing:
        click.echo(f"Skipped {existing} existing files (use --force to overwrite).")
    suffix = f" {skipped} failed." if skipped else ""
    click.echo(f"Generated {generated} files to {output_dir}/{suffix}")


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
