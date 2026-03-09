"""Gradio web UI for RAG system - Chat interface."""

import hashlib
import html as html_lib
import json
import os
import shutil
import sqlite3
import subprocess
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
from scripts.generate_database import (
    create_indexes,
    create_schema,
    create_views,
    insert_data,
)
from scripts.generate_files import (
    generate_docx,
    generate_eml,
    generate_md,
    generate_pdf_easy,
    generate_pdf_hard,
    generate_pptx,
    generate_xlsx,
)

from config import get_config
from ingestion import DocumentIngester
from ingestion.loader import DocumentLoader
from pipeline.rag_pipeline import RAGPipeline
from storage.metadata_db import MetadataDB
from storage.vector_db import VectorStore


def _ask_directory() -> str:
    """Open the native macOS folder picker via osascript."""
    try:
        proc = subprocess.run(
            [
                "osascript",
                "-e",
                'POSIX path of (choose folder with prompt "Select folder to ingest")',
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        path = proc.stdout.strip()
        # osascript returns path with trailing slash
        return path.rstrip("/") if path else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


FILE_ICONS = {
    ".pdf": "[PDF]",
    ".docx": "[DOC]",
    ".xlsx": "[XLS]",
    ".pptx": "[PPT]",
    ".md": "[MD]",
    ".txt": "[TXT]",
    ".eml": "[EML]",
}

_SETTINGS_DEFAULTS: dict[str, Any] = {
    "rag_type": "basic",
    "provider": "Ollama",
    "model": "",
    "top_k": 20,
    "temperature": 0.0,
    "ocr_enabled": False,
    "ocr_model": "llava:7b",
}


def _settings_path() -> Path:
    return Path(get_config().metadata_db_path).parent / "ui_settings.json"


def _load_settings() -> dict[str, Any]:
    try:
        p = _settings_path()
        if p.exists():
            data = json.loads(p.read_text())
            return {**_SETTINGS_DEFAULTS, **data}
    except Exception:
        pass
    return dict(_SETTINGS_DEFAULTS)


def _save_settings(updates: dict[str, Any]) -> None:
    try:
        current = _load_settings()
        current.update(updates)
        _settings_path().write_text(json.dumps(current, indent=2))
    except Exception:
        pass


def check_ollama(base_url: str, model: str) -> tuple[bool, str]:
    """Check Ollama connectivity and model availability. Returns (ok, message)."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{base_url}/api/tags")
            if resp.status_code != 200:
                return False, f"Ollama returned {resp.status_code}"
            models = [m["name"] for m in resp.json().get("models", [])]
            if not models:
                return False, f"No models installed. Run: ollama pull {model}"
            if model not in models:
                return False, f"Model '{model}' not found. Run: ollama pull {model}"
            return True, "OK"
    except httpx.ConnectError:
        return False, "Ollama not running. Run: ollama serve"
    except httpx.TimeoutException:
        return False, "Ollama not responding (timeout)"
    except Exception as e:
        return False, str(e)


class RAGInterface:
    """Web interface for RAG system."""

    OPENROUTER_MODELS = [
        ("Claude Sonnet 4", "anthropic/claude-sonnet-4"),
        ("Claude Opus 4", "anthropic/claude-opus-4"),
        ("GPT-4o", "openai/gpt-4o"),
        ("GPT-4o Mini", "openai/gpt-4o-mini"),
        ("DeepSeek V3", "deepseek/deepseek-chat"),
        ("Custom...", "custom"),
    ]

    def __init__(self):
        self.rag = None
        self.conversation_history: list[dict[str, str]] = []
        self.history_enabled: bool = False
        self.message_metadata: list[dict[str, Any]] = []
        self.session_input_tokens: int = 0
        self.session_output_tokens: int = 0

    def get_token_stats_md(self) -> str:
        """Return session token usage as a short markdown string."""
        if self.session_input_tokens == 0 and self.session_output_tokens == 0:
            return ""
        return (
            f"*Session tokens — in: {self.session_input_tokens:,} "
            f"/ out: {self.session_output_tokens:,}*"
        )

    def get_ollama_models(self) -> list[tuple[str, str]]:
        """Get list of installed Ollama models."""
        config = get_config()
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{config.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                else:
                    return [("Ollama error", ""), ("Custom...", "custom")]
        except Exception:
            return [("Ollama not running", ""), ("Custom...", "custom")]

        if not models:
            return [("No models - run: ollama pull <model>", ""), ("Custom...", "custom")]

        # Format model names nicely
        choices = []
        for model_name in models:
            # Clean up the display name
            if ":" in model_name:
                base, tag = model_name.split(":", 1)
                display = f"{base.title()} ({tag})"
            else:
                display = model_name.title()
            choices.append((display, model_name))

        choices.append(("Custom...", "custom"))
        return choices

    def get_status_html(self) -> str:
        """Get current system status as formatted HTML."""
        try:
            mdb = MetadataDB()
            docs = mdb.get_document_count()
            chunks = mdb.get_chunk_count()
            model_info = f"{self.rag._llm_client.model}" if self.rag else "Not initialized"

            if docs > 0:
                data_info = f"{docs} docs, {chunks} chunks"
            else:
                data_info = "No data"

            return f"""
            <div style="background: #10b981; border: 2px solid #059669; color: white;
                        padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                        margin-bottom: 1rem; font-size: 0.95rem;">
                <div style="margin: 0.35rem 0;"><strong>Model:</strong> {model_info}</div>
                <div style="margin: 0.35rem 0;"><strong>Data:</strong> {data_info}</div>
            </div>
            """
        except Exception:
            return """
            <div style="background: #f59e0b; border: 2px solid #d97706; color: white;
                        padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                        margin-bottom: 1rem; font-size: 0.95rem;">
                <div><strong>Status:</strong> Not initialized</div>
            </div>
            """

    def initialize(self, provider: str, model: str, custom: str) -> str:
        """Initialize RAG pipeline. Returns status message."""
        actual_model = custom.strip() if model == "custom" else model
        if not actual_model:
            return "[X] Model name required"

        config = get_config()

        # Check provider first
        if provider.lower() == "ollama":
            ok, msg = check_ollama(config.ollama_base_url, actual_model)
            if not ok:
                return f"[X] {msg}"
        else:
            key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                return "[X] OPENROUTER_API_KEY not set"

        # Initialize or update model
        try:
            if self.rag:
                # Pipeline already exists - just change the LLM model (fast!)
                self.rag.set_llm_model(provider=provider.lower(), model=actual_model)
            else:
                # First initialization - loads embedding model (slow first time)
                self.rag = RAGPipeline(llm_provider=provider.lower(), llm_model=actual_model)
            return self.get_status_html()
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

    def chat(self, message: str, history: list, top_k: int):
        """Process chat message."""
        if not self.rag:
            history.append({"role": "user", "content": message})
            history.append(
                {
                    "role": "assistant",
                    "content": "System not initialized. Click 'Initialize' in Settings.",
                }
            )
            return history, "*No sources*"

        if not message.strip():
            return history, self._format_sources()

        try:
            result = self.rag.query(message, top_k=top_k, return_sources=True)
            self.last_sources = result.get("chunks", [])
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": result["answer"]})
            return history, self._format_sources()
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Error: {e}"})
            self.last_sources = []
            return history, "*No sources*"

    def _format_prompt_details(
        self,
        prompt: str = "",
        usage: dict[str, int] | None = None,
        conversation_history: list[dict] | None = None,
    ) -> str:
        """Format prompt, usage, and conversation history details."""
        if not prompt and not usage and not conversation_history:
            return "*No details available*"

        parts = []

        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            parts.append("### Token Usage")
            parts.append("")
            parts.append(f"- **Input tokens:** {prompt_tokens:,}")
            parts.append(f"- **Output tokens:** {completion_tokens:,}")
            parts.append(f"- **Total tokens:** {total_tokens:,}")
            parts.append("")

        if conversation_history:
            n_turns = len(conversation_history) // 2
            parts.append(f"### Conversation History ({n_turns} prior turn(s))")
            parts.append("")
            for msg in conversation_history:
                role = msg.get("role", "?").upper()
                content = msg.get("content", "")
                parts.append(f"**{role}:** {content}")
                parts.append("")

        if prompt:
            parts.append("### Full Prompt")
            parts.append("")
            parts.append("```")
            parts.append(prompt)
            parts.append("```")
            parts.append("")

        return "\n".join(parts)

    def _format_sources_html(self, sources: list[dict] | None = None) -> str:
        """Format sources grouped by document with collapsible chunk text."""
        if not sources:
            return "<p style='color:var(--body-text-color);opacity:0.5;'>No sources</p>"

        # Group chunks by document, preserving order
        docs: dict[str, list[dict]] = {}
        for chunk in sources:
            source = chunk.get("source", "unknown")
            if source not in docs:
                docs[source] = []
            docs[source].append(chunk)

        out = ["<div style='display:flex;flex-direction:column;gap:0.75rem;'>"]

        for source, chunks in docs.items():
            ext = Path(source).suffix.lower()
            icon = FILE_ICONS.get(ext, "[FILE]")
            name = html_lib.escape(Path(source).name)

            out.append("<div>")
            # Document header
            out.append(
                f"<div style='font-weight:600;font-size:0.875rem;"
                f"padding:0.25rem 0;margin-bottom:0.35rem;"
                f"border-bottom:1px solid var(--border-color-primary,#e5e7eb);'>"
                f"{icon} {name}</div>"
            )

            # Chunks
            for i, chunk in enumerate(chunks, 1):
                content = chunk.get("content", "")
                preview = content.replace("\n", " ")[:100]
                if len(content) > 100:
                    preview += "…"

                meta_parts = []
                if "distance" in chunk:
                    meta_parts.append(f"dist {chunk['distance']:.3f}")
                if "rerank_score" in chunk:
                    meta_parts.append(f"rerank {chunk['rerank_score']:.3f}")
                meta_str = (
                    f"<span style='opacity:0.5;margin-left:0.5rem;font-size:0.75rem;'>"
                    f"({', '.join(meta_parts)})</span>"
                    if meta_parts
                    else ""
                )

                escaped_content = html_lib.escape(content)
                escaped_preview = html_lib.escape(preview)

                out.append(
                    f"<details style='margin-bottom:0.2rem;'>"
                    f"<summary style='cursor:pointer;font-size:0.8rem;padding:0.2rem 0;"
                    f"list-style:none;display:flex;gap:0.4rem;align-items:baseline;"
                    f"opacity:0.85;'>"
                    f"<span style='flex-shrink:0;opacity:0.5;'>#{i}</span>"
                    f"{meta_str}"
                    f"<span style='overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>"
                    f"{escaped_preview}</span>"
                    f"</summary>"
                    f"<pre style='margin:0.3rem 0 0.3rem 1rem;padding:0.5rem 0.75rem;"
                    f"font-size:0.8rem;white-space:pre-wrap;border-radius:4px;"
                    f"background:var(--background-fill-secondary,#f9fafb);"
                    f"border:1px solid var(--border-color-primary,#e5e7eb);'>"
                    f"{escaped_content}</pre>"
                    f"</details>"
                )

            out.append("</div>")

        out.append("</div>")
        return "\n".join(out)

    def load_demo(self, progress=gr.Progress()) -> str:
        """Load demo data."""
        try:
            dataset_paths = [
                Path("../sme-synth-data-gen/dataset"),
                Path("dataset"),
            ]
            dataset_dir = None
            for p in dataset_paths:
                if (p / "documents.json").exists():
                    dataset_dir = p
                    break
            if not dataset_dir:
                return "[X] documents.json not found"

            docs_json = dataset_dir / "documents.json"
            with open(docs_json, encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
            docs_hash = hashlib.sha256(raw.encode()).hexdigest()

            out = Path(get_config().metadata_db_path).parent / "demo_docs"
            hash_file = out / ".docs_hash"
            if out.exists() and hash_file.exists() and hash_file.read_text() == docs_hash:
                print("[kb] demo_docs up to date, skipping generation")
            else:
                if out.exists():
                    shutil.rmtree(out)
                out.mkdir(parents=True, exist_ok=True)

            progress(0.1, desc="Generating files...")
            gens = {
                "eml": generate_eml,
                "docx": generate_docx,
                "xlsx": generate_xlsx,
                "pptx": generate_pptx,
                "md": generate_md,
            }
            docs = data.get("documents", [])
            for i, doc in enumerate(docs):
                fmt = doc.get("format", "")
                if fmt == "pdf":
                    difficulty = doc.get("pdf_difficulty", "easy")
                    gen_fn = generate_pdf_hard if difficulty == "hard" else generate_pdf_easy
                    try:
                        gen_fn(doc, out)
                    except Exception:
                        pass
                elif fmt in gens:
                    try:
                        gens[fmt](doc, out)
                    except Exception:
                        pass
                if i % 10 == 0:
                    progress(0.1 + 0.4 * i / len(docs))

            # Generate database if database.json exists alongside documents.json
            db_json = dataset_dir / "database.json"
            if db_json.exists():
                try:
                    with open(db_json, encoding="utf-8") as f:
                        db_def = json.load(f)
                    db_path = out / db_def["meta"]["database_name"]
                    conn = sqlite3.connect(str(db_path))
                    try:
                        create_schema(conn, db_def["schema"])
                        insert_data(conn, db_def["data"])
                        create_indexes(conn)
                        create_views(conn)
                    finally:
                        conn.close()
                except Exception:
                    pass  # DB generation is best-effort

            hash_file.write_text(docs_hash)

            progress(0.5, desc="Ingesting...")
            # Auto-enable OCR for demo if vision model is available
            ocr_model = _load_settings().get("ocr_model", "llava:7b")
            ocr_available, _ = check_ollama(get_config().ollama_base_url, ocr_model)
            if ocr_available:
                print(f"[kb] OCR model '{ocr_model}' detected, enabling OCR for demo ingestion")
            ing = _make_ingester(ocr_override=True if ocr_available else None)
            stats = ing.ingest_directory(out, recursive=True)
            progress(1.0)

            _log_ingestion_stats(stats, source="demo dataset")
            if stats["ingested"] == 0 and stats["skipped_duplicate"] > 0:
                return f"Already loaded ({stats['skipped_duplicate']} docs)"
            return _format_ingest_result(stats)
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

    def ingest_dir(self, path: str, progress=gr.Progress()) -> str:
        """Ingest directory."""
        if not path.strip():
            return ""
        p = Path(path)
        if not p.exists():
            return f"[X] Directory not found: {path}"
        try:
            progress(0.1)
            ing = _make_ingester()
            stats = ing.ingest_directory(p, recursive=True)
            progress(1.0)
            _log_ingestion_stats(stats, source=str(p))
            if stats["ingested"] == 0 and stats["skipped_duplicate"] > 0:
                return f"Already loaded ({stats['skipped_duplicate']} docs)"
            return _format_ingest_result(stats)
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

    def clear_all(self) -> str:
        """Clear all data from databases."""
        try:
            mdb = MetadataDB()
            vdb = VectorStore()

            mdb.clear_all()
            vdb.delete_collection()

            return "All data cleared"
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

    def clear_conversation_history(self) -> None:
        """Clear conversation history, message metadata, and session token counts."""
        self.conversation_history = []
        self.message_metadata = []
        self.session_input_tokens = 0
        self.session_output_tokens = 0

    def toggle_history_mode(self, enabled: bool) -> None:
        """Toggle conversation history mode."""
        self.history_enabled = enabled
        if not enabled:
            # Only reset LLM context — keep message_metadata so details panel stays intact
            self.conversation_history = []

    def examples(self) -> list[str]:
        try:
            for p in [
                Path("../sme-synth-data-gen/dataset/ground_truth.json"),
                Path("dataset/ground_truth.json"),
            ]:
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        gt = json.load(f)
                    out = []
                    for cat in ["exact_match_questions", "multi_document_synthesis_questions"]:
                        for q in gt.get(cat, [])[:3]:
                            if not q.get("requires_ocr") and not q.get("requires_database"):
                                out.append(q.get("question_pl", ""))
                    return out[:5]
        except Exception:
            pass
        return ["Kiedy Maciej Boryna podpisał umowe ze Smakosz?"]


CSS = """
.status-box {
    padding: 0.75rem 1rem !important;
    border-radius: 6px !important;
    font-family: monospace !important;
    margin-bottom: 1rem !important;
}
.status-box.error {
    background: #fef2f2 !important;
    border: 1px solid #fecaca !important;
    color: #991b1b !important;
}
.status-box.ok {
    background: #f0fdf4 !important;
    border: 1px solid #bbf7d0 !important;
    color: #166534 !important;
}
.status-box.loading {
    background: #fefce8 !important;
    border: 1px solid #fef08a !important;
    color: #854d0e !important;
}
.text-center {
    text-align: center;
    margin: 0.5rem 0;
}
/* User chat bubbles are not clickable */
.user-row, .user-row * { cursor: default !important; }
/* Token stats */
.token-stats { font-size: 0.75rem; opacity: 0.6; text-align: right; margin-top: 0.25rem; }
/* Browse button */
.browse-btn button { font-family: monospace; letter-spacing: 0.05em; }
/* Loading overlay */
.loading-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(0, 0, 0, 0.8) !important;
    z-index: 9999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    backdrop-filter: blur(5px) !important;
}
.loading-message {
    background: #ff6600 !important;
    padding: 3rem 4rem !important;
    border-radius: 16px !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: white !important;
    box-shadow: 0 8px 32px rgba(255, 102, 0, 0.5) !important;
    text-align: center !important;
}
"""


def _make_ingester(ocr_override: bool | None = None) -> DocumentIngester:
    """Create a DocumentIngester with OCR settings from the persisted UI settings.

    Args:
        ocr_override: If set, force-enable/disable OCR regardless of saved settings.
    """
    saved = _load_settings()
    config = get_config()
    ocr_enabled = ocr_override if ocr_override is not None else saved.get("ocr_enabled", False)
    config.ocr_enabled = ocr_enabled
    config.ocr_model = saved.get("ocr_model", "llava:7b")
    return DocumentIngester()


def _format_ingest_result(stats: dict) -> str:
    """Return a short human-readable summary of an ingest run."""
    ingested = stats["ingested"]
    total = stats["total_files"]
    unreadable = stats.get("unreadable", 0)
    unsupported = stats.get("unsupported", 0)
    notes = []
    if unreadable:
        notes.append(f"{unreadable} unreadable")
    if unsupported:
        notes.append(f"{unsupported} unsupported type")
    msg = f"Loaded {ingested}/{total} docs"
    if notes:
        msg += f" ({', '.join(notes)})"
    return msg


def _log_ingestion_stats(stats: dict, source: str = "") -> None:
    """Print ingestion results to the terminal."""
    label = f" from {source}" if source else ""
    total = stats["total_files"]
    ingested = stats["ingested"]
    dupes = stats["skipped_duplicate"]
    unreadable = stats.get("unreadable", 0)
    unsupported = stats.get("unsupported", 0)
    failed = stats["failed"]
    errors = stats.get("errors", [])
    ocr_files = stats.get("ocr_files", [])

    print(
        f"[kb] ingested{label}: {ingested}/{total} new"
        + (f", {dupes} duplicate(s) skipped" if dupes else "")
        + (f", {unreadable} unreadable" if unreadable else "")
        + (f", {unsupported} unsupported type(s)" if unsupported else "")
        + (f", {failed} failed" if failed else "")
        + (f", {len(ocr_files)} OCR'd" if ocr_files else "")
    )

    if ocr_files:
        for f in ocr_files:
            model = f.get("ocr_model") or "?"
            pdf_path = Path(f["filepath"])
            pdf_link = _term_link(pdf_path.resolve().as_uri(), pdf_path.name)
            text_path = f.get("ocr_text_path")
            if text_path:
                text_link = _term_link(Path(text_path).resolve().as_uri(), text_path)
                path_info = f" -> {text_link}"
            else:
                path_info = ""
            print(f"  [ocr:{model}] {pdf_link}{path_info}")

    if dupes:
        for filepath in stats.get("duplicate_files", []):
            print(f"  [duplicate] {Path(filepath).name}")

    if unreadable:
        for filepath in stats.get("unreadable_files", []):
            print(f"  [unreadable] {Path(filepath).name}")

    if unsupported:
        for filepath in stats.get("unsupported_files", []):
            p = Path(filepath)
            print(f"  [unsupported] {p.name} ({p.suffix})")

    for err in errors:
        name = Path(err["file"]).name
        print(f"  [fail] {name}: {err['error']}")


def _term_link(url: str, text: str) -> str:
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def _log_startup_diagnostics() -> None:
    """Print a summary of what is currently indexed."""
    try:
        mdb = MetadataDB()
        doc_count = mdb.get_document_count()
        chunk_count = mdb.get_chunk_count()
    except Exception as e:
        print(f"[warn] Could not read metadata DB: {e}")
        return

    print(f"[kb] {doc_count} documents indexed, {chunk_count} chunks")

    config = get_config()
    ocr_dir = Path(config.metadata_db_path).parent / "ocr_texts"

    docs = mdb.get_all_documents()
    indexed_paths = set()
    for doc in docs:
        fp = Path(doc["filepath"])
        indexed_paths.add(fp.resolve())
        src_link = _term_link(fp.resolve().as_uri(), str(fp)) if fp.exists() else str(fp)
        ocr_txt = ocr_dir / (fp.stem + ".txt")
        if ocr_txt.exists():
            txt_link = _term_link(ocr_txt.resolve().as_uri(), str(ocr_txt))
            print(f"  [doc] {src_link} -> {txt_link}")
        else:
            print(f"  [doc] {src_link}")

    # Report source files that were not indexed (unreadable, unsupported, failed)
    demo_dir = Path(config.metadata_db_path).parent / "demo_docs"
    not_indexed = []
    if demo_dir.is_dir():
        for f in sorted(demo_dir.iterdir()):
            if f.is_file() and not f.name.startswith(".") and f.resolve() not in indexed_paths:
                not_indexed.append(f)
    if not_indexed:
        supported = DocumentLoader.SUPPORTED_EXTENSIONS
        unsupported = [f for f in not_indexed if f.suffix.lower() not in supported]
        missing = [f for f in not_indexed if f.suffix.lower() in supported]
        if unsupported:
            print(f"[kb] {len(unsupported)} unsupported file(s):")
            for f in unsupported:
                link = _term_link(f.resolve().as_uri(), f.name)
                print(f"  [unsupported] {link} ({f.suffix})")
        if missing:
            print(f"[kb] {len(missing)} file(s) not indexed (re-run ingestion):")
            for f in missing:
                link = _term_link(f.resolve().as_uri(), f.name)
                print(f"  [not indexed] {link}")


def launch_ui(server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch chat interface."""
    ui = RAGInterface()
    saved = _load_settings()

    # Define the loading overlay HTML
    loading_overlay_html = (
        '<div style="position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; '
        "background: rgba(0, 0, 0, 0.9); z-index: 99999; display: flex; "
        "align-items: center; justify-content: center; backdrop-filter: blur(5px); "
        'pointer-events: all;">'
        '<div style="background: #ff6600; padding: 3rem 4rem; border-radius: 16px; '
        "font-size: 2rem; font-weight: 700; color: white; "
        'box-shadow: 0 8px 32px rgba(255, 102, 0, 0.6); text-align: center;">'
        "Loading model & demo data..."
        "</div>"
        "</div>"
    )

    with gr.Blocks(title="RAG Chat", fill_height=True) as demo:
        # Loading overlay - starts visible, then hidden after init
        loading_overlay = gr.HTML(value=loading_overlay_html, visible=True)

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    height=600,
                    show_label=False,
                    layout="bubble",
                    buttons=["copy"],
                )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask about your documents...",
                        show_label=False,
                        scale=4,
                        container=False,
                        autofocus=True,
                    )
                    send = gr.Button("Send", scale=1, variant="primary")

                clear = gr.Button("Clear Chat", size="sm")
                token_stats = gr.Markdown("", elem_classes=["token-stats"])

            # Sidebar
            with gr.Column(scale=1):
                # Status display
                status = gr.HTML(
                    value="""
                    <div style="background: #f59e0b; border: 2px solid #d97706; color: white;
                                padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                                margin-bottom: 1rem; font-size: 0.95rem;">
                        <div><strong>Status:</strong> Not initialized</div>
                    </div>
                    """
                )

                with gr.Accordion("Settings", open=False):
                    rag_type = gr.Radio(
                        [("Basic", "basic"), ("Knowledge Graph", "knowledge_graph")],
                        value=saved["rag_type"],
                        label="RAG Type",
                    )

                    provider = gr.Radio(
                        ["Ollama", "OpenRouter"],
                        value=saved["provider"],
                        label="Provider",
                    )

                    # Get initial model choices based on saved provider
                    if saved["provider"] == "OpenRouter":
                        initial_choices = ui.OPENROUTER_MODELS
                        default_model = saved["model"] or "anthropic/claude-sonnet-4"
                    else:
                        initial_choices = ui.get_ollama_models()
                        saved_m = saved["model"]
                        model_vals = [v for _, v in initial_choices]
                        default_model = (
                            saved_m
                            if saved_m in model_vals
                            else (initial_choices[0][1] if initial_choices else "")
                        )

                    model = gr.Dropdown(
                        initial_choices,
                        value=default_model,
                        label="Model",
                    )
                    custom = gr.Textbox(label="Custom model", visible=False)
                    init_btn = gr.Button("Initialize", variant="primary")

                    gr.Markdown("---")
                    top_k = gr.Slider(1, 50, saved["top_k"], step=1, label="Retrieval chunks")
                    temperature = gr.Slider(
                        0.0,
                        1.0,
                        saved["temperature"],
                        step=0.1,
                        label="Temperature",
                        info="0 = deterministic, higher = more creative",
                    )

                    history_toggle = gr.Checkbox(
                        label="Enable conversation history",
                        value=False,
                        info="LLM remembers previous messages. Uses more tokens.",
                    )

                    language_selector = gr.Radio(
                        choices=[("Polski", "pl"), ("English", "en")],
                        value="pl",
                        label="Response language",
                        info="Language for LLM responses",
                    )

                    gr.Markdown("---")
                    gr.Markdown("**OCR (scanned PDFs)**")
                    ocr_toggle = gr.Checkbox(
                        label="Enable OCR fallback",
                        value=saved.get("ocr_enabled", False),
                        info="Use a vision model to extract text from image-only PDFs.",
                    )
                    ocr_model = gr.Textbox(
                        label="OCR Model",
                        value=saved.get("ocr_model", "llava:7b"),
                        info="Ollama vision model for OCR (e.g. llava:7b, moondream)",
                    )

                def update_model_choices(p):
                    if p == "Ollama":
                        choices = ui.get_ollama_models()
                        value = choices[0][1] if choices else ""
                    else:
                        choices = ui.OPENROUTER_MODELS
                        value = "anthropic/claude-sonnet-4"
                    return gr.update(choices=choices, value=value)

                rag_type.change(
                    fn=lambda v: _save_settings({"rag_type": v}),
                    inputs=rag_type,
                )
                provider.change(
                    fn=update_model_choices,
                    inputs=provider,
                    outputs=model,
                )
                model.change(
                    fn=lambda m: gr.update(visible=(m == "custom")),
                    inputs=model,
                    outputs=custom,
                )
                history_toggle.change(
                    fn=ui.toggle_history_mode,
                    inputs=history_toggle,
                )
                top_k.change(
                    fn=lambda v: _save_settings({"top_k": v}),
                    inputs=top_k,
                )

                def _update_temperature(v):
                    _save_settings({"temperature": v})
                    get_config().temperature = v

                temperature.change(
                    fn=_update_temperature,
                    inputs=temperature,
                )
                ocr_toggle.change(
                    fn=lambda v: _save_settings({"ocr_enabled": v}),
                    inputs=ocr_toggle,
                )
                ocr_model.change(
                    fn=lambda v: _save_settings({"ocr_model": v}),
                    inputs=ocr_model,
                )

                with gr.Accordion("Data", open=False, elem_id="data-accordion") as data_acc:
                    data_status = gr.Markdown("")
                    with gr.Row():
                        ingest_path = gr.Textbox(
                            placeholder="/path/to/docs",
                            show_label=False,
                            scale=5,
                            container=False,
                        )
                        browse_btn = gr.Button(
                            "...",
                            scale=1,
                            size="sm",
                            min_width=36,
                            elem_classes=["browse-btn"],
                        )
                    ingest_btn = gr.Button("Ingest Directory", size="sm", variant="secondary")

                    gr.Markdown("**OR**", elem_classes=["text-center"])
                    demo_btn = gr.Button("Load Demo Data", size="sm", variant="secondary")

                    gr.Markdown("---")
                    clear_data_btn = gr.Button("Clear All Data", size="sm", variant="stop")

                with gr.Column(visible=False) as examples_col:
                    gr.Markdown("**Example Questions**")
                    example_btns = []
                    for ex in ui.examples():
                        btn = gr.Button(ex, size="sm")
                        btn.click(fn=lambda e=ex: e, outputs=msg)
                        example_btns.append(btn)

        # Message Details Section (full-width below chat)
        with gr.Accordion("Message Details", open=False):
            selected_message_label = gr.Markdown("*Click on an assistant message to view details*")

            with gr.Tabs():
                with gr.Tab("Sources"):
                    sources = gr.HTML("<p><em>Click on a message to see sources</em></p>")

                with gr.Tab("Prompt & Usage"):
                    prompt_details = gr.Markdown("*Click on a message to see prompt details*")

        # Initialize handler
        def do_init(prov, mod, cust, rt):
            if rt == "knowledge_graph":
                return """
                <div style="background: #ef4444; border: 2px solid #dc2626; color: white;
                            padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                            margin-bottom: 1rem; font-size: 0.95rem;">
                    <div><strong>Error:</strong> Knowledge Graph RAG is not yet supported.</div>
                </div>
                """
            result = ui.initialize(prov, mod, cust)
            if result.startswith("[X]"):
                return f"""
                <div style="background: #ef4444; border: 2px solid #dc2626; color: white;
                            padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                            margin-bottom: 1rem; font-size: 0.95rem;">
                    <div><strong>Error:</strong> {result}</div>
                </div>
                """
            actual_model = cust.strip() if mod == "custom" else mod
            _save_settings({"provider": prov, "model": actual_model})
            return ui.get_status_html()

        init_btn.click(
            fn=lambda: (
                """
            <div style="background: #f59e0b; border: 2px solid #d97706; color: white;
                        padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                        margin-bottom: 1rem; font-size: 0.95rem;">
                <div><strong>Status:</strong> Initializing...</div>
            </div>
            """
            ),
            outputs=status,
        ).then(
            fn=do_init,
            inputs=[provider, model, custom, rag_type],
            outputs=status,
        )

        # Data handlers
        def handle_demo_result(result):
            # Show examples if demo loaded successfully or already loaded
            show_examples = not result.startswith("[X]") and (
                "Loaded" in result or "Already loaded" in result
            )
            return (ui.get_status_html(), gr.update(visible=show_examples))

        demo_btn.click(
            fn=lambda: ("Loading...", gr.update(visible=False)),
            outputs=[data_status, examples_col],
        ).then(
            fn=ui.load_demo,
            outputs=data_status,
        ).then(
            fn=handle_demo_result,
            inputs=data_status,
            outputs=[status, examples_col],
        )

        # Native OS folder picker → auto-ingest
        browse_btn.click(
            fn=_ask_directory,
            outputs=ingest_path,
        ).then(
            fn=lambda p: "Ingesting..." if p.strip() else "",
            inputs=ingest_path,
            outputs=data_status,
        ).then(
            fn=ui.ingest_dir,
            inputs=ingest_path,
            outputs=data_status,
        ).then(
            fn=ui.get_status_html,
            outputs=status,
        )

        ingest_btn.click(
            fn=lambda: "Ingesting...",
            outputs=data_status,
        ).then(
            fn=ui.ingest_dir,
            inputs=ingest_path,
            outputs=data_status,
        ).then(
            fn=ui.get_status_html,
            outputs=status,
        )

        def clear_all_data():
            result = ui.clear_all()
            return (result, ui.get_status_html(), gr.update(visible=False))

        clear_data_btn.click(
            fn=clear_all_data,
            outputs=[data_status, status, examples_col],
        )

        # Message details handlers
        def show_message_details(evt: gr.SelectData):
            """Display details for the clicked message."""
            if not ui.message_metadata:
                return (
                    "*No message data available*",
                    "*No sources*",
                    "*No details*",
                )

            # evt.index is the index in the chatbot history
            # Chat history alternates: user (even), assistant (odd)
            # Only show details for assistant messages (odd indices)
            if evt.index % 2 == 0:
                # User message clicked - ignore
                return (
                    "*Click on an assistant response (not your question) to view details*",
                    "*No sources*",
                    "*No details*",
                )

            # Assistant message clicked
            metadata_idx = (evt.index - 1) // 2

            # Check if metadata exists for this index
            if metadata_idx < 0 or metadata_idx >= len(ui.message_metadata):
                return (
                    "*No details available for this message*",
                    "*No sources*",
                    "*No details*",
                )

            meta = ui.message_metadata[metadata_idx]

            label_parts = [f"**Response #{metadata_idx + 1}**", ""]
            history = meta.get("conversation_history", [])
            for i in range(0, len(history) - 1, 2):
                label_parts.append(f"**Q:** {history[i].get('content', '')}")
                label_parts.append("")
                label_parts.append(f"**A:** {history[i + 1].get('content', '')}")
                label_parts.append("")
                label_parts.append("---")
                label_parts.append("")
            label_parts.append(f"**Q:** {meta['query']}")
            label_parts.append("")
            label_parts.append(f"**A:** {meta['answer']}")
            message_label = "\n".join(label_parts)

            sources_html = ui._format_sources_html(meta["sources"])
            details_text = ui._format_prompt_details(
                meta["prompt"],
                meta["usage"],
                meta.get("conversation_history"),
            )

            return message_label, sources_html, details_text

        chatbot.select(
            fn=show_message_details,
            outputs=[selected_message_label, sources, prompt_details],
        )

        # Chat handlers
        def add_user_message(message, history):
            """Add user message immediately and clear input."""
            if not message.strip():
                return history if history else [], ""
            if history is None:
                history = []
            history.append({"role": "user", "content": message})
            return history, ""

        def get_bot_response(history, k, history_enabled, language, rt):
            """Get bot response for the last user message."""
            no_change = ui.get_token_stats_md()
            if not history or len(history) == 0:
                return history, no_change

            if not ui.rag:
                if rt == "knowledge_graph":
                    msg = "Knowledge Graph RAG is not yet supported. Switch to Basic in Settings."
                else:
                    msg = "System not initialized. Click 'Initialize' in Settings."
                history.append({"role": "assistant", "content": msg})
                return history, no_change

            try:
                # Get the last message - must be from user
                last_msg = history[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                    # Handle case where content might be a list of dicts (Gradio multimodal)
                    if isinstance(content, list):
                        user_msg = " ".join(
                            c.get("text", str(c)) if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    else:
                        user_msg = str(content) if content else ""
                else:
                    user_msg = str(last_msg) if last_msg else ""

                # Validate message
                if not user_msg or not user_msg.strip():
                    history.append({"role": "assistant", "content": "Please enter a question."})
                    return history, no_change

                # Determine what to send to pipeline
                conv_history = ui.conversation_history if history_enabled else None

                # Query with full details and optional conversation history
                result = ui.rag.query(
                    user_msg.strip(),
                    top_k=k,
                    return_sources=True,
                    return_details=True,
                    conversation_history=conv_history,
                    language=language,
                )

                # Get answer without token count
                answer = result["answer"]

                # Accumulate session token usage
                usage = result.get("usage", {})
                ui.session_input_tokens += usage.get("prompt_tokens", 0)
                ui.session_output_tokens += usage.get("completion_tokens", 0)

                # Store metadata for this message
                metadata = {
                    "sources": result.get("chunks", []),
                    "prompt": result.get("prompt", ""),
                    "usage": usage,
                    "query": user_msg.strip(),
                    "answer": answer,
                    "conversation_history": list(conv_history) if conv_history else [],
                }
                ui.message_metadata.append(metadata)

                # Update conversation history if enabled
                if history_enabled:
                    ui.conversation_history.append({"role": "user", "content": user_msg.strip()})
                    ui.conversation_history.append({"role": "assistant", "content": answer})

                history.append({"role": "assistant", "content": answer})
                return history, ui.get_token_stats_md()
            except Exception as e:
                error_msg = f"Error: {e}\n{traceback.format_exc()}"
                history.append({"role": "assistant", "content": error_msg})
                return history, ui.get_token_stats_md()

        # Disable send button and show loading in textbox while processing
        send.click(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=get_bot_response,
            inputs=[chatbot, top_k, history_toggle, language_selector, rag_type],
            outputs=[chatbot, token_stats],
        )

        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=get_bot_response,
            inputs=[chatbot, top_k, history_toggle, language_selector, rag_type],
            outputs=[chatbot, token_stats],
        )

        def clear_chat_and_history():
            ui.clear_conversation_history()
            return [], "", ""

        clear.click(fn=clear_chat_and_history, outputs=[chatbot, msg, token_stats])

        # Auto-initialize on page load
        def auto_init():
            s = _load_settings()
            init_provider = s["provider"]
            init_model = s["model"]
            get_config().temperature = s.get("temperature", 0.0)

            # Fall back to first available Ollama model if no saved model
            if not init_model and init_provider == "Ollama":
                ollama_models = ui.get_ollama_models()
                init_model = ollama_models[0][1] if ollama_models else ""

            init_result = ui.initialize(init_provider, init_model, "")
            if init_result.startswith("[X]"):
                print(f"[warn] Auto-init failed: {init_result[4:]}")

            # Load demo data if no data exists
            try:
                mdb = MetadataDB()
                has_data = mdb.get_document_count() > 0
            except Exception:
                has_data = False

            data_msg = ""
            if not has_data:
                # Try to load demo data
                data_msg = ui.load_demo()
                has_data = not data_msg.startswith("[X]") and (
                    "Loaded" in data_msg or "Already loaded" in data_msg
                )

            _log_startup_diagnostics()

            return (
                ui.get_status_html(),
                gr.update(visible=has_data),  # examples col: show if data
                data_msg,
                gr.update(visible=False),  # hide loading overlay
                gr.update(open=False),  # data accordion: start collapsed
            )

        demo.load(
            fn=auto_init,
            outputs=[status, examples_col, data_status, loading_overlay, data_acc],
        )

    demo.launch(server_name=server_name, server_port=server_port, inbrowser=True, css=CSS)


if __name__ == "__main__":
    launch_ui()
