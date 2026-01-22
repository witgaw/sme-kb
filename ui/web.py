"""Gradio web UI for RAG system - Chat interface."""

import json
import tempfile
from pathlib import Path

import gradio as gr
import httpx

from config import get_config
from ingestion import DocumentIngester
from storage.metadata_db import MetadataDB

FILE_ICONS = {
    ".pdf": "[PDF]",
    ".docx": "[DOC]",
    ".xlsx": "[XLS]",
    ".pptx": "[PPT]",
    ".md": "[MD]",
    ".txt": "[TXT]",
    ".eml": "[EML]",
}


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
        self.last_sources: list[dict] = []

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
        # Import here to avoid slow startup
        from pipeline.rag_pipeline import RAGPipeline

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
            import os

            key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not key:
                return "[X] OPENROUTER_API_KEY not set"

        # Initialize (this loads embedding model - can be slow first time)
        try:
            if self.rag:
                self.rag.close()
            self.rag = RAGPipeline(llm_provider=provider.lower(), llm_model=actual_model)
            return self.get_status()
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

    def _format_sources(self) -> str:
        if not self.last_sources:
            return "*No sources*"
        docs = {}
        for chunk in self.last_sources:
            source = chunk.get("source", "unknown")
            if source not in docs:
                docs[source] = []
            docs[source].append(chunk.get("content", "")[:200])

        parts = []
        for source, chunks in docs.items():
            ext = Path(source).suffix.lower()
            icon = FILE_ICONS.get(ext, "[FILE]")
            name = Path(source).name
            parts.append(f"**{icon} {name}**")
            for i, chunk in enumerate(chunks, 1):
                preview = chunk.replace("\n", " ")[:100]
                parts.append(f"_{i}. {preview}..._")
            parts.append("")
        return "\n".join(parts)

    def load_demo(self, progress=gr.Progress()) -> str:
        """Load demo data."""
        try:
            from scripts.generate_files import (
                generate_docx,
                generate_eml,
                generate_md,
                generate_pptx,
                generate_xlsx,
            )

            paths = [
                Path("../sme-synth-data-gen/dataset/documents.json"),
                Path("dataset/documents.json"),
            ]
            data = None
            for p in paths:
                if p.exists():
                    with open(p, encoding="utf-8") as f:
                        data = json.load(f)
                    break
            if not data:
                return "[X] documents.json not found"

            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
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
                    if fmt in gens:
                        try:
                            gens[fmt](doc, out)
                        except Exception:
                            pass
                    if i % 10 == 0:
                        progress(0.1 + 0.4 * i / len(docs))

                progress(0.5, desc="Ingesting...")
                ing = DocumentIngester()
                stats = ing.ingest_directory(out, recursive=True)
                progress(1.0)

                if stats["ingested"] == 0 and stats["skipped_duplicate"] > 0:
                    return f"Already loaded ({stats['skipped_duplicate']} docs)"
                return f"Loaded {stats['ingested']} docs"
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
            ing = DocumentIngester()
            stats = ing.ingest_directory(p, recursive=True)
            progress(1.0)
            if stats["ingested"] == 0 and stats["skipped_duplicate"] > 0:
                return f"Already loaded ({stats['skipped_duplicate']} docs)"
            return f"Loaded {stats['ingested']} docs"
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

    def clear_all(self) -> str:
        """Clear all data from databases."""
        try:
            from storage.vector_db import VectorStore

            mdb = MetadataDB()
            vdb = VectorStore()

            mdb.clear_all()
            vdb.delete_collection()

            return "All data cleared"
        except Exception as e:
            return f"[X] {type(e).__name__}: {e}"

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
                    for cat in ["exact_match_questions", "multi_document_questions"]:
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


def launch_ui(server_name: str = "0.0.0.0", server_port: int = 7860):
    """Launch chat interface."""
    ui = RAGInterface()

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

                with gr.Accordion("Sources", open=False):
                    sources = gr.Markdown("*Ask a question to see sources*")

                with gr.Accordion("Settings", open=True):
                    provider = gr.Radio(
                        ["Ollama", "OpenRouter"],
                        value="Ollama",
                        label="Provider",
                    )

                    # Get initial Ollama models
                    ollama_models = ui.get_ollama_models()
                    default_ollama = ollama_models[0][1] if ollama_models else ""

                    model = gr.Dropdown(
                        ollama_models,
                        value=default_ollama,
                        label="Model",
                    )
                    custom = gr.Textbox(label="Custom model", visible=False)
                    init_btn = gr.Button("Initialize", variant="primary")

                    gr.Markdown("---")
                    top_k = gr.Slider(1, 50, 10, step=1, label="Retrieval chunks")

                def update_model_choices(p):
                    if p == "Ollama":
                        choices = ui.get_ollama_models()
                        value = choices[0][1] if choices else ""
                    else:
                        choices = ui.OPENROUTER_MODELS
                        value = "anthropic/claude-sonnet-4"
                    return gr.update(choices=choices, value=value)

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

                with gr.Accordion("Data", open=False):
                    data_status = gr.Markdown("")
                    ingest_path = gr.Textbox(
                        label="Directory path",
                        placeholder="/path/to/docs",
                        show_label=False,
                    )
                    ingest_btn = gr.Button("Ingest Directory", size="sm", variant="secondary")

                    gr.Markdown("**OR**", elem_classes=["text-center"])
                    demo_btn = gr.Button("Load Demo Data", size="sm", variant="secondary")

                    gr.Markdown("---")
                    clear_data_btn = gr.Button("Clear All Data", size="sm", variant="stop")

                    examples_acc = gr.Accordion("Example Questions", open=False, visible=False)
                    with examples_acc:
                        example_btns = []
                        for ex in ui.examples():
                            btn = gr.Button(ex, size="sm")
                            btn.click(fn=lambda e=ex: e, outputs=msg)
                            example_btns.append(btn)

        # Initialize handler
        def do_init(prov, mod, cust):
            result = ui.initialize(prov, mod, cust)
            if result.startswith("[X]"):
                return f"""
                <div style="background: #ef4444; border: 2px solid #dc2626; color: white;
                            padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                            margin-bottom: 1rem; font-size: 0.95rem;">
                    <div><strong>Error:</strong> {result}</div>
                </div>
                """
            return ui.get_status_html()

        init_btn.click(
            fn=lambda: """
            <div style="background: #f59e0b; border: 2px solid #d97706; color: white;
                        padding: 1rem 1.25rem; border-radius: 8px; font-family: monospace;
                        margin-bottom: 1rem; font-size: 0.95rem;">
                <div><strong>Status:</strong> Initializing...</div>
            </div>
            """,
            outputs=status,
        ).then(
            fn=do_init,
            inputs=[provider, model, custom],
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
            outputs=[data_status, examples_acc],
        ).then(
            fn=ui.load_demo,
            outputs=data_status,
        ).then(
            fn=handle_demo_result,
            inputs=data_status,
            outputs=[status, examples_acc],
        )

        ingest_btn.click(
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
            outputs=[data_status, status, examples_acc],
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

        def get_bot_response(history, k):
            """Get bot response for the last user message."""
            if not history or len(history) == 0:
                return history, "*No sources*"

            if not ui.rag:
                history.append(
                    {
                        "role": "assistant",
                        "content": "System not initialized. Click 'Initialize' in Settings.",
                    }
                )
                return history, "*No sources*"

            try:
                # Get the last message - must be from user
                last_msg = history[-1]
                if isinstance(last_msg, dict):
                    content = last_msg.get("content", "")
                    # Handle case where content might be a list
                    if isinstance(content, list):
                        user_msg = " ".join(str(c) for c in content)
                    else:
                        user_msg = str(content) if content else ""
                else:
                    user_msg = str(last_msg) if last_msg else ""

                # Validate message
                if not user_msg or not user_msg.strip():
                    history.append({"role": "assistant", "content": "Please enter a question."})
                    return history, "*No sources*"

                result = ui.rag.query(user_msg.strip(), top_k=k, return_sources=True)
                ui.last_sources = result.get("chunks", [])
                history.append({"role": "assistant", "content": result["answer"]})
                return history, ui._format_sources()
            except Exception as e:
                import traceback

                error_msg = f"Error: {e}\n{traceback.format_exc()}"
                history.append({"role": "assistant", "content": error_msg})
                ui.last_sources = []
                return history, "*No sources*"

        # Disable send button and show loading in textbox while processing
        send.click(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=get_bot_response,
            inputs=[chatbot, top_k],
            outputs=[chatbot, sources],
        )

        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg],
        ).then(
            fn=get_bot_response,
            inputs=[chatbot, top_k],
            outputs=[chatbot, sources],
        )

        clear.click(fn=lambda: ([], ""), outputs=[chatbot, msg])

        # Auto-initialize on page load
        def auto_init():
            # Get first available Ollama model
            ollama_models = ui.get_ollama_models()
            default_model = ollama_models[0][1] if ollama_models else ""

            # Initialize model
            ui.initialize("Ollama", default_model, "")

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

            # Return status HTML, examples visibility, data message, hide loading overlay
            return (
                ui.get_status_html(),
                gr.update(visible=has_data),
                data_msg,
                gr.update(visible=False),  # Hide overlay after initialization
            )

        demo.load(fn=auto_init, outputs=[status, examples_acc, data_status, loading_overlay])

    demo.launch(server_name=server_name, server_port=server_port, inbrowser=True, css=CSS)


if __name__ == "__main__":
    launch_ui()
