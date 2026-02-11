"""Document ingestion orchestration."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langdetect import LangDetectException, detect

from config import get_config
from ingestion.chunker import DocumentChunker
from ingestion.deduplicator import compute_document_hash, is_duplicate
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from storage.metadata_db import MetadataDB
from storage.vector_db import VectorStore

logger = logging.getLogger(__name__)

# langdetect code → human-readable name for the OCR prompt
_LANG_NAMES = {
    "pl": "Polish",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "cs": "Czech",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "ru": "Russian",
}


class DocumentIngester:
    """Orchestrates document ingestion pipeline."""

    def __init__(
        self,
        vector_db_path: str | None = None,
        metadata_db_path: str | None = None,
        embedding_mode: str | None = None,
    ):
        """Initialize ingester with all required components.

        Args:
            vector_db_path: Path to ChromaDB storage. Uses config default if None.
            metadata_db_path: Path to SQLite database. Uses config default if None.
            embedding_mode: 'local' or 'hosted'. Uses config default if None.
        """
        config = get_config()

        metadata_db_path_val = metadata_db_path or config.metadata_db_path
        ocr_output_dir = Path(metadata_db_path_val).parent / "ocr_texts"

        self.loader = DocumentLoader(
            ocr_enabled=config.ocr_enabled,
            ocr_model=config.ocr_model,
            ocr_output_dir=ocr_output_dir,
        )
        self.chunker = DocumentChunker()
        self.embedder = Embedder(mode=embedding_mode or config.embedding_mode)
        self.vector_db = VectorStore(persist_directory=vector_db_path or config.vector_db_path)
        self.metadata_db = MetadataDB(db_path=metadata_db_path or config.metadata_db_path)

    def ingest_directory(
        self,
        directory: Path | str,
        recursive: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Ingest all supported documents from a directory.

        Args:
            directory: Directory to scan for documents.
            recursive: Whether to scan subdirectories.
            progress_callback: Optional callback(filepath, current, total).

        Returns:
            Dict with statistics about ingestion.
        """
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"

        all_files = [f for f in directory.glob(pattern) if f.is_file()]
        files = [f for f in all_files if DocumentLoader.is_supported(f)]
        unsupported = [f for f in all_files if not DocumentLoader.is_supported(f)]

        stats = {
            "total_files": len(files),
            "ingested": 0,
            "skipped_duplicate": 0,
            "duplicate_files": [],
            "unreadable": 0,
            "unreadable_files": [],
            "unsupported": len(unsupported),
            "unsupported_files": [str(f) for f in unsupported],
            "failed": 0,
            "errors": [],
            "ocr_files": [],
        }

        # Pre-scan a few text-extractable files to detect corpus language for OCR
        if self.loader._ocr_enabled:
            self.loader.ocr_language = self._detect_corpus_language(files)

        for i, filepath in enumerate(files):
            if progress_callback:
                progress_callback(str(filepath), i + 1, len(files))

            try:
                result = self.ingest_file(filepath)
                if result["status"] == "ingested":
                    stats["ingested"] += 1
                    if result.get("ocr_used"):
                        stats["ocr_files"].append(
                            {
                                "filepath": result["filepath"],
                                "ocr_text_path": result.get("ocr_text_path"),
                                "ocr_model": result.get("ocr_model"),
                            }
                        )
                elif result["status"] == "duplicate":
                    stats["skipped_duplicate"] += 1
                    stats["duplicate_files"].append(result["filepath"])
                elif result["status"] == "unreadable":
                    stats["unreadable"] += 1
                    stats["unreadable_files"].append(result["filepath"])
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"file": str(filepath), "error": str(e)})

        return stats

    def _detect_corpus_language(self, files: list[Path], sample_count: int = 5) -> str | None:
        """Detect the dominant language from a sample of text-extractable files."""
        text_extensions = {".txt", ".md"}
        sample_files = [f for f in files if f.suffix.lower() in text_extensions][:sample_count]
        if not sample_files:
            return None

        combined = []
        for f in sample_files:
            try:
                text = f.read_text(encoding="utf-8", errors="replace")[:500]
                if text.strip():
                    combined.append(text)
            except Exception:
                continue

        if not combined:
            return None

        try:
            lang_code = detect(" ".join(combined))
            lang_name = _LANG_NAMES.get(lang_code, lang_code)
            print(f"[kb] detected corpus language: {lang_name}")
            return lang_name
        except LangDetectException:
            return None

    def ingest_file(self, filepath: Path | str) -> dict[str, Any]:
        """Ingest a single document.

        Args:
            filepath: Path to the document.

        Returns:
            Dict with status and details.
        """
        filepath = Path(filepath)

        # Load document
        doc_data = self.loader.load(filepath)
        content = doc_data["content"]
        metadata = doc_data["metadata"]

        # Check for duplicates; use filepath as hash basis for empty content
        # to avoid false duplicate detection between unreadable documents
        # (e.g. image-only PDFs all produce the same empty-string hash)
        content_hash = compute_document_hash(content if content.strip() else str(filepath))
        if is_duplicate(content_hash, self.metadata_db):
            return {
                "status": "duplicate",
                "filepath": str(filepath),
                "content_hash": content_hash,
            }

        # Chunk document
        chunks = self.chunker.chunk(content)

        if not chunks:
            return {
                "status": "unreadable",
                "filepath": str(filepath),
                "chunks": 0,
            }

        # Add to metadata DB
        doc_id = self.metadata_db.add_document(
            filepath=str(filepath),
            content_hash=content_hash,
            file_type=filepath.suffix,
            metadata=metadata,
        )

        # Store chunks in metadata DB
        chunk_ids = self.metadata_db.add_chunks(doc_id, chunks)

        # Compute embeddings
        embeddings = self.embedder.embed(chunks)

        # Prepare metadata for vector DB
        embedding_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                "document_id": doc_id,
                "chunk_index": i,
                "source": str(filepath),
                "file_type": filepath.suffix,
            }
            for i in range(len(chunks))
        ]

        # Add to vector DB
        self.vector_db.add_embeddings(
            embeddings=embeddings,
            documents=chunks,
            metadatas=chunk_metadatas,
            ids=embedding_ids,
        )

        # Update chunk embedding IDs in metadata DB
        for chunk_id, embedding_id in zip(chunk_ids, embedding_ids):
            self.metadata_db.update_chunk_embedding_id(chunk_id, embedding_id)

        result: dict[str, Any] = {
            "status": "ingested",
            "filepath": str(filepath),
            "document_id": doc_id,
            "chunks": len(chunks),
        }
        if metadata.get("ocr_used"):
            result["ocr_used"] = True
            result["ocr_model"] = metadata.get("ocr_model")
            if "ocr_text_path" in metadata:
                result["ocr_text_path"] = metadata["ocr_text_path"]
        return result

    def get_stats(self) -> dict[str, Any]:
        """Get current ingestion statistics.

        Returns:
            Dict with document and chunk counts.
        """
        return {
            "documents": self.metadata_db.get_document_count(),
            "chunks": self.metadata_db.get_chunk_count(),
            "embeddings": self.vector_db.count(),
        }
