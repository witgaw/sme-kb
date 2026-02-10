"""Document ingestion orchestration."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import get_config
from ingestion.chunker import DocumentChunker
from ingestion.deduplicator import compute_document_hash, is_duplicate
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader
from storage.metadata_db import MetadataDB
from storage.vector_db import VectorStore


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

        self.loader = DocumentLoader()
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

        files = [
            f for f in directory.glob(pattern) if f.is_file() and DocumentLoader.is_supported(f)
        ]

        stats = {
            "total_files": len(files),
            "ingested": 0,
            "skipped_duplicate": 0,
            "failed": 0,
            "errors": [],
        }

        for i, filepath in enumerate(files):
            if progress_callback:
                progress_callback(str(filepath), i + 1, len(files))

            try:
                result = self.ingest_file(filepath)
                if result["status"] == "ingested":
                    stats["ingested"] += 1
                elif result["status"] == "duplicate":
                    stats["skipped_duplicate"] += 1
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append({"file": str(filepath), "error": str(e)})

        return stats

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

        # Add to metadata DB
        doc_id = self.metadata_db.add_document(
            filepath=str(filepath),
            content_hash=content_hash,
            file_type=filepath.suffix,
            metadata=metadata,
        )

        # Chunk document
        chunks = self.chunker.chunk(content)

        if not chunks:
            return {
                "status": "ingested",
                "filepath": str(filepath),
                "document_id": doc_id,
                "chunks": 0,
            }

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

        return {
            "status": "ingested",
            "filepath": str(filepath),
            "document_id": doc_id,
            "chunks": len(chunks),
        }

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
