"""SQLite metadata storage with thread-safe connection handling."""

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config import get_config


class MetadataDB:
    """SQLite database for document metadata with thread-safe connections."""

    def __init__(self, db_path: str | None = None):
        """Initialize metadata database.

        Args:
            db_path: Path to SQLite database file. If None, uses config default.
        """
        if db_path is None:
            db_path = get_config().metadata_db_path

        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._transaction() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT NOT NULL,
                    content_hash TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_id TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents(id)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_hash
                ON documents(content_hash)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id)
            """)

    def hash_exists(self, content_hash: str) -> bool:
        """Check if document with given hash already exists.

        Args:
            content_hash: Content hash to check.

        Returns:
            True if hash exists, False otherwise.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT 1 FROM documents WHERE content_hash = ?",
                (content_hash,),
            )
            return cursor.fetchone() is not None

    def add_document(
        self,
        filepath: str,
        content_hash: str,
        file_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a document to the database.

        Args:
            filepath: Path to the document file.
            content_hash: Hash of document content.
            file_type: File extension/type.
            metadata: Optional document metadata.

        Returns:
            ID of the inserted document.
        """
        with self._transaction() as cursor:
            cursor.execute(
                """INSERT INTO documents (filepath, content_hash, file_type, metadata)
                   VALUES (?, ?, ?, ?)""",
                (filepath, content_hash, file_type, json.dumps(metadata or {})),
            )
            return cursor.lastrowid or 0

    def add_chunks(self, document_id: int, chunks: list[str]) -> list[int]:
        """Add chunks for a document.

        Args:
            document_id: ID of parent document.
            chunks: List of chunk texts.

        Returns:
            List of chunk IDs.
        """
        chunk_ids = []
        with self._transaction() as cursor:
            for i, chunk in enumerate(chunks):
                cursor.execute(
                    """INSERT INTO chunks (document_id, chunk_index, content)
                       VALUES (?, ?, ?)""",
                    (document_id, i, chunk),
                )
                chunk_ids.append(cursor.lastrowid or 0)
        return chunk_ids

    def update_chunk_embedding_id(self, chunk_id: int, embedding_id: str) -> None:
        """Update the embedding ID for a chunk.

        Args:
            chunk_id: ID of the chunk.
            embedding_id: ID in the vector store.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "UPDATE chunks SET embedding_id = ? WHERE id = ?",
                (embedding_id, chunk_id),
            )

    def get_document(self, document_id: int) -> dict[str, Any] | None:
        """Get document by ID.

        Args:
            document_id: Document ID.

        Returns:
            Document dict or None if not found.
        """
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM documents WHERE id = ?",
                (document_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Get all documents.

        Returns:
            List of document dicts.
        """
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM documents ORDER BY ingested_at DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_document_count(self) -> int:
        """Get total number of documents."""
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM documents")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        with self._transaction() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM chunks")
            row = cursor.fetchone()
            return row["count"] if row else 0

    def delete_document(self, document_id: int) -> None:
        """Delete document and its chunks.

        Args:
            document_id: Document ID to delete.
        """
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def clear_all(self) -> None:
        """Clear all data from database (useful for testing)."""
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM chunks")
            cursor.execute("DELETE FROM documents")

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    @classmethod
    def from_path(cls, path: str | Path) -> "MetadataDB":
        """Create MetadataDB from a path.

        Args:
            path: Path to database file.

        Returns:
            MetadataDB instance.
        """
        return cls(db_path=str(path))
