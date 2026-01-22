"""ChromaDB vector store implementation."""

from typing import Any

import chromadb

from config import get_config


class VectorStore:
    """Vector store using ChromaDB with PersistentClient."""

    def __init__(self, persist_directory: str | None = None):
        """Initialize vector store.

        Args:
            persist_directory: Path to persist ChromaDB data. If None, uses config default.
        """
        if persist_directory is None:
            persist_directory = get_config().vector_db_path

        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_embeddings(
        self,
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        """Add embeddings to the collection.

        Args:
            embeddings: List of embedding vectors.
            documents: List of document texts.
            metadatas: List of metadata dicts.
            ids: List of unique IDs.
        """
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Query the collection for similar embeddings.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            filter_dict: Optional metadata filter.

        Returns:
            Dict with ids, documents, metadatas, and distances.
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
        )
        return results

    def count(self) -> int:
        """Get total number of embeddings in collection."""
        return self.collection.count()

    def delete_collection(self) -> None:
        """Delete the collection (useful for testing)."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )
