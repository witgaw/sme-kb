"""Query retrieval from vector store."""

from typing import Any

from config import get_config
from ingestion.embedder import Embedder
from storage.vector_db import VectorStore


class Retriever:
    """Retrieve relevant chunks for queries."""

    def __init__(
        self,
        vector_db_path: str | None = None,
        embedding_mode: str | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
    ):
        """Initialize retriever.

        Args:
            vector_db_path: Path to ChromaDB storage. Uses config default if None.
            embedding_mode: 'local' or 'hosted'. Uses config default if None.
            embedder: Optional pre-configured Embedder instance.
            vector_store: Optional pre-configured VectorStore instance.
        """
        config = get_config()

        if vector_store is not None:
            self.vector_db = vector_store
        else:
            self.vector_db = VectorStore(persist_directory=vector_db_path or config.vector_db_path)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = Embedder(mode=embedding_mode or config.embedding_mode)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query text.
            top_k: Number of results to return. Uses config default if None.
            filter_dict: Optional metadata filter.

        Returns:
            List of chunk dicts with id, content, metadata, distance, source.
        """
        config = get_config()
        top_k = top_k or config.top_k

        # Embed query
        query_embedding = self.embedder.embed_single(query)

        # Query vector DB
        results = self.vector_db.query(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        # Format results
        chunks = []
        if results["ids"] and results["ids"][0]:
            distances = results.get("distances")
            for i in range(len(results["ids"][0])):
                chunks.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": distances[0][i] if distances else None,
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                    }
                )

        return chunks

    def retrieve_by_filter(
        self,
        filter_dict: dict[str, Any],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve chunks matching a metadata filter without query.

        Uses a zero vector to retrieve by metadata only.

        Args:
            filter_dict: Metadata filter.
            top_k: Number of results to return.

        Returns:
            List of matching chunks.
        """
        config = get_config()
        top_k = top_k or config.top_k

        # Use zero vector - filter is what matters
        zero_embedding = [0.0] * self.embedder.embedding_dimension

        results = self.vector_db.query(
            query_embedding=zero_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                chunks.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "source": results["metadatas"][0][i].get("source", "Unknown"),
                    }
                )

        return chunks
