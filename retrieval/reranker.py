"""Optional reranking using cross-encoder models."""

from typing import Any


class Reranker:
    """Rerank retrieved chunks using cross-encoder.

    This is an optional component that can improve retrieval quality
    by using a more expensive cross-encoder model to rerank results
    from the cheaper bi-encoder retrieval.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize reranker.

        Args:
            model_name: Cross-encoder model to use.
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank chunks by relevance to query.

        Args:
            query: Query text.
            chunks: List of chunk dicts with 'content' key.
            top_k: Number of top results to return. Returns all if None.

        Returns:
            Reranked list of chunks with added 'rerank_score' key.
        """
        if not chunks:
            return []

        self._load_model()

        # Create query-document pairs
        pairs = [[query, chunk["content"]] for chunk in chunks]

        # Get scores
        scores = self._model.predict(pairs)

        # Add scores to chunks and sort
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked


class NoOpReranker:
    """Pass-through reranker that doesn't change ordering.

    Useful as a default when reranking is disabled.
    """

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return chunks unchanged (optionally truncated).

        Args:
            query: Query text (ignored).
            chunks: List of chunks.
            top_k: Number to return. Returns all if None.

        Returns:
            Chunks unchanged.
        """
        if top_k is not None:
            return chunks[:top_k]
        return chunks
