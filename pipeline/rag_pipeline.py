"""Main RAG pipeline orchestration."""

from collections.abc import Generator
from typing import Any

from config import get_config
from generation.llm_client import LLMClient
from generation.prompt_builder import build_rag_prompt, format_context
from ingestion.embedder import Embedder
from pipeline.graph import RAGComponents, RAGState, build_rag_graph
from retrieval.reranker import NoOpReranker, Reranker
from retrieval.retriever import Retriever
from storage.vector_db import VectorStore


class RAGPipeline:
    """High-level RAG pipeline interface."""

    def __init__(
        self,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        embedding_mode: str | None = None,
        use_reranker: bool | None = None,
    ):
        """Initialize RAG pipeline.

        Args:
            llm_provider: 'ollama' or 'openrouter'. Uses config if None.
            llm_model: Model name. Uses config if None.
            embedding_mode: 'local' or 'hosted'. Uses config if None.
            use_reranker: Whether to use reranker. Uses config if None.
        """
        config = get_config()

        self.llm_provider = llm_provider or config.llm_provider
        self.llm_model = llm_model or config.llm_model
        self.embedding_mode = embedding_mode or config.embedding_mode
        self.use_reranker = use_reranker if use_reranker is not None else config.use_reranker

        # Initialize components
        self._embedder = Embedder(mode=self.embedding_mode)
        self._vector_store = VectorStore()
        self._retriever = Retriever(embedder=self._embedder, vector_store=self._vector_store)
        self._reranker = Reranker() if self.use_reranker else NoOpReranker()
        self._llm_client = LLMClient(provider=self.llm_provider, model=self.llm_model)

        # Build graph with components
        self._components = RAGComponents(
            retriever=self._retriever,
            reranker=self._reranker,
            llm_client=self._llm_client,
        )
        self._graph = build_rag_graph(self._components)

    def query(
        self,
        question: str,
        top_k: int | None = None,
        return_sources: bool = True,
    ) -> dict[str, Any]:
        """Execute RAG query.

        Args:
            question: User's question.
            top_k: Number of chunks to retrieve. Uses config if None.
            return_sources: Whether to include sources in response.

        Returns:
            Dict with 'answer' and optionally 'sources' and 'chunks'.
        """
        config = get_config()
        top_k = top_k or config.top_k

        initial_state: RAGState = {
            "query": question,
            "top_k": top_k,
            "use_reranker": self.use_reranker,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "context": "",
            "response": "",
            "sources": [],
        }

        # Run through LangGraph
        final_state = self._graph.invoke(initial_state)

        result = {"answer": final_state["response"]}

        if return_sources:
            result["sources"] = final_state["sources"]
            result["chunks"] = final_state["reranked_chunks"]

        return result

    def query_stream(
        self,
        question: str,
        top_k: int | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Execute RAG query with streaming response.

        Args:
            question: User's question.
            top_k: Number of chunks to retrieve.

        Yields:
            Dicts with 'type' key ('chunk' for text, 'sources' for final sources).
        """
        config = get_config()
        top_k = top_k or config.top_k

        # Retrieve chunks
        chunks = self._retriever.retrieve(query=question, top_k=top_k)

        # Rerank if enabled
        if self.use_reranker:
            chunks = self._reranker.rerank(query=question, chunks=chunks, top_k=top_k)

        # Format context and build prompt
        context = format_context(chunks, include_source=True)
        prompt = build_rag_prompt(question, context)

        # Stream response
        for text_chunk in self._llm_client.generate_stream(prompt):
            yield {"type": "chunk", "content": text_chunk}

        # Send sources at the end
        sources = [chunk["source"] for chunk in chunks]
        yield {"type": "sources", "sources": sources, "chunks": chunks}

    def retrieve_only(
        self,
        question: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve without generation (useful for debugging).

        Args:
            question: Query text.
            top_k: Number of results.

        Returns:
            List of retrieved chunks.
        """
        config = get_config()
        top_k = top_k or config.top_k

        chunks = self._retriever.retrieve(query=question, top_k=top_k)

        if self.use_reranker:
            chunks = self._reranker.rerank(query=question, chunks=chunks, top_k=top_k)

        return chunks

    def close(self) -> None:
        """Clean up resources."""
        self._llm_client.close()
