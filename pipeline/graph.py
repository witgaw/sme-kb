"""LangGraph workflow definition for RAG pipeline."""

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from config import get_config
from generation.llm_client import LLMClient
from generation.prompt_builder import (
    build_rag_prompt,
    build_rag_prompt_with_history,
    format_context,
)
from ingestion.embedder import Embedder
from retrieval.reranker import NoOpReranker, Reranker
from retrieval.retriever import Retriever
from storage.vector_db import VectorStore


class RAGState(TypedDict):
    """State for RAG workflow."""

    query: str
    top_k: int
    use_reranker: bool
    retrieved_chunks: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]
    context: str
    response: str
    sources: list[str]
    prompt: str
    usage: dict[str, int]
    conversation_history: NotRequired[list[dict[str, str]]]
    language: str


class RAGComponents:
    """Container for RAG pipeline components (dependency injection)."""

    def __init__(
        self,
        retriever: Retriever | None = None,
        reranker: Reranker | NoOpReranker | None = None,
        llm_client: LLMClient | None = None,
    ):
        """Initialize components.

        Args:
            retriever: Retriever instance. Creates default if None.
            reranker: Reranker instance. Creates based on config if None.
            llm_client: LLM client instance. Creates default if None.
        """
        config = get_config()

        if retriever is not None:
            self.retriever = retriever
        else:
            embedder = Embedder()
            vector_store = VectorStore()
            self.retriever = Retriever(embedder=embedder, vector_store=vector_store)

        if reranker is not None:
            self.reranker = reranker
        elif config.use_reranker:
            self.reranker = Reranker()
        else:
            self.reranker = NoOpReranker()

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            self.llm_client = LLMClient()


def create_retrieve_node(components: RAGComponents):
    """Create retrieval node function.

    Args:
        components: RAG components container.

    Returns:
        Node function for retrieval.
    """

    def retrieve_node(state: RAGState) -> dict[str, Any]:
        """Retrieve relevant chunks from vector DB."""
        chunks = components.retriever.retrieve(
            query=state["query"],
            top_k=state["top_k"],
        )
        return {"retrieved_chunks": chunks}

    return retrieve_node


def create_rerank_node(components: RAGComponents):
    """Create reranking node function.

    Args:
        components: RAG components container.

    Returns:
        Node function for reranking.
    """

    def rerank_node(state: RAGState) -> dict[str, Any]:
        """Rerank retrieved chunks."""
        if state["use_reranker"]:
            reranked = components.reranker.rerank(
                query=state["query"],
                chunks=state["retrieved_chunks"],
                top_k=state["top_k"],
            )
        else:
            reranked = state["retrieved_chunks"]

        return {"reranked_chunks": reranked}

    return rerank_node


def create_generate_node(components: RAGComponents):
    """Create generation node function.

    Args:
        components: RAG components container.

    Returns:
        Node function for generation.
    """

    def generate_node(state: RAGState) -> dict[str, Any]:
        """Generate response using LLM."""
        # Format context
        context = format_context(state["reranked_chunks"], include_source=True)

        # Get language preference
        language = state.get("language", "pl")

        # Check if conversation history is present
        conversation_history = state.get("conversation_history")

        if conversation_history:
            # Use history-aware prompting
            messages = build_rag_prompt_with_history(
                query=state["query"],
                context=context,
                conversation_history=conversation_history,
                language=language,
            )
            response, usage_obj = components.llm_client.generate_with_history(messages)
            # For prompt details, store the full message array as a formatted string
            prompt = "\n\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in messages)
        else:
            # Use single-turn prompting (existing behavior)
            prompt = build_rag_prompt(state["query"], context, language=language)
            response, usage_obj = components.llm_client.generate(prompt)

        # Convert usage to dict
        usage = {
            "prompt_tokens": usage_obj.prompt_tokens,
            "completion_tokens": usage_obj.completion_tokens,
            "total_tokens": usage_obj.total_tokens,
        }

        # Extract sources
        sources = [chunk["source"] for chunk in state["reranked_chunks"]]

        return {
            "context": context,
            "response": response,
            "sources": sources,
            "prompt": prompt,
            "usage": usage,
        }

    return generate_node


def build_rag_graph(components: RAGComponents | None = None) -> StateGraph:
    """Build the RAG workflow graph.

    Args:
        components: Optional pre-configured components.

    Returns:
        Compiled LangGraph StateGraph.
    """
    if components is None:
        components = RAGComponents()

    # Create node functions with injected components
    retrieve_node = create_retrieve_node(components)
    rerank_node = create_rerank_node(components)
    generate_node = create_generate_node(components)

    # Build graph
    workflow = StateGraph(RAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
