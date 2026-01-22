"""RAG prompt construction."""

from typing import Any


def build_rag_prompt(query: str, context: str, language: str = "pl") -> str:
    """Build RAG prompt with query and retrieved context.

    Args:
        query: User's question.
        context: Retrieved and formatted context.
        language: Language for the prompt ('pl' or 'en').

    Returns:
        Formatted prompt string.
    """
    if language == "pl":
        no_answer_msg = "Nie mogę znaleźć odpowiedzi w dostępnych dokumentach."
        return f"""Odpowiedz na pytanie używając wyłącznie informacji z poniższych dokumentów.
Jeśli odpowiedź nie znajduje się w dokumentach, powiedz "{no_answer_msg}"

DOKUMENTY:
{context}

PYTANIE: {query}

ODPOWIEDŹ:"""
    else:
        return f"""Answer the question using only information from the documents below.
If the answer is not in the documents, say "I cannot find the answer in the available documents."

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER:"""


def format_context(chunks: list[dict[str, Any]], include_source: bool = True) -> str:
    """Format retrieved chunks into context string.

    Args:
        chunks: List of chunk dicts with 'content' and optionally 'source'.
        include_source: Whether to include source information.

    Returns:
        Formatted context string.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        if include_source and "source" in chunk:
            parts.append(f"[{i}] (Source: {chunk['source']})\n{chunk['content']}")
        else:
            parts.append(f"[{i}] {chunk['content']}")

    return "\n\n".join(parts)


def build_system_prompt(language: str = "pl") -> str:
    """Build system prompt for chat-style LLMs.

    Args:
        language: Language for the prompt ('pl' or 'en').

    Returns:
        System prompt string.
    """
    if language == "pl":
        return (
            "Jesteś pomocnym asystentem, który odpowiada na pytania "
            "wyłącznie na podstawie dostarczonych dokumentów.\n"
            "Jeśli nie możesz znaleźć odpowiedzi w dokumentach, przyznaj to wprost.\n"
            "Cytuj źródła gdy to możliwe."
        )
    else:
        return (
            "You are a helpful assistant that answers questions "
            "based solely on the provided documents.\n"
            "If you cannot find the answer in the documents, admit it directly.\n"
            "Cite sources when possible."
        )
