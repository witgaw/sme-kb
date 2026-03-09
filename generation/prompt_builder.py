"""RAG prompt construction."""

import importlib.resources
import json
from typing import Any


def _load_dataset_prompts() -> tuple[dict, dict]:
    """Load system prompts and negative-answer strings from the sme-synth-data-gen dataset."""
    text = importlib.resources.files("dataset").joinpath("system_prompt.json").read_text()
    data = json.loads(text)
    system_prompts = {lang: data[lang]["system_prompt"] for lang in ("pl", "en") if lang in data}
    negative_formats = {
        lang: data[lang]["negative_answer_format"] for lang in ("pl", "en") if lang in data
    }
    return system_prompts, negative_formats


_SYSTEM_PROMPTS, _NEGATIVE_FORMATS = _load_dataset_prompts()


def build_rag_prompt(query: str, context: str, language: str = "pl") -> str:
    """Build RAG prompt with query and retrieved context.

    Args:
        query: User's question.
        context: Retrieved and formatted context.
        language: Language for the prompt ('pl' or 'en').

    Returns:
        Formatted prompt string.
    """
    no_answer_msg = _NEGATIVE_FORMATS[language]
    if language == "pl":
        return f"""Odpowiedz na pytanie używając wyłącznie informacji z poniższych dokumentów.
Jeśli odpowiedź nie znajduje się w dokumentach, powiedz "{no_answer_msg}"

DOKUMENTY:
{context}

PYTANIE: {query}

ODPOWIEDŹ:"""
    else:
        return f"""Answer the question using only information from the documents below.
If the answer is not in the documents, say "{no_answer_msg}"

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

    Loaded from the sme-synth-data-gen dataset package (dataset/system_prompt.json)
    so that negative and partial-answer phrasing always matches what the scorer expects.

    Args:
        language: Language for the prompt ('pl' or 'en').

    Returns:
        System prompt string.
    """
    return _SYSTEM_PROMPTS[language]


def build_rag_prompt_with_history(
    query: str,
    context: str,
    conversation_history: list[dict[str, str]],
    language: str = "pl",
) -> list[dict[str, str]]:
    """Build RAG prompt with conversation history as message array.

    This function creates a message array suitable for chat APIs, including:
    - System message with instructions
    - Previous conversation turns (clean Q&A pairs, without RAG chunks)
    - Current query with fresh RAG chunks prepended

    Args:
        query: Current user's question.
        context: Retrieved and formatted context for current query.
        conversation_history: Previous conversation turns as message dicts.
        language: Language for the prompt ('pl' or 'en').

    Returns:
        List of message dicts with 'role' and 'content' keys, suitable for chat APIs.
    """
    messages = []

    # Add system message
    system_prompt = build_system_prompt(language)
    messages.append({"role": "system", "content": system_prompt})

    # Add conversation history (clean Q&A pairs, no RAG chunks)
    messages.extend(conversation_history)

    # Build current query with RAG context
    no_answer_msg = _NEGATIVE_FORMATS[language]
    if language == "pl":
        instruction = "Odpowiedz na pytanie używając wyłącznie informacji z poniższych dokumentów."
        current_message = f"""{instruction}
Jeśli odpowiedź nie znajduje się w dokumentach, powiedz "{no_answer_msg}"

DOKUMENTY:
{context}

PYTANIE: {query}"""
    else:
        instruction = "Answer the question using only information from the documents below."
        current_message = f"""{instruction}
If the answer is not in the documents, say "{no_answer_msg}"

DOCUMENTS:
{context}

QUESTION: {query}"""

    messages.append({"role": "user", "content": current_message})

    return messages
