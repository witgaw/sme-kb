"""Tests for generation functionality."""

import pytest

from generation.prompt_builder import (
    build_rag_prompt,
    build_system_prompt,
    format_context,
)


class TestPromptBuilder:
    """Tests for prompt building functions."""

    def test_build_rag_prompt_polish(self):
        """Test building RAG prompt in Polish."""
        query = "Kiedy odnowienie umowy?"
        context = "[1] Umowa odnawia sie 15 marca."

        prompt = build_rag_prompt(query, context, language="pl")

        assert query in prompt
        assert context in prompt
        assert "DOKUMENTY:" in prompt
        assert "PYTANIE:" in prompt
        assert "ODPOWIEDŹ:" in prompt

    def test_build_rag_prompt_english(self):
        """Test building RAG prompt in English."""
        query = "When is the contract renewal?"
        context = "[1] The contract renews on March 15."

        prompt = build_rag_prompt(query, context, language="en")

        assert query in prompt
        assert context in prompt
        assert "DOCUMENTS:" in prompt
        assert "QUESTION:" in prompt
        assert "ANSWER:" in prompt

    def test_format_context_with_sources(self):
        """Test formatting context with source information."""
        chunks = [
            {"content": "First chunk content", "source": "doc1.txt"},
            {"content": "Second chunk content", "source": "doc2.txt"},
        ]

        context = format_context(chunks, include_source=True)

        assert "[1]" in context
        assert "[2]" in context
        assert "doc1.txt" in context
        assert "doc2.txt" in context
        assert "First chunk content" in context
        assert "Second chunk content" in context

    def test_format_context_without_sources(self):
        """Test formatting context without source information."""
        chunks = [
            {"content": "First chunk content", "source": "doc1.txt"},
            {"content": "Second chunk content", "source": "doc2.txt"},
        ]

        context = format_context(chunks, include_source=False)

        assert "[1]" in context
        assert "[2]" in context
        assert "doc1.txt" not in context
        assert "First chunk content" in context

    def test_format_context_empty(self):
        """Test formatting empty chunk list."""
        context = format_context([])
        assert context == ""

    def test_build_system_prompt_polish(self):
        """Test building system prompt in Polish."""
        prompt = build_system_prompt(language="pl")
        assert "pomocnym asystentem" in prompt

    def test_build_system_prompt_english(self):
        """Test building system prompt in English."""
        prompt = build_system_prompt(language="en")
        assert "helpful assistant" in prompt


class TestLLMClient:
    """Tests for LLM client."""

    def test_ollama_client_initialization(self, test_config):
        """Test Ollama client initialization."""
        from generation.llm_client import LLMClient

        client = LLMClient(provider="ollama", model="llama3.1:8b")

        assert client.provider == "ollama"
        assert client.model == "llama3.1:8b"
        assert client.base_url == "http://localhost:11434"

    def test_openrouter_requires_api_key(self, test_config):
        """Test OpenRouter client requires API key."""
        # Clear any existing API key
        import os

        from generation.llm_client import LLMClient

        original_key = os.environ.pop("OPENROUTER_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY required"):
                LLMClient(provider="openrouter", model="test")
        finally:
            if original_key:
                os.environ["OPENROUTER_API_KEY"] = original_key

    def test_generate_ollama(self, test_config, mock_llm_client):
        """Test generation with Ollama (mocked)."""
        from generation.llm_client import LLMClient

        client = LLMClient(provider="ollama", model="llama3.1:8b")
        response = client.generate("Test prompt")

        assert response == "Test response from LLM"
        mock_llm_client.post.assert_called_once()
