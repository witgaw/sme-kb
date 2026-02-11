"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

import pytest

from config import RAGConfig, reset_config, set_config


@pytest.fixture(autouse=True)
def reset_global_config():
    """Reset global config before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_db_paths(tmp_path):
    """Create temporary database paths."""
    return {
        "vector_db_path": str(tmp_path / "chroma_db"),
        "metadata_db_path": str(tmp_path / "metadata.db"),
    }


@pytest.fixture
def test_config(tmp_db_paths):
    """Create test configuration with temporary paths."""
    config = RAGConfig(
        vector_db_path=tmp_db_paths["vector_db_path"],
        metadata_db_path=tmp_db_paths["metadata_db_path"],
        embedding_mode="local",
        llm_provider="ollama",
        llm_model="llama3.1:8b",
    )
    set_config(config)
    return config


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing."""
    content = (
        "This is a test document about software engineering.\n"
        "It contains multiple sentences for chunking."
    )
    file_path = tmp_path / "test_doc.txt"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path):
    """Create a sample markdown file for testing."""
    content = """# Test Document

## Introduction

This is a test markdown document with multiple sections.

## Details

Here are some important details about the topic.
The document discusses various aspects of the subject matter.

## Conclusion

This concludes the test document.
"""
    file_path = tmp_path / "test_doc.md"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_documents_dir(tmp_path):
    """Create a directory with multiple sample documents."""
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    # Create text file
    (docs_dir / "doc1.txt").write_text("Odnowienie umowy z Klientem B przypada na 15 marca 2024.")

    # Create another text file
    (docs_dir / "doc2.txt").write_text("Przychod w Q3 wyniosl 287,500 PLN.")

    # Create markdown file
    (docs_dir / "doc3.md").write_text(
        """# Raport kwartalny

## Podsumowanie

Firma osiagnela dobre wyniki w tym kwartale.
"""
    )

    return docs_dir


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic embeddings."""
    import numpy as np

    with patch("ingestion.embedder.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        # Return simple deterministic embeddings as numpy array (has .tolist())
        # Use 768 dimensions to match nomic-embed-text-v1.5
        mock_model.encode.side_effect = lambda texts, **kwargs: np.array(
            [[float(len(t) % 10) / 10] * 768 for t in texts]
        )
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    with patch("generation.llm_client.httpx.Client") as mock_http:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": "Test response from LLM",
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        mock_client.post.return_value = mock_response
        mock_http.return_value = mock_client
        yield mock_client
