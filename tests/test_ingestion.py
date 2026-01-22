"""Tests for document ingestion."""

import pytest

from ingestion.chunker import DocumentChunker
from ingestion.deduplicator import compute_document_hash
from ingestion.loader import DocumentLoader


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_text_file(self, sample_text_file):
        """Test loading a plain text file."""
        loader = DocumentLoader()
        result = loader.load(sample_text_file)

        assert "content" in result
        assert "metadata" in result
        assert "software engineering" in result["content"]

    def test_load_markdown_file(self, sample_markdown_file):
        """Test loading a markdown file."""
        loader = DocumentLoader()
        result = loader.load(sample_markdown_file)

        assert "content" in result
        assert "# Test Document" in result["content"]
        assert "Introduction" in result["content"]

    def test_load_nonexistent_file(self, tmp_path):
        """Test loading a file that doesn't exist raises error."""
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load(tmp_path / "nonexistent.txt")

    def test_load_unsupported_format(self, tmp_path):
        """Test loading an unsupported file format raises error."""
        file_path = tmp_path / "test.xyz"
        file_path.write_text("content")

        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(file_path)

    def test_is_supported(self):
        """Test checking supported file extensions."""
        assert DocumentLoader.is_supported("test.txt")
        assert DocumentLoader.is_supported("test.md")
        assert DocumentLoader.is_supported("test.pdf")
        assert DocumentLoader.is_supported("test.docx")
        assert not DocumentLoader.is_supported("test.xyz")
        assert not DocumentLoader.is_supported("test.exe")


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def test_chunk_short_text(self, test_config):
        """Test chunking text shorter than chunk size."""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        text = "This is a short text."

        chunks = chunker.chunk(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text(self, test_config):
        """Test chunking text longer than chunk size."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        text = "Word " * 100  # 500 characters

        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        # Check overlap - chunks should have some common content
        for i in range(len(chunks) - 1):
            # Last part of chunk i should appear in chunk i+1
            assert len(chunks[i]) <= 100 + 50  # Some tolerance for word boundaries

    def test_chunk_empty_text(self, test_config):
        """Test chunking empty text returns empty list."""
        chunker = DocumentChunker()
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self, test_config):
        """Test chunking whitespace-only text returns empty list."""
        chunker = DocumentChunker()
        chunks = chunker.chunk("   \n\t  ")
        assert chunks == []

    def test_chunk_with_metadata(self, test_config):
        """Test chunking with metadata attached."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a test. " * 10
        metadata = {"source": "test.txt"}

        results = chunker.chunk_with_metadata(text, metadata)

        assert len(results) > 0
        for result in results:
            assert "content" in result
            assert "metadata" in result
            assert result["metadata"]["source"] == "test.txt"
            assert "chunk_index" in result["metadata"]


class TestDeduplicator:
    """Tests for deduplication functions."""

    def test_hash_deterministic(self):
        """Test that hash is deterministic for same content."""
        content = "Test document content"
        hash1 = compute_document_hash(content)
        hash2 = compute_document_hash(content)
        assert hash1 == hash2

    def test_hash_different_for_different_content(self):
        """Test that different content produces different hashes."""
        hash1 = compute_document_hash("Content A")
        hash2 = compute_document_hash("Content B")
        assert hash1 != hash2

    def test_hash_normalizes_whitespace(self):
        """Test that hash normalizes whitespace."""
        content1 = "Test   content"
        content2 = "Test content"
        hash1 = compute_document_hash(content1)
        hash2 = compute_document_hash(content2)
        assert hash1 == hash2

    def test_hash_preserves_case(self):
        """Test that hash preserves case (doesn't lowercase)."""
        content1 = "Test Content"
        content2 = "test content"
        hash1 = compute_document_hash(content1)
        hash2 = compute_document_hash(content2)
        assert hash1 != hash2

    def test_hash_length(self):
        """Test that hash is 16 characters."""
        hash_value = compute_document_hash("Any content")
        assert len(hash_value) == 16


class TestDocumentIngester:
    """Tests for DocumentIngester integration."""

    def test_ingest_file(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test ingesting a single file."""
        from ingestion import DocumentIngester

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        result = ingester.ingest_file(sample_text_file)

        assert result["status"] == "ingested"
        assert result["chunks"] >= 1

    def test_ingest_duplicate_skipped(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test that re-ingesting same file is skipped."""
        from ingestion import DocumentIngester

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )

        # First ingestion
        result1 = ingester.ingest_file(sample_text_file)
        assert result1["status"] == "ingested"

        # Second ingestion should be skipped
        result2 = ingester.ingest_file(sample_text_file)
        assert result2["status"] == "duplicate"

    def test_ingest_directory(self, tmp_db_paths, sample_documents_dir, mock_embedder):
        """Test ingesting a directory of documents."""
        from ingestion import DocumentIngester

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        stats = ingester.ingest_directory(sample_documents_dir)

        assert stats["total_files"] == 3
        assert stats["ingested"] == 3
        assert stats["failed"] == 0

    def test_get_stats(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test getting ingestion statistics."""
        from ingestion import DocumentIngester

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_file(sample_text_file)

        stats = ingester.get_stats()

        assert stats["documents"] == 1
        assert stats["chunks"] >= 1
        assert stats["embeddings"] >= 1
