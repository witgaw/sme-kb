"""Tests for document ingestion."""

from unittest.mock import patch

import fitz
import pytest

from config import RAGConfig, set_config
from ingestion.chunker import DocumentChunker
from ingestion.deduplicator import compute_document_hash
from ingestion.loader import DocumentLoader
from ingestion.ocr import _build_ocr_prompt, _is_ocr_failure


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

    def test_ingest_empty_file_returns_unreadable(self, tmp_db_paths, tmp_path, mock_embedder):
        """Test that a file with no extractable text returns 'unreadable' status."""
        from ingestion import DocumentIngester

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n\t  ")  # whitespace only — produces 0 chunks

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        result = ingester.ingest_file(empty_file)

        assert result["status"] == "unreadable"
        assert result["chunks"] == 0
        # Must NOT be stored in the metadata DB
        assert ingester.metadata_db.get_document_count() == 0

    def test_unreadable_file_not_counted_in_stats(self, tmp_db_paths, tmp_path, mock_embedder):
        """Unreadable files increment stats['unreadable'] and are excluded from 'ingested'."""
        from ingestion import DocumentIngester

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   ")

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        stats = ingester.ingest_directory(tmp_path)

        assert stats["ingested"] == 0
        assert stats["unreadable"] == 1
        assert str(empty_file) in stats["unreadable_files"]
        assert stats["failed"] == 0

    def test_unsupported_files_tracked_in_stats(self, tmp_db_paths, tmp_path, mock_embedder):
        """Unsupported file types are counted separately and excluded from total_files."""
        from ingestion import DocumentIngester

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc.txt").write_text("Some readable content here.")
        (docs_dir / "schema.db").write_bytes(b"SQLite format 3\x00")
        (docs_dir / "photo.jpg").write_bytes(b"\xff\xd8\xff")

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        stats = ingester.ingest_directory(docs_dir)

        assert stats["total_files"] == 1  # only .txt counted
        assert stats["ingested"] == 1
        assert stats["unsupported"] == 2
        assert any("schema.db" in f for f in stats["unsupported_files"])
        assert any("photo.jpg" in f for f in stats["unsupported_files"])

    def test_unreadable_file_not_a_duplicate(self, tmp_db_paths, tmp_path, mock_embedder):
        """Re-ingesting an unreadable file is unreadable again, not duplicate."""
        from ingestion import DocumentIngester

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   ")

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        result1 = ingester.ingest_file(empty_file)
        result2 = ingester.ingest_file(empty_file)

        assert result1["status"] == "unreadable"
        assert result2["status"] == "unreadable"


def _make_image_only_pdf(filepath):
    """Create a minimal PDF with an image and no extractable text."""
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # Draw a filled rectangle to simulate an image-only page
    page.draw_rect(fitz.Rect(10, 10, 190, 190), color=(0, 0, 0), fill=(0.5, 0.5, 0.5))
    doc.save(str(filepath))
    doc.close()


def _make_text_pdf(filepath, text="Hello, this is a text-heavy PDF with plenty of content."):
    """Create a PDF with real extractable text."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(filepath))
    doc.close()


class TestOCRFallback:
    """Tests for the OCR fallback in _load_pdf."""

    def test_text_pdf_does_not_trigger_ocr(self, tmp_path):
        """A text PDF should NOT trigger OCR, and ocr_used should be False."""
        pdf_path = tmp_path / "text.pdf"
        _make_text_pdf(pdf_path)

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b")
        with patch("ingestion.loader.ocr_pdf_pages") as mock_ocr:
            result = loader.load(pdf_path)

        mock_ocr.assert_not_called()
        assert result["metadata"]["ocr_used"] is False
        assert len(result["content"].strip()) > 0

    def test_image_pdf_with_ocr_enabled_calls_ocr(self, tmp_path):
        """An image-only PDF with ocr_enabled=True should call the OCR path."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b")
        with patch("ingestion.loader.ocr_pdf_pages", return_value="OCR extracted text") as mock_ocr:
            result = loader.load(pdf_path)

        args, kwargs = mock_ocr.call_args
        assert args[0] == pdf_path
        assert kwargs["model"] == "llava:7b"
        assert kwargs["provider"] == "ollama"
        assert kwargs["base_url"] == "http://localhost:11434"
        assert kwargs["language"] is None
        assert result["content"] == "OCR extracted text"
        assert result["metadata"]["ocr_used"] is True

    def test_image_pdf_with_ocr_disabled_stays_empty(self, tmp_path):
        """An image-only PDF with ocr_enabled=False should NOT trigger OCR (default behavior)."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        loader = DocumentLoader(ocr_enabled=False)
        with patch("ingestion.loader.ocr_pdf_pages") as mock_ocr:
            result = loader.load(pdf_path)

        mock_ocr.assert_not_called()
        assert result["metadata"]["ocr_used"] is False
        assert result["content"].strip() == ""

    def test_sparse_text_pdf_triggers_ocr(self, tmp_path):
        """A PDF with very little text per page should still trigger OCR."""
        pdf_path = tmp_path / "sparse.pdf"
        # Insert only a few characters — below the threshold
        _make_text_pdf(pdf_path, text="Hi")

        loader = DocumentLoader(ocr_enabled=True, ocr_model="moondream")
        with patch("ingestion.loader.ocr_pdf_pages", return_value="Full OCR output") as mock_ocr:
            result = loader.load(pdf_path)

        mock_ocr.assert_called_once()
        assert result["metadata"]["ocr_used"] is True
        assert result["content"] == "Full OCR output"

    def test_ocr_failure_falls_back_to_original_content(self, tmp_path):
        """If OCR raises an exception, the original (sparse) content is preserved."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b")
        with patch("ingestion.loader.ocr_pdf_pages", side_effect=RuntimeError("Ollama down")):
            result = loader.load(pdf_path)

        assert result["metadata"]["ocr_used"] is False

    def test_ocr_returns_empty_keeps_original(self, tmp_path):
        """If OCR returns empty text, the original content is kept and ocr_used stays False."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b")
        with patch("ingestion.loader.ocr_pdf_pages", return_value="   "):
            result = loader.load(pdf_path)

        assert result["metadata"]["ocr_used"] is False

    def test_default_ocr_disabled_from_config(self, tmp_path):
        """DocumentLoader with no explicit OCR params defaults to config (disabled)."""
        config = RAGConfig(OCR_ENABLED=False, OCR_MODEL="test-model")
        set_config(config)

        loader = DocumentLoader()
        assert loader._ocr_enabled is False
        assert loader._ocr_model == "test-model"

    def test_ocr_cache_hit_skips_model_call(self, tmp_path):
        """If a cached OCR text file exists, skip the vision model entirely."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        ocr_dir = tmp_path / "ocr_texts"
        ocr_dir.mkdir()
        cached = ocr_dir / "scan.txt"
        cached.write_text("Cached OCR content", encoding="utf-8")

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b", ocr_output_dir=ocr_dir)
        with patch("ingestion.loader.ocr_pdf_pages") as mock_ocr:
            result = loader.load(pdf_path)

        mock_ocr.assert_not_called()
        assert result["content"] == "Cached OCR content"
        assert result["metadata"]["ocr_used"] is True
        assert result["metadata"]["ocr_text_path"] == str(cached)

    def test_ocr_cache_miss_calls_model_and_writes_cache(self, tmp_path):
        """If no cached file exists, run OCR and write the result to cache."""
        pdf_path = tmp_path / "scan.pdf"
        _make_image_only_pdf(pdf_path)

        ocr_dir = tmp_path / "ocr_texts"

        loader = DocumentLoader(ocr_enabled=True, ocr_model="llava:7b", ocr_output_dir=ocr_dir)
        with patch("ingestion.loader.ocr_pdf_pages", return_value="Fresh OCR output") as mock_ocr:
            result = loader.load(pdf_path)

        mock_ocr.assert_called_once()
        assert result["content"] == "Fresh OCR output"
        assert result["metadata"]["ocr_used"] is True
        # Cache file should have been written
        assert (ocr_dir / "scan.txt").read_text(encoding="utf-8") == "Fresh OCR output"


class TestOCRFailureDetection:
    """Tests for _is_ocr_failure phrase matching."""

    @pytest.mark.parametrize(
        "text",
        [
            "The text is too small to be legible and clear.",
            "I'm unable to extract the content from this image.",
            "There is no text in this document image.",
            "I cannot read the content. Feel free to let me know!",
            "NO READABLE TEXT found in the image.",
            "I cannot provide a transcription of this image.",
        ],
    )
    def test_known_failure_phrases_detected(self, text):
        assert _is_ocr_failure(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Faktura VAT nr 2024/001\nData: 2024-01-15\nKwota: 1500.00 PLN",
            "Protokol ze spotkania z dnia 10 marca 2024",
            "The quarterly report shows revenue of $2.3M",
            "",
            "Notatki: prosze o kontakt w sprawie zamowienia",
        ],
    )
    def test_valid_content_not_flagged(self, text):
        assert _is_ocr_failure(text) is False

    def test_case_insensitive(self):
        assert _is_ocr_failure("THE TEXT IS TOO SMALL TO BE LEGIBLE") is True
        assert _is_ocr_failure("Too Small To Read") is True


class TestOCRPromptBuilder:
    """Tests for _build_ocr_prompt."""

    def test_prompt_without_language(self):
        prompt = _build_ocr_prompt()
        assert "Extract ALL text" in prompt
        assert "original language" in prompt
        assert "Polish" not in prompt

    def test_prompt_with_language(self):
        prompt = _build_ocr_prompt("Polish")
        assert "Extract ALL text" in prompt
        assert "The document is in Polish" in prompt
        assert "Keep ALL text in Polish" in prompt
        assert "original language" not in prompt

    def test_prompt_with_none_language(self):
        prompt = _build_ocr_prompt(None)
        assert "original language" in prompt


class TestCorpusLanguageDetection:
    """Tests for DocumentIngester._detect_corpus_language."""

    def test_detects_polish_from_md_files(self, tmp_path):
        """Polish markdown files should be detected as Polish."""
        from ingestion import DocumentIngester

        (tmp_path / "notatki.md").write_text(
            "Spotkanie z klientem w sprawie nowej kampanii reklamowej. "
            "Omowilismy budzet i harmonogram prac na kolejny kwartal.",
            encoding="utf-8",
        )
        (tmp_path / "raport.md").write_text(
            "Raport kwartalny za okres od stycznia do marca. "
            "Przychody wzrosly o dwadziescia procent w porownaniu z rokiem poprzednim.",
            encoding="utf-8",
        )
        ing = DocumentIngester()
        files = list(tmp_path.glob("*"))
        result = ing._detect_corpus_language(files)
        assert result == "Polish"

    def test_detects_english_from_txt_files(self, tmp_path):
        """English text files should be detected as English."""
        from ingestion import DocumentIngester

        (tmp_path / "notes.txt").write_text(
            "Meeting with the client to discuss the new marketing campaign. "
            "We reviewed the budget and timeline for the upcoming quarter.",
            encoding="utf-8",
        )
        ing = DocumentIngester()
        files = list(tmp_path.glob("*"))
        result = ing._detect_corpus_language(files)
        assert result == "English"

    def test_returns_none_when_no_text_files(self, tmp_path):
        """Should return None when no .txt or .md files are present."""
        from ingestion import DocumentIngester

        (tmp_path / "data.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "sheet.xlsx").write_bytes(b"fake xlsx")
        ing = DocumentIngester()
        files = list(tmp_path.glob("*"))
        result = ing._detect_corpus_language(files)
        assert result is None

    def test_returns_none_when_files_are_empty(self, tmp_path):
        """Should return None when text files exist but are empty."""
        from ingestion import DocumentIngester

        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        (tmp_path / "blank.md").write_text("   \n  \n", encoding="utf-8")
        ing = DocumentIngester()
        files = list(tmp_path.glob("*"))
        result = ing._detect_corpus_language(files)
        assert result is None

    def test_ignores_non_text_files(self, tmp_path):
        """Should only sample .txt and .md, not .eml or .docx."""
        from ingestion import DocumentIngester

        (tmp_path / "email.eml").write_text(
            "Spotkanie z klientem w sprawie kampanii", encoding="utf-8"
        )
        (tmp_path / "doc.docx").write_bytes(b"fake docx")
        ing = DocumentIngester()
        files = list(tmp_path.glob("*"))
        result = ing._detect_corpus_language(files)
        assert result is None
