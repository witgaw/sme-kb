"""Tests for retrieval functionality."""

from retrieval.reranker import NoOpReranker


class TestRetriever:
    """Tests for Retriever."""

    def test_retrieve_returns_results(self, tmp_db_paths, sample_documents_dir, mock_embedder):
        """Test that retrieval returns results after ingestion."""
        from ingestion import DocumentIngester
        from retrieval import Retriever

        # Ingest documents first
        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_directory(sample_documents_dir)

        # Retrieve
        retriever = Retriever(vector_db_path=tmp_db_paths["vector_db_path"])
        results = retriever.retrieve("Kiedy odnowienie Klient B?", top_k=3)

        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("source" in r for r in results)

    def test_retrieve_respects_top_k(self, tmp_db_paths, sample_documents_dir, mock_embedder):
        """Test that top_k parameter limits results."""
        from ingestion import DocumentIngester
        from retrieval import Retriever

        # Ingest documents
        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_directory(sample_documents_dir)

        retriever = Retriever(vector_db_path=tmp_db_paths["vector_db_path"])

        results_1 = retriever.retrieve("test", top_k=1)
        results_2 = retriever.retrieve("test", top_k=2)

        assert len(results_1) == 1
        assert len(results_2) <= 2

    def test_retrieve_empty_db(self, tmp_db_paths, mock_embedder):
        """Test retrieval on empty database returns empty list."""
        from retrieval import Retriever

        retriever = Retriever(vector_db_path=tmp_db_paths["vector_db_path"])
        results = retriever.retrieve("any query", top_k=5)

        assert results == []

    def test_retrieve_includes_metadata(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test that retrieved chunks include metadata."""
        from ingestion import DocumentIngester
        from retrieval import Retriever

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_file(sample_text_file)

        retriever = Retriever(vector_db_path=tmp_db_paths["vector_db_path"])
        results = retriever.retrieve("software engineering", top_k=1)

        assert len(results) == 1
        assert "metadata" in results[0]
        assert "document_id" in results[0]["metadata"]


class TestReranker:
    """Tests for Reranker classes."""

    def test_noop_reranker_preserves_order(self):
        """Test NoOpReranker returns chunks in original order."""
        reranker = NoOpReranker()
        chunks = [
            {"content": "First", "id": 1},
            {"content": "Second", "id": 2},
            {"content": "Third", "id": 3},
        ]

        result = reranker.rerank("query", chunks)

        assert result == chunks

    def test_noop_reranker_respects_top_k(self):
        """Test NoOpReranker respects top_k parameter."""
        reranker = NoOpReranker()
        chunks = [
            {"content": "First", "id": 1},
            {"content": "Second", "id": 2},
            {"content": "Third", "id": 3},
        ]

        result = reranker.rerank("query", chunks, top_k=2)

        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2

    def test_noop_reranker_empty_chunks(self):
        """Test NoOpReranker handles empty chunk list."""
        reranker = NoOpReranker()
        result = reranker.rerank("query", [])
        assert result == []

    # Note: Full Reranker tests would require the cross-encoder model
    # which is expensive to load. We test the NoOpReranker instead.
