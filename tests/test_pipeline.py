"""Tests for RAG pipeline."""

from unittest.mock import patch


class TestRAGPipeline:
    """Tests for RAG pipeline integration."""

    def test_pipeline_initialization(self, tmp_db_paths, mock_embedder):
        """Test pipeline initializes with config."""
        from config import RAGConfig, set_config
        from pipeline.rag_pipeline import RAGPipeline

        config = RAGConfig(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        set_config(config)

        with patch("generation.llm_client.httpx.Client"):
            pipeline = RAGPipeline()

            assert pipeline.llm_provider == "ollama"
            assert pipeline.embedding_mode == "local"

    def test_pipeline_query_returns_answer(
        self, tmp_db_paths, sample_documents_dir, mock_embedder, mock_llm_client
    ):
        """Test pipeline query returns answer with sources."""
        from config import RAGConfig, set_config
        from ingestion import DocumentIngester
        from pipeline.rag_pipeline import RAGPipeline

        config = RAGConfig(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        set_config(config)

        # Ingest documents
        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_directory(sample_documents_dir)

        # Query
        pipeline = RAGPipeline()
        result = pipeline.query("Kiedy odnowienie Klient B?", return_sources=True)

        assert "answer" in result
        assert "sources" in result
        assert "chunks" in result
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)

    def test_pipeline_query_without_sources(
        self, tmp_db_paths, sample_documents_dir, mock_embedder, mock_llm_client
    ):
        """Test pipeline query without sources."""
        from config import RAGConfig, set_config
        from ingestion import DocumentIngester
        from pipeline.rag_pipeline import RAGPipeline

        config = RAGConfig(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        set_config(config)

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_directory(sample_documents_dir)

        pipeline = RAGPipeline()
        result = pipeline.query("test query", return_sources=False)

        assert "answer" in result
        assert "sources" not in result

    def test_pipeline_retrieve_only(self, tmp_db_paths, sample_documents_dir, mock_embedder):
        """Test retrieve_only method."""
        from config import RAGConfig, set_config
        from ingestion import DocumentIngester
        from pipeline.rag_pipeline import RAGPipeline

        config = RAGConfig(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        set_config(config)

        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_directory(sample_documents_dir)

        with patch("generation.llm_client.httpx.Client"):
            pipeline = RAGPipeline()
            chunks = pipeline.retrieve_only("Kiedy odnowienie?", top_k=2)

        assert isinstance(chunks, list)
        assert len(chunks) <= 2


class TestRAGGraph:
    """Tests for LangGraph workflow."""

    def test_graph_builds_successfully(self, tmp_db_paths, mock_embedder):
        """Test that RAG graph compiles without errors."""
        from config import RAGConfig, set_config
        from pipeline.graph import RAGComponents, build_rag_graph

        config = RAGConfig(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        set_config(config)

        with patch("generation.llm_client.httpx.Client"):
            components = RAGComponents()
            graph = build_rag_graph(components)

            assert graph is not None

    def test_graph_state_types(self):
        """Test RAGState has expected fields."""
        from pipeline.graph import RAGState

        # RAGState is a TypedDict, check its annotations
        annotations = RAGState.__annotations__

        assert "query" in annotations
        assert "retrieved_chunks" in annotations
        assert "reranked_chunks" in annotations
        assert "response" in annotations
        assert "sources" in annotations


class TestStorageIntegration:
    """Tests for storage layer integration."""

    def test_metadata_db_persistence(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test that metadata persists across instances."""
        from ingestion import DocumentIngester
        from storage.metadata_db import MetadataDB

        # Ingest with one instance
        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_file(sample_text_file)

        # Check with new instance
        db = MetadataDB(db_path=tmp_db_paths["metadata_db_path"])
        count = db.get_document_count()

        assert count == 1

    def test_vector_db_persistence(self, tmp_db_paths, sample_text_file, mock_embedder):
        """Test that vector embeddings persist across instances."""
        from ingestion import DocumentIngester
        from storage.vector_db import VectorStore

        # Ingest
        ingester = DocumentIngester(
            vector_db_path=tmp_db_paths["vector_db_path"],
            metadata_db_path=tmp_db_paths["metadata_db_path"],
        )
        ingester.ingest_file(sample_text_file)

        # Check with new instance
        vector_db = VectorStore(persist_directory=tmp_db_paths["vector_db_path"])
        count = vector_db.count()

        assert count >= 1
