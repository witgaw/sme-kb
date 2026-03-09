"""Configuration management for RAG system."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGConfig(BaseSettings):
    """RAG system configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    vector_db_path: str = Field(default="./chroma_db", alias="VECTOR_DB_PATH")
    metadata_db_path: str = Field(default="./metadata.db", alias="METADATA_DB_PATH")

    # Embeddings
    embedding_mode: Literal["local", "hosted"] = Field(default="local", alias="EMBEDDING_MODE")
    embedding_model: str = Field(default="nomic-ai/nomic-embed-text-v1.5", alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # Chunking
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(default=5, alias="TOP_K")
    use_reranker: bool = Field(default=False, alias="USE_RERANKER")

    # RAG type
    rag_type: Literal["basic", "knowledge_graph"] = Field(default="basic", alias="RAG_TYPE")

    # LLM
    llm_provider: Literal["ollama", "openrouter"] = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model: str = Field(default="llama3.1:8b", alias="LLM_MODEL")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")

    # OCR
    ocr_enabled: bool = Field(default=False, alias="OCR_ENABLED")
    ocr_provider: Literal["ollama", "openrouter"] = Field(default="ollama", alias="OCR_PROVIDER")
    ocr_model: str = Field(default="llava:7b", alias="OCR_MODEL")

    # Generation
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    max_tokens: int = Field(default=2000, alias="MAX_TOKENS")


# Global config instance - can be overridden for testing
_config: RAGConfig | None = None


def get_config() -> RAGConfig:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration instance (useful for testing)."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset configuration to default (reload from environment)."""
    global _config
    _config = None
