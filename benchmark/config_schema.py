"""Pydantic model for benchmark YAML configuration files."""

from typing import Literal

import yaml
from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """A single benchmark configuration to evaluate."""

    name: str
    description: str = ""

    # Chunking
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)

    # Embedding
    embedding_mode: Literal["local", "hosted"] = "local"
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Retrieval
    top_k: int = Field(default=5)
    use_reranker: bool = False

    # LLM
    llm_provider: Literal["ollama", "openrouter"] = "ollama"
    llm_model: str = "llama3.1:8b"
    temperature: float = 0.7
    max_tokens: int = 2000

    # OCR / VLM
    ocr_enabled: bool = False
    ocr_provider: Literal["ollama", "openrouter"] = "ollama"
    ocr_model: str = "llava:7b"

    # Filtering
    skip_ocr: bool = True
    skip_db: bool = True

    @property
    def index_fingerprint(self) -> str:
        """Configs with the same fingerprint share a single ingested index."""
        ocr_part = f"_ocr_{self.ocr_model}" if self.ocr_enabled else "_noocr"
        return (
            f"{self.chunk_size}_{self.chunk_overlap}"
            f"_{self.embedding_mode}_{self.embedding_model}{ocr_part}"
        )

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load a BenchmarkConfig from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
