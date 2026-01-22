# RAG System

Minimal RAG (Retrieval-Augmented Generation) system built with LangGraph, ChromaDB, and Gradio.

## Features

- Multi-format document ingestion (.pdf, .docx, .xlsx, .pptx, .txt, .md, .eml)
- Content-based deduplication
- Local embeddings via sentence-transformers or hosted via OpenRouter
- ChromaDB vector storage (serverless)
- LangGraph workflow orchestration
- Ollama and OpenRouter LLM support
- CLI and web interfaces
- Evaluation with synthetic datasets

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd sme-kb

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```bash
# 1. Start Ollama (if using local models)
ollama serve

# 2. Pull required models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 3. Ingest documents
uv run rag ingest /path/to/documents

# 4. Query via CLI
uv run rag query "Your question here"

# 5. Or launch web UI
uv run rag serve
```

## CLI Commands

```bash
# Ingest documents from a directory
uv run rag ingest /path/to/documents --recursive

# Query the system
uv run rag query "What is the contract renewal date?"

# Retrieve chunks without generation (debug)
uv run rag retrieve "search query" --top-k 5

# Show database statistics
uv run rag stats

# Launch web UI
uv run rag serve

# Clear all data
uv run rag clear --yes
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# LLM Provider
LLM_PROVIDER=ollama          # or "openrouter"
LLM_MODEL=llama3.1:8b

# Embeddings
EMBEDDING_MODE=local         # or "hosted"
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# OpenRouter (if using hosted)
OPENROUTER_API_KEY=your_key_here

# Ollama
OLLAMA_BASE_URL=http://localhost:11434

# Storage
VECTOR_DB_PATH=./chroma_db
METADATA_DB_PATH=./metadata.db
```

## Docker

```bash
# Build and run with docker-compose
docker-compose up -d

# The web UI will be available at http://localhost:7860
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linter
uv run ruff check .
uv run ruff format .
```

## Architecture

```
sme-kb/
├── config/             # Configuration management
├── ingestion/          # Document loading, chunking, embedding
├── storage/            # ChromaDB + SQLite
├── retrieval/          # Query retrieval and reranking
├── generation/         # LLM client and prompt building
├── pipeline/           # LangGraph workflow
├── ui/                 # CLI and web interfaces
├── eval/               # Evaluation tools
└── tests/              # Test suite
```

## License

MIT
