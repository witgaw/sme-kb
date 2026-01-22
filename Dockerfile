FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev

# Copy application code
COPY config/ ./config/
COPY ingestion/ ./ingestion/
COPY storage/ ./storage/
COPY retrieval/ ./retrieval/
COPY generation/ ./generation/
COPY pipeline/ ./pipeline/
COPY ui/ ./ui/
COPY eval/ ./eval/

# Create data directories
RUN mkdir -p /app/data /app/chroma_db

# Set environment variables
ENV VECTOR_DB_PATH=/app/chroma_db
ENV METADATA_DB_PATH=/app/metadata.db
ENV PYTHONUNBUFFERED=1

# Expose Gradio port
EXPOSE 7860

# Default command - run web UI
CMD ["uv", "run", "rag", "serve"]
