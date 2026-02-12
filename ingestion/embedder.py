"""Embedding computation with batch support."""

import os
from typing import Literal

import openai
from sentence_transformers import SentenceTransformer

from config import get_config


class Embedder:
    """Compute embeddings using local or hosted models."""

    def __init__(
        self,
        mode: Literal["local", "hosted"] | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ):
        """Initialize embedder.

        Args:
            mode: 'local' for sentence-transformers, 'hosted' for OpenRouter.
                  Uses config default if None.
            model_name: Model to use. Uses config default if None.
            batch_size: Batch size for embedding computation. Uses config default if None.
        """
        config = get_config()
        self.mode = mode or config.embedding_mode
        self.model_name = model_name or config.embedding_model
        self.batch_size = batch_size or config.embedding_batch_size

        if self.mode == "local":
            try:
                self._model = SentenceTransformer(
                    self.model_name, trust_remote_code=True, local_files_only=True
                )
            except OSError:
                # Model not cached yet — download it
                self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        else:
            api_key = config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY required for hosted embeddings")
            self._client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self.mode == "local":
            return self._embed_local(texts)
        else:
            return self._embed_hosted(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using local model with batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def _embed_hosted(self, texts: list[str]) -> list[list[float]]:
        """Compute embeddings using OpenRouter API with batching."""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                input=batch,
                model="openai/text-embedding-3-small",
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Compute embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        embeddings = self.embed([text])
        return embeddings[0]

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        if self.mode == "local":
            return self._model.get_sentence_embedding_dimension()
        else:
            # text-embedding-3-small has 1536 dimensions
            return 1536
