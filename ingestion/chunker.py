"""Text chunking strategies."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_config


class DocumentChunker:
    """Split documents into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize chunker.

        Args:
            chunk_size: Maximum chunk size in characters. Uses config default if None.
            chunk_overlap: Overlap between chunks. Uses config default if None.
        """
        config = get_config()
        self.chunk_size = chunk_size or config.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def chunk(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []

        return self.splitter.split_text(text)

    def chunk_with_metadata(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        """Split text and attach metadata to each chunk.

        Args:
            text: Text to split.
            metadata: Metadata to attach to each chunk.

        Returns:
            List of dicts with 'content' and 'metadata' keys.
        """
        chunks = self.chunk(text)
        metadata = metadata or {}

        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
            for i, chunk in enumerate(chunks)
        ]
