"""Content-based deduplication using hashing."""

import hashlib

from storage.metadata_db import MetadataDB


def compute_document_hash(text: str) -> str:
    """Compute deterministic hash of document content.

    Normalizes whitespace but preserves case to avoid collisions
    between documents that differ only in capitalization.

    Args:
        text: Document text content.

    Returns:
        16-character hex hash.
    """
    # Normalize whitespace only (preserve case)
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_duplicate(content_hash: str, metadata_db: MetadataDB) -> bool:
    """Check if document with this hash already exists.

    Args:
        content_hash: Hash computed by compute_document_hash.
        metadata_db: MetadataDB instance to check.

    Returns:
        True if document exists, False otherwise.
    """
    return metadata_db.hash_exists(content_hash)
