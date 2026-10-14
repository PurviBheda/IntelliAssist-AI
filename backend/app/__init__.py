"""IntelliAssist AI - Backend Application Package."""

from .document_processor import process_document
from .text_chunker import chunk_document, split_text_into_chunks
from .embedding_service import (
    EmbeddingService,
    embed_chunks,
    embed_text,
    embed_texts,
)

__all__ = [
    "process_document",
    "chunk_document",
    "split_text_into_chunks",
    "EmbeddingService",
    "embed_chunks",
    "embed_text",
    "embed_texts",
]
