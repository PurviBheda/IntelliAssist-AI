"""IntelliAssist AI - Text Embeddings Module.

This module provides local text embeddings using sentence-transformers with the
'all-MiniLM-L6-v2' model (384-dimensional dense vectors). It operates completely
locally without requiring paid API keys, lazy-loads the model only on demand,
and batches texts efficiently for subsequent vector storage (e.g. FAISS).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingService:
    """Manages lazy loading and generation of text embeddings using SentenceTransformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize_embeddings: bool = True,
        embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        """Initialize the embedding service without loading the model.

        Args:
            model_name: HuggingFace model identifier.
            normalize_embeddings: Whether to L2-normalize vectors for cosine similarity.
            embedding_dimension: Expected dimension of output vectors.
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dimension = embedding_dimension
        self._model: Any = None

    @property
    def is_model_loaded(self) -> bool:
        """Return True if the underlying model has already been loaded into memory."""
        return self._model is not None

    def get_model(self) -> Any:
        """Load and return the underlying SentenceTransformer model lazily.

        Reuses the existing model instance if already loaded.

        Returns:
            SentenceTransformer model instance.

        Raises:
            RuntimeError: If model initialization or download fails due to network
                          or environment issues.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load sentence-transformer model '{self.model_name}': {e}. "
                    "Ensure sentence-transformers is installed and an active internet connection "
                    "is available for initial model download."
                ) from e
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text string into a 1D numerical embedding vector.

        Args:
            text: Non-empty string to embed.

        Returns:
            1D numpy array with shape (embedding_dimension,) and dtype float32.

        Raises:
            ValueError: If text is None or empty/whitespace-only.
            TypeError: If text is not a string.
        """
        if text is None:
            raise ValueError("Input text cannot be None")
        if not isinstance(text, str):
            raise TypeError(
                f"Input text must be a string, got: {type(text).__name__}"
            )
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Input text cannot be empty or whitespace-only")

        model = self.get_model()
        vector = model.encode(
            cleaned,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        arr = np.asarray(vector, dtype=np.float32).flatten()
        return arr

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Convert a list of text strings into a 2D numerical embedding array.

        If texts is empty, returns an empty array of shape (0, embedding_dimension)
        without triggering lazy model initialization.

        Args:
            texts: List of non-empty strings to embed.
            batch_size: Number of texts to process per batch. Must be a positive integer.

        Returns:
            2D numpy array with shape (len(texts), embedding_dimension) and dtype float32.

        Raises:
            ValueError: If texts is None, contains empty/whitespace-only items,
                        or batch_size is invalid.
            TypeError: If texts is not a list or contains non-string items.
        """
        if texts is None:
            raise ValueError("Input texts list cannot be None")
        if not isinstance(texts, list):
            raise TypeError(
                f"Input texts must be a list of strings, got: {type(texts).__name__}"
            )

        # Handle empty list without loading model
        if len(texts) == 0:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        # Validate configuration
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got: {batch_size!r}"
            )

        # Validate individual elements
        cleaned_texts: List[str] = []
        for idx, item in enumerate(texts):
            if item is None:
                raise ValueError(f"Text item at index {idx} cannot be None")
            if not isinstance(item, str):
                raise TypeError(
                    f"Text item at index {idx} must be a string, got: {type(item).__name__}"
                )
            cleaned = item.strip()
            if not cleaned:
                raise ValueError(
                    f"Text item at index {idx} cannot be empty or whitespace-only"
                )
            cleaned_texts.append(cleaned)

        model = self.get_model()
        vectors = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Generate embeddings for a list of chunk dictionaries.

        Preserves the original chunk_id, text, and metadata while appending
        the 1D embedding vector under the 'embedding' key.

        Args:
            chunks: List of chunk dictionaries produced by chunk_document().
            batch_size: Batch size for embedding generation.

        Returns:
            List of dictionaries, each containing:
            {
                "chunk_id": str,
                "text": str,
                "metadata": dict,
                "embedding": np.ndarray (shape: (384,), dtype: float32)
            }

        Raises:
            ValueError: If chunks is None or any chunk dictionary is malformed.
            TypeError: If chunks is not a list.
        """
        if chunks is None:
            raise ValueError("chunks list cannot be None")
        if not isinstance(chunks, list):
            raise TypeError(
                f"chunks must be a list of chunk dictionaries, got: {type(chunks).__name__}"
            )

        if len(chunks) == 0:
            return []

        # Validate chunk dictionaries
        texts: List[str] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {idx} must be a dictionary")
            if "chunk_id" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing 'chunk_id' field")
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing 'text' field")
            if "metadata" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing 'metadata' field")

            texts.append(chunk["text"])

        vectors = self.embed_texts(texts, batch_size=batch_size)

        embedded_chunks: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": dict(chunk["metadata"]),
                "embedding": vectors[idx],
            })

        return embedded_chunks

    def as_langchain_embeddings(self) -> "LangChainEmbeddingsAdapter":
        """Return a LangChain-compatible Embeddings adapter wrapping this service."""
        return LangChainEmbeddingsAdapter(self)


# ============================================================================
# LangChain Adapter Implementation
# ============================================================================

try:
    from langchain_core.embeddings import Embeddings as BaseLangChainEmbeddings
except ImportError:
    class BaseLangChainEmbeddings:  # type: ignore[no-redef]
        """Fallback base class when langchain-core is not installed."""
        pass


class LangChainEmbeddingsAdapter(BaseLangChainEmbeddings):
    """LangChain-compatible Embeddings adapter delegating to EmbeddingService."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search documents using the underlying EmbeddingService."""
        if not texts:
            return []
        arr = self.embedding_service.embed_texts(texts)
        return arr.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using the underlying EmbeddingService."""
        arr = self.embedding_service.embed_text(text)
        return arr.tolist()


# Global singleton instance for module-level convenience functions
_default_service: Optional[EmbeddingService] = None


def get_default_embedding_service() -> EmbeddingService:
    """Retrieve or create the module-level default EmbeddingService singleton."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def embed_text(text: str) -> np.ndarray:
    """Convert a single text string into a 1D numerical embedding vector.

    Convenience wrapper around get_default_embedding_service().embed_text.
    """
    return get_default_embedding_service().embed_text(text)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Convert a list of text strings into a 2D numerical embedding array.

    Convenience wrapper around get_default_embedding_service().embed_texts.
    """
    return get_default_embedding_service().embed_texts(texts, batch_size=batch_size)


def embed_chunks(
    chunks: List[Dict[str, Any]], batch_size: int = 32
) -> List[Dict[str, Any]]:
    """Generate embeddings for a list of document chunks.

    Convenience wrapper around get_default_embedding_service().embed_chunks.
    """
    return get_default_embedding_service().embed_chunks(chunks, batch_size=batch_size)
