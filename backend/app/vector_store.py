"""IntelliAssist AI - FAISS Vector Store Module.

This module provides an in-memory vector database interface backed by FAISS (IndexFlatIP)
for dense vector storage and semantic retrieval of document chunks. It maintains an
exact mapping between internal FAISS integer vector IDs and chunk metadata, guaranteeing
reproducible source attribution, cosine similarity ranking, and strict input validation.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Union
import faiss
import numpy as np


DEFAULT_EMBEDDING_DIMENSION = 384


class FAISSVectorStore:
    """Manages an in-memory FAISS IndexFlatIP vector index with chunk metadata mapping.

    Attributes:
        dimension (int): The dimensional size of vectors stored in this index.
        normalize_vectors (bool): Whether input and query vectors are automatically
            L2-normalized for cosine similarity. If False, vectors are validated to
            be already normalized within floating-point tolerance.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        normalize_vectors: bool = True,
    ) -> None:
        """Initialize an empty FAISS vector store.

        Args:
            dimension: Dimensionality of embedding vectors (default 384).
            normalize_vectors: If True (default), automatically normalizes incoming
                vectors and queries to unit L2 norm. If False, asserts that vectors
                are already L2 normalized (unit length).

        Raises:
            ValueError: If dimension is not a positive integer.
        """
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError(f"dimension must be a positive integer, got: {dimension!r}")

        self._dimension: int = dimension
        self._normalize_vectors: bool = bool(normalize_vectors)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dimension)
        self._records: List[Dict[str, Any]] = []
        self._chunk_ids: Set[str] = set()

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality of this store."""
        return self._dimension

    @property
    def normalize_vectors(self) -> bool:
        """Return whether vectors are automatically L2-normalized."""
        return self._normalize_vectors

    @property
    def total_vectors(self) -> int:
        """Return the total number of vectors currently stored in the index."""
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        """Return True if the index contains zero vectors."""
        return self.total_vectors == 0

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return self.total_vectors

    def clear(self) -> None:
        """Reset the FAISS index and clear all stored chunk metadata and ID mappings."""
        self._index.reset()
        self._records.clear()
        self._chunk_ids.clear()

    def has_chunk(self, chunk_id: str) -> bool:
        """Check whether a chunk_id is already present in the index.

        Args:
            chunk_id: Unique string identifier to check.

        Returns:
            True if chunk_id exists in the store, False otherwise.
        """
        if not isinstance(chunk_id, str):
            return False
        return chunk_id in self._chunk_ids

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored chunk record by its chunk_id.

        Args:
            chunk_id: Unique string identifier to retrieve.

        Returns:
            Copy of chunk dictionary (containing 'chunk_id', 'text', 'metadata')
            or None if chunk_id is not found.
        """
        for rec in self._records:
            if rec["chunk_id"] == chunk_id:
                return {
                    "chunk_id": rec["chunk_id"],
                    "text": rec["text"],
                    "metadata": copy.deepcopy(rec["metadata"]),
                }
        return None

    def _prepare_and_validate_vectors(
        self,
        raw_embeddings: Any,
        expected_count: Optional[int] = None,
    ) -> np.ndarray:
        """Validate, convert, and normalize embedding vectors.

        Args:
            raw_embeddings: Array-like object containing numerical embeddings.
            expected_count: Expected number of rows.

        Returns:
            Contiguous float32 2D numpy array of shape (N, dimension).

        Raises:
            ValueError: If embeddings are malformed, mismatched, non-finite,
                        have wrong dimensions, or fail normalization policy.
            TypeError: If input is None or not convertible to array.
        """
        if raw_embeddings is None:
            raise ValueError("Embeddings cannot be None")

        try:
            arr = np.asarray(raw_embeddings, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert embeddings to float32 numpy array: {e}") from e

        # Handle 1D array representing a single vector when expected_count == 1
        if arr.ndim == 1:
            if expected_count == 1:
                arr = arr.reshape(1, -1)
            elif expected_count == 0:
                arr = arr.reshape(0, self._dimension)
            else:
                arr = arr.reshape(1, -1)

        if arr.ndim != 2:
            raise ValueError(f"Embeddings must be a 2D array, got ndim={arr.ndim}")

        if expected_count is not None and arr.shape[0] != expected_count:
            raise ValueError(
                f"Mismatched counts: chunks has {expected_count} items but embeddings has {arr.shape[0]} items"
            )

        if arr.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimension}, got {arr.shape[1]}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError("Embeddings contain NaN or infinite values")

        if arr.shape[0] == 0:
            return np.empty((0, self._dimension), dtype=np.float32)

        # Calculate L2 norms for each vector row
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError("Zero-magnitude embedding vector encountered; cannot be normalized")

        if self._normalize_vectors:
            arr = arr / norms
        else:
            # Validate that vectors are already unit-normalized within numerical tolerance
            if not np.allclose(norms, 1.0, atol=1e-3):
                raise ValueError(
                    "Embeddings are not L2-normalized. Set normalize_vectors=True to normalize automatically."
                )

        return np.ascontiguousarray(arr, dtype=np.float32)

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[Union[np.ndarray, List[Any]]] = None,
    ) -> int:
        """Add document chunks and their corresponding embedding vectors to the vector store.

        This operation is atomic: if any chunk or embedding vector fails validation,
        the entire batch is rejected without altering the FAISS index or existing metadata.

        Duplicate Handling Policy:
            Each chunk must have a non-empty string `chunk_id`. Duplicate IDs within the
            incoming batch or colliding with previously added chunks are strictly rejected
            with a ValueError to maintain unambiguous 1-to-1 chunk identification.

        Args:
            chunks: List of chunk dictionaries. Each dictionary must contain:
                    - 'chunk_id' (non-empty str)
                    - 'text' (str)
                    - 'metadata' (dict)
            embeddings: Optional 2D array-like or list of numerical vectors of shape
                        (len(chunks), dimension). If None, each dictionary in `chunks`
                        must contain an 'embedding' key.

        Returns:
            Number of chunks successfully added to the store.

        Raises:
            ValueError: If chunks/embeddings lengths mismatch, duplicate chunk IDs
                        are detected, required fields are missing, or vectors are invalid.
            TypeError: If chunks is not a list or elements have invalid types.
        """
        if chunks is None:
            raise ValueError("chunks list cannot be None")
        if not isinstance(chunks, list):
            raise TypeError(f"chunks must be a list of chunk dictionaries, got: {type(chunks).__name__}")

        # Handle empty chunks list
        if len(chunks) == 0:
            if embeddings is not None:
                emb_arr = np.asarray(embeddings)
                if emb_arr.size != 0:
                    raise ValueError(
                        f"Mismatched counts: chunks is empty but embeddings has {len(emb_arr)} items"
                    )
            return 0

        # Extract embeddings if embedded directly in chunks
        if embeddings is None:
            extracted_embeddings: List[Any] = []
            for idx, c in enumerate(chunks):
                if not isinstance(c, dict):
                    raise ValueError(f"Chunk at index {idx} must be a dictionary")
                if "embedding" not in c:
                    raise ValueError(
                        f"Embeddings parameter was None and chunk at index {idx} is missing 'embedding' field"
                    )
                extracted_embeddings.append(c["embedding"])
            vectors_to_validate = extracted_embeddings
        else:
            vectors_to_validate = embeddings

        # Validate vectors before modifying any state
        validated_vectors = self._prepare_and_validate_vectors(
            vectors_to_validate, expected_count=len(chunks)
        )

        # Validate chunk dictionaries before modifying any state
        batch_ids: Set[str] = set()
        validated_records: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {idx} must be a dictionary")

            # Validate chunk_id
            if "chunk_id" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing required 'chunk_id' field")
            chunk_id = chunk["chunk_id"]
            if not isinstance(chunk_id, str):
                raise TypeError(
                    f"Chunk at index {idx} 'chunk_id' must be a string, got: {type(chunk_id).__name__}"
                )
            if not chunk_id.strip():
                raise ValueError(f"Chunk at index {idx} has empty or whitespace-only 'chunk_id'")

            # Duplicate ID checks
            if chunk_id in self._chunk_ids:
                raise ValueError(
                    f"Duplicate chunk_id detected: '{chunk_id}' already exists in index"
                )
            if chunk_id in batch_ids:
                raise ValueError(
                    f"Duplicate chunk_id detected within batch: '{chunk_id}'"
                )
            batch_ids.add(chunk_id)

            # Validate text
            if "text" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing required 'text' field")
            text = chunk["text"]
            if not isinstance(text, str):
                raise TypeError(
                    f"Chunk at index {idx} 'text' must be a string, got: {type(text).__name__}"
                )

            # Validate metadata
            if "metadata" not in chunk:
                raise ValueError(f"Chunk at index {idx} missing required 'metadata' field")
            metadata = chunk["metadata"]
            if not isinstance(metadata, dict):
                raise TypeError(
                    f"Chunk at index {idx} 'metadata' must be a dict, got: {type(metadata).__name__}"
                )

            validated_records.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": copy.deepcopy(metadata),
            })

        # Atomic commit: All validations passed; add to FAISS index and local records
        self._index.add(validated_vectors)
        self._records.extend(validated_records)
        self._chunk_ids.update(batch_ids)

        return len(validated_records)

    def search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top_k most similar chunks for a given query embedding vector.

        Similarity scores represent the inner product of normalized vectors,
        which corresponds to cosine similarity in the range [-1.0, 1.0]. Scores
        are strictly ranked in descending order (highest similarity first).

        Args:
            query_embedding: 1D array or list of floats of length `dimension`.
            top_k: Number of nearest chunks to return (default 5). Must be a positive integer.

        Returns:
            List of result dictionaries sorted in descending order of similarity:
            [
                {
                    "chunk_id": str,
                    "text": str,
                    "metadata": dict,
                    "similarity_score": float
                }
            ]

        Raises:
            ValueError: If query_embedding is malformed, has wrong dimensions,
                        contains NaN/Inf, or top_k is non-positive.
            TypeError: If top_k is not an integer or query_embedding is invalid type.
        """
        # Validate top_k
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

        # Safe return on empty index
        if self.is_empty:
            return []

        # Validate query_embedding
        if query_embedding is None:
            raise ValueError("query_embedding cannot be None")

        try:
            q_arr = np.asarray(query_embedding, dtype=np.float32)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to convert query_embedding to float32 array: {e}") from e

        if q_arr.ndim == 1:
            if q_arr.shape[0] != self._dimension:
                raise ValueError(
                    f"Query embedding dimension mismatch: expected {self._dimension}, got {q_arr.shape[0]}"
                )
            q_vec = q_arr.reshape(1, -1)
        elif q_arr.ndim == 2:
            if q_arr.shape[0] != 1 or q_arr.shape[1] != self._dimension:
                raise ValueError(
                    f"Query embedding must have shape ({self._dimension},) or (1, {self._dimension}), got {q_arr.shape}"
                )
            q_vec = q_arr
        else:
            raise ValueError(f"Query embedding must be 1D (or 2D with 1 row), got ndim={q_arr.ndim}")

        if not np.all(np.isfinite(q_vec)):
            raise ValueError("Query embedding contains NaN or infinite values")

        norm = float(np.linalg.norm(q_vec))
        if norm == 0.0:
            raise ValueError("Zero-magnitude query embedding cannot be normalized")

        if self._normalize_vectors:
            q_vec = q_vec / norm
        else:
            if not np.isclose(norm, 1.0, atol=1e-3):
                raise ValueError("Query embedding is not L2-normalized")

        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        # Retrieve at most min(top_k, self.total_vectors)
        k = min(top_k, self.total_vectors)
        distances, indices = self._index.search(q_vec, k)

        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            # FAISS returns -1 for unassigned neighbor slots
            if idx == -1 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            results.append({
                "chunk_id": rec["chunk_id"],
                "text": rec["text"],
                "metadata": copy.deepcopy(rec["metadata"]),
                "similarity_score": float(dist),
            })

        return results


def create_vector_store(
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    normalize_vectors: bool = True,
) -> FAISSVectorStore:
    """Create and return an initialized FAISSVectorStore instance."""
    return FAISSVectorStore(dimension=dimension, normalize_vectors=normalize_vectors)
