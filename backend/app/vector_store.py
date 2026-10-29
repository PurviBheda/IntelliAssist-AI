"""IntelliAssist AI - FAISS Vector Store Module.

This module provides an in-memory vector database interface backed by FAISS (IndexFlatIP)
for dense vector storage and semantic retrieval of document chunks. It maintains an
exact mapping between internal FAISS integer vector IDs and chunk metadata, guaranteeing
reproducible source attribution, cosine similarity ranking, and strict input validation.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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
        self._stored_embeddings: List[np.ndarray] = []
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix: Optional[Any] = None

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

    @property
    def tfidf_vectorizer(self) -> Optional[TfidfVectorizer]:
        """Return the fitted scikit-learn TfidfVectorizer instance, or None if unindexed."""
        return self._tfidf_vectorizer

    @property
    def tfidf_matrix(self) -> Optional[Any]:
        """Return the sparse TF-IDF document-term matrix, or None if unindexed."""
        return self._tfidf_matrix

    @property
    def tfidf_vocabulary_size(self) -> int:
        """Return the number of unique terms in the active TF-IDF vocabulary."""
        if self._tfidf_vectorizer is not None and hasattr(self._tfidf_vectorizer, "vocabulary_"):
            return len(self._tfidf_vectorizer.vocabulary_)
        return 0

    def __len__(self) -> int:
        """Return the number of stored vectors."""
        return self.total_vectors

    def _rebuild_tfidf_index(self) -> None:
        """Fit TF-IDF on active document chunks and construct sparse matrix representation.

        Safely handles empty collections, texts with only stopwords or punctuation,
        and ensures strict row alignment with self._records.
        """
        if not self._records:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        corpus = [rec.get("text", "") for rec in self._records]
        has_content = any(t and t.strip() for t in corpus)
        if not has_content:
            self._tfidf_vectorizer = None
            self._tfidf_matrix = None
            return

        try:
            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                lowercase=True,
                stop_words="english",
                sublinear_tf=True,
                norm="l2",
            )
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # If all tokens were English stopwords, retry without stop word filtering
            try:
                vectorizer = TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    lowercase=True,
                    stop_words=None,
                    sublinear_tf=True,
                    norm="l2",
                )
                matrix = vectorizer.fit_transform(corpus)
            except ValueError:
                self._tfidf_vectorizer = None
                self._tfidf_matrix = None
                return

        self._tfidf_vectorizer = vectorizer
        self._tfidf_matrix = matrix

    def clear(self) -> None:
        """Reset the FAISS index, clear stored chunk metadata, and purge TF-IDF indices."""
        self._index.reset()
        self._records.clear()
        self._chunk_ids.clear()
        self._stored_embeddings.clear()
        self._tfidf_vectorizer = None
        self._tfidf_matrix = None

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

        # Atomic commit: All validations passed; add to FAISS index, local records, and rebuild TF-IDF
        self._index.add(validated_vectors)
        self._stored_embeddings.extend(list(validated_vectors))
        self._records.extend(validated_records)
        self._chunk_ids.update(batch_ids)
        self._rebuild_tfidf_index()

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

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by chunk_id and synchronize FAISS and TF-IDF indices.

        Args:
            chunk_id: String identifier of the chunk to remove.

        Returns:
            True if the chunk was located and removed, False otherwise.
        """
        if not isinstance(chunk_id, str) or chunk_id not in self._chunk_ids:
            return False

        target_idx: Optional[int] = None
        for idx, rec in enumerate(self._records):
            if rec["chunk_id"] == chunk_id:
                target_idx = idx
                break

        if target_idx is None:
            return False

        del self._records[target_idx]
        del self._stored_embeddings[target_idx]
        self._chunk_ids.remove(chunk_id)

        # Rebuild FAISS index from remaining embeddings
        self._index.reset()
        if self._stored_embeddings:
            stacked = np.ascontiguousarray(np.stack(self._stored_embeddings), dtype=np.float32)
            self._index.add(stacked)

        # Rebuild TF-IDF index
        self._rebuild_tfidf_index()
        return True

    def delete_chunks_by_document(self, filename: str) -> int:
        """Delete all chunks belonging to a document and synchronize FAISS and TF-IDF indices.

        Args:
            filename: Target document filename (e.g. 'security_protocol.txt').

        Returns:
            Number of chunks successfully removed.
        """
        if not filename or not isinstance(filename, str):
            return 0

        target_name = Path(filename).name.lower()
        target_stem = Path(filename).stem.lower()

        keep_records: List[Dict[str, Any]] = []
        keep_embeddings: List[np.ndarray] = []
        keep_ids: Set[str] = set()
        deleted_count = 0

        for rec, emb in zip(self._records, self._stored_embeddings):
            rec_meta = rec.get("metadata") or {}
            rec_file = str(rec_meta.get("filename") or "").lower()
            rec_cid = str(rec.get("chunk_id") or "").lower()

            is_match = (
                rec_file == target_name
                or rec_file.endswith(f"/{target_name}")
                or rec_file.endswith(f"\\{target_name}")
                or rec_cid.startswith(f"{target_name}_")
                or rec_cid.startswith(f"{target_stem}_")
            )

            if is_match:
                deleted_count += 1
            else:
                keep_records.append(rec)
                keep_embeddings.append(emb)
                keep_ids.add(rec["chunk_id"])

        if deleted_count == 0:
            return 0

        self._records = keep_records
        self._stored_embeddings = keep_embeddings
        self._chunk_ids = keep_ids

        # Rebuild FAISS index
        self._index.reset()
        if self._stored_embeddings:
            stacked = np.ascontiguousarray(np.stack(self._stored_embeddings), dtype=np.float32)
            self._index.add(stacked)

        # Rebuild TF-IDF index
        self._rebuild_tfidf_index()
        return deleted_count

    def search_tfidf(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks ranked by TF-IDF sparse vector cosine similarity.

        Args:
            query: User query string to match against chunk vocabulary.
            top_k: Maximum number of chunks to return (default 5).

        Returns:
            List of chunk result dictionaries ordered by descending similarity score.

        Raises:
            ValueError: If top_k is not a positive integer.
            TypeError: If query is not a string or top_k is not an int.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")

        if query is None:
            raise ValueError("query cannot be None")
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got: {type(query).__name__}")

        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        if self.is_empty or self._tfidf_vectorizer is None or self._tfidf_matrix is None:
            return []

        try:
            q_vec = self._tfidf_vectorizer.transform([cleaned_query])
        except Exception:
            return []

        if q_vec.nnz == 0:
            return []

        sim_matrix = self._tfidf_matrix.dot(q_vec.T)
        scores = sim_matrix.toarray().ravel()

        matching_indices = np.where(scores > 0.0)[0]
        if len(matching_indices) == 0:
            return []

        sorted_order = np.argsort(-scores[matching_indices])
        top_indices = matching_indices[sorted_order][:top_k]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            rec = self._records[idx]
            results.append({
                "chunk_id": rec["chunk_id"],
                "text": rec["text"],
                "metadata": copy.deepcopy(rec["metadata"]),
                "similarity_score": float(scores[idx]),
            })

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[Union[np.ndarray, List[float]]] = None,
        top_k: int = 5,
        dense_weight: float = 0.6,
        tfidf_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using hybrid dense semantic (FAISS) and sparse keyword (TF-IDF) search.

        Combines dense semantic retrieval and sparse TF-IDF keyword retrieval using
        Reciprocal Rank Fusion (RRF) with normalized scoring:
            RRF_score(d) = dense_weight * (1 / (rrf_k + rank_dense)) + tfidf_weight * (1 / (rrf_k + rank_tfidf))
            normalized_score = RRF_score(d) / max_possible_rrf

        Preserves exact citation metadata, deduplicates by stable chunk_id, and decorates
        metadata with dense_score, tfidf_score, and retrieval_source.

        Args:
            query_text: User question string.
            query_embedding: Optional precomputed dense embedding vector for the query.
            top_k: Maximum number of chunks to return (default 5).
            dense_weight: Weight assigned to dense semantic rank (default 0.6).
            tfidf_weight: Weight assigned to sparse TF-IDF rank (default 0.4).
            rrf_k: Constant smoothing parameter for reciprocal rank fusion (default 60).

        Returns:
            List of deduplicated, hybrid-ranked chunk dictionaries.

        Raises:
            ValueError: If top_k <= 0, rrf_k <= 0, weights negative, or all inputs empty.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k!r}")
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k <= 0:
            raise ValueError(f"rrf_k must be a positive integer, got: {rrf_k!r}")
        if dense_weight < 0.0 or tfidf_weight < 0.0 or (dense_weight + tfidf_weight <= 0.0):
            raise ValueError("dense_weight and tfidf_weight must be non-negative and sum to > 0")

        if self.is_empty:
            return []

        cleaned_query = query_text.strip() if isinstance(query_text, str) else ""

        # Retrieve candidate pools
        candidate_pool_size = max(top_k * 3, 20)
        dense_results: List[Dict[str, Any]] = []
        if query_embedding is not None:
            dense_results = self.search(query_embedding=query_embedding, top_k=candidate_pool_size)

        tfidf_results: List[Dict[str, Any]] = []
        if cleaned_query:
            tfidf_results = self.search_tfidf(query=cleaned_query, top_k=candidate_pool_size)

        if not dense_results and not tfidf_results:
            return []

        # Single retrieval path handling
        if not tfidf_results and dense_results:
            results: List[Dict[str, Any]] = []
            for res in dense_results[:top_k]:
                item = copy.deepcopy(res)
                item["metadata"]["retrieval_source"] = "dense"
                item["metadata"]["dense_score"] = float(item["similarity_score"])
                results.append(item)
            return results

        if not dense_results and tfidf_results:
            results = []
            for res in tfidf_results[:top_k]:
                item = copy.deepcopy(res)
                item["metadata"]["retrieval_source"] = "tfidf"
                item["metadata"]["tfidf_score"] = float(item["similarity_score"])
                results.append(item)
            return results

        # Reciprocal Rank Fusion (RRF)
        dense_map = {item["chunk_id"]: (rank, item) for rank, item in enumerate(dense_results, 1)}
        tfidf_map = {item["chunk_id"]: (rank, item) for rank, item in enumerate(tfidf_results, 1)}

        all_chunk_ids = list(dict.fromkeys(
            [it["chunk_id"] for it in dense_results] + [it["chunk_id"] for it in tfidf_results]
        ))

        max_possible_rrf = (dense_weight / (rrf_k + 1)) + (tfidf_weight / (rrf_k + 1))

        scored_candidates: List[Dict[str, Any]] = []
        for cid in all_chunk_ids:
            dense_entry = dense_map.get(cid)
            tfidf_entry = tfidf_map.get(cid)

            score = 0.0
            if dense_entry is not None:
                dense_rank = dense_entry[0]
                score += dense_weight * (1.0 / (rrf_k + dense_rank))
            if tfidf_entry is not None:
                tfidf_rank = tfidf_entry[0]
                score += tfidf_weight * (1.0 / (rrf_k + tfidf_rank))

            normalized_score = (score / max_possible_rrf) if max_possible_rrf > 0 else 0.0
            normalized_score = min(1.0, max(0.0, normalized_score))

            base_item = dense_entry[1] if dense_entry is not None else tfidf_entry[1]
            meta = copy.deepcopy(base_item["metadata"])
            meta["retrieval_source"] = "hybrid" if (dense_entry and tfidf_entry) else ("dense" if dense_entry else "tfidf")
            if dense_entry is not None:
                meta["dense_score"] = float(dense_entry[1]["similarity_score"])
                meta["dense_rank"] = dense_entry[0]
            if tfidf_entry is not None:
                meta["tfidf_score"] = float(tfidf_entry[1]["similarity_score"])
                meta["tfidf_rank"] = tfidf_entry[0]

            scored_candidates.append({
                "chunk_id": cid,
                "text": base_item["text"],
                "metadata": meta,
                "similarity_score": round(float(normalized_score), 6),
            })

        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scored_candidates[:top_k]

    def as_retriever(
        self,
        embedding_service: Any,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        retrieval_mode: str = "hybrid",
        dense_weight: float = 0.6,
        tfidf_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> "FAISSRetrieverAdapter":
        """Return a LangChain-compatible BaseRetriever adapter for this vector store."""
        return FAISSRetrieverAdapter(
            vector_store=self,
            embedding_service=embedding_service,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            retrieval_mode=retrieval_mode,
            dense_weight=dense_weight,
            tfidf_weight=tfidf_weight,
            rrf_k=rrf_k,
        )


# ============================================================================
# LangChain Retriever Adapter Implementation
# ============================================================================

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.retrievers import BaseRetriever as BaseLangChainRetriever
except ImportError:
    class LangChainDocument:  # type: ignore[no-redef]
        def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

    class BaseLangChainRetriever:  # type: ignore[no-redef]
        pass
    CallbackManagerForRetrieverRun = Any  # type: ignore[assignment,misc]


class FAISSRetrieverAdapter(BaseLangChainRetriever):
    """LangChain-compatible BaseRetriever adapter delegating to FAISSVectorStore."""

    vector_store: Any
    embedding_service: Any
    top_k: int = 5
    similarity_threshold: Optional[float] = None
    retrieval_mode: str = "hybrid"
    dense_weight: float = 0.6
    tfidf_weight: float = 0.4
    rrf_k: int = 60

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[LangChainDocument]:
        """Retrieve documents relevant to the query from the underlying FAISSVectorStore."""
        if self.vector_store.is_empty:
            return []

        if self.retrieval_mode == "hybrid" and hasattr(self.vector_store, "hybrid_search"):
            query_vec = None
            if self.embedding_service is not None:
                try:
                    query_vec = self.embedding_service.embed_text(query)
                except Exception:
                    query_vec = None
            raw_chunks = self.vector_store.hybrid_search(
                query_text=query,
                query_embedding=query_vec,
                top_k=self.top_k,
                dense_weight=self.dense_weight,
                tfidf_weight=self.tfidf_weight,
                rrf_k=self.rrf_k,
            )
        elif self.retrieval_mode == "tfidf" and hasattr(self.vector_store, "search_tfidf"):
            raw_chunks = self.vector_store.search_tfidf(query=query, top_k=self.top_k)
        else:
            query_vec = self.embedding_service.embed_text(query)
            raw_chunks = self.vector_store.search(query_vec, top_k=self.top_k)

        if self.similarity_threshold is not None:
            raw_chunks = [
                c for c in raw_chunks
                if c.get("similarity_score", -1.0) >= self.similarity_threshold
            ]

        docs: List[LangChainDocument] = []
        for c in raw_chunks:
            meta = copy.deepcopy(c.get("metadata", {}))
            meta["chunk_id"] = c.get("chunk_id")
            meta["similarity_score"] = c.get("similarity_score")
            docs.append(LangChainDocument(
                page_content=c.get("text", ""),
                metadata=meta,
            ))
        return docs


def create_vector_store(
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    normalize_vectors: bool = True,
) -> FAISSVectorStore:
    """Create and return an initialized FAISSVectorStore instance."""
    return FAISSVectorStore(dimension=dimension, normalize_vectors=normalize_vectors)
