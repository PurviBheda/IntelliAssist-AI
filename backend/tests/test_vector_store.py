"""Automated unit tests for IntelliAssist AI FAISS Vector Store.

Tests coverage:
  1. Empty index initialization.
  2. Adding one chunk.
  3. Adding multiple chunks.
  4. Correct total vector count after multiple additions.
  5. Searching for the nearest vector.
  6. Correct descending similarity ranking.
  7. Returning the correct chunk ID.
  8. Returning the correct text.
  9. Preserving filename metadata.
  10. Preserving page numbers.
  11. Searching across multiple documents.
  12. Searching an empty index returns empty list safely.
  13. top_k larger than the number of stored vectors.
  14. Invalid top_k values (0, negative, non-integer, boolean).
  15. Wrong embedding dimensions are rejected.
  16. Mismatched chunk and embedding counts are rejected.
  17. Invalid chunk records are rejected.
  18. Duplicate chunk IDs are rejected (within batch and across batches).
  19. NaN and infinite vector values are rejected.
  20. Invalid query embeddings are rejected.
  21. Clearing the index resets index and metadata mappings.
  22. Adding chunks after clearing functions correctly.
  23. Normalization policy behavior (automatic vs explicit validation).
  24. Atomic batch insertion prevents partial data corruption on failure.
  25. Embedded chunks format (embeddings passed within chunk dicts).
  26. Factory helper create_vector_store.

All tests use deterministic synthetic vectors with real FAISS operations
and require zero model downloads and no internet connectivity.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.vector_store import (
    DEFAULT_EMBEDDING_DIMENSION,
    FAISSVectorStore,
    create_vector_store,
)


def _make_unit_vector(dim: int, non_zero_index: int) -> np.ndarray:
    """Create a unit basis vector with 1.0 at non_zero_index and 0 elsewhere."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[non_zero_index] = 1.0
    return vec


def _make_deterministic_vector(dim: int, seed: int) -> np.ndarray:
    """Create a deterministic unit-normalized dense float32 vector."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


# ============================================================================
# 1. Empty index initialization
# ============================================================================


def test_empty_index_initialization() -> None:
    """1. Empty vector store initializes with 0 vectors and correct dimension."""
    store = FAISSVectorStore(dimension=384)

    assert store.dimension == 384
    assert store.total_vectors == 0
    assert len(store) == 0
    assert store.is_empty is True
    assert store.normalize_vectors is True

    # Test invalid dimension values
    with pytest.raises(ValueError, match="positive integer"):
        FAISSVectorStore(dimension=0)
    with pytest.raises(ValueError, match="positive integer"):
        FAISSVectorStore(dimension=-10)
    with pytest.raises(ValueError, match="positive integer"):
        FAISSVectorStore(dimension="384")  # type: ignore
    with pytest.raises(ValueError, match="positive integer"):
        FAISSVectorStore(dimension=True)  # type: ignore


# ============================================================================
# 2. Adding one chunk
# ============================================================================


def test_add_single_chunk() -> None:
    """2. Adding a single valid chunk updates count and preserves index state."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {
            "chunk_id": "doc1.pdf_page_1_chunk_1",
            "text": "First chunk text about AI architectures.",
            "metadata": {
                "filename": "doc1.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        }
    ]
    embeddings = np.array([_make_unit_vector(384, 0)])

    count = store.add_chunks(chunks, embeddings)

    assert count == 1
    assert store.total_vectors == 1
    assert len(store) == 1
    assert store.is_empty is False
    assert store.has_chunk("doc1.pdf_page_1_chunk_1") is True


# ============================================================================
# 3. Adding multiple chunks
# ============================================================================


def test_add_multiple_chunks() -> None:
    """3. Adding multiple chunks in one batch correctly stores all records."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {
            "chunk_id": f"manual.pdf_page_1_chunk_{i}",
            "text": f"Instruction paragraph {i}",
            "metadata": {"filename": "manual.pdf", "page_number": 1, "chunk_index": i},
        }
        for i in range(1, 4)
    ]
    embeddings = np.vstack([_make_unit_vector(384, i) for i in range(3)])

    count = store.add_chunks(chunks, embeddings)

    assert count == 3
    assert store.total_vectors == 3
    for i in range(1, 4):
        assert store.has_chunk(f"manual.pdf_page_1_chunk_{i}") is True


# ============================================================================
# 4. Correct total vector count after multiple batches
# ============================================================================


def test_total_vector_count_after_multiple_adds() -> None:
    """4. Total vector count increments accurately across multiple sequential adds."""
    store = FAISSVectorStore(dimension=384)

    # Batch 1 (2 chunks)
    chunks_1 = [
        {
            "chunk_id": f"batch1_chunk_{i}",
            "text": f"Batch 1 text {i}",
            "metadata": {"filename": "batch1.pdf", "page_number": 1, "chunk_index": i},
        }
        for i in range(2)
    ]
    vecs_1 = np.vstack([_make_unit_vector(384, i) for i in range(2)])
    store.add_chunks(chunks_1, vecs_1)
    assert store.total_vectors == 2

    # Batch 2 (3 chunks)
    chunks_2 = [
        {
            "chunk_id": f"batch2_chunk_{i}",
            "text": f"Batch 2 text {i}",
            "metadata": {"filename": "batch2.pdf", "page_number": 1, "chunk_index": i},
        }
        for i in range(3)
    ]
    vecs_2 = np.vstack([_make_unit_vector(384, i + 2) for i in range(3)])
    store.add_chunks(chunks_2, vecs_2)

    assert store.total_vectors == 5
    assert len(store) == 5


# ============================================================================
# 5. Searching for the nearest vector
# ============================================================================


def test_search_nearest_vector() -> None:
    """5. Nearest vector search identifies the closest embedding with high similarity."""
    store = FAISSVectorStore(dimension=384)
    v0 = _make_unit_vector(384, 0)
    v1 = _make_unit_vector(384, 1)
    v2 = _make_unit_vector(384, 2)

    chunks = [
        {"chunk_id": "c0", "text": "Chunk zero", "metadata": {"page": 1}},
        {"chunk_id": "c1", "text": "Chunk one", "metadata": {"page": 2}},
        {"chunk_id": "c2", "text": "Chunk two", "metadata": {"page": 3}},
    ]
    store.add_chunks(chunks, np.vstack([v0, v1, v2]))

    # Search with exact vector v1
    results = store.search(v1, top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["text"] == "Chunk one"
    assert pytest.approx(results[0]["similarity_score"], abs=1e-5) == 1.0


# ============================================================================
# 6. Correct descending similarity ranking
# ============================================================================


def test_search_descending_similarity_ranking() -> None:
    """6. Results are strictly ranked in descending order of similarity."""
    store = FAISSVectorStore(dimension=4)

    # Orthogonal basis vectors e0, e1, e2, e3
    chunks = [
        {"chunk_id": "c0", "text": "Text 0", "metadata": {"idx": 0}},
        {"chunk_id": "c1", "text": "Text 1", "metadata": {"idx": 1}},
        {"chunk_id": "c2", "text": "Text 2", "metadata": {"idx": 2}},
        {"chunk_id": "c3", "text": "Text 3", "metadata": {"idx": 3}},
    ]
    embeddings = np.eye(4, dtype=np.float32)
    store.add_chunks(chunks, embeddings)

    # Query composed of 0.8*e0 + 0.5*e1 + 0.2*e2 + 0.0*e3
    query = np.array([0.8, 0.5, 0.2, 0.0], dtype=np.float32)
    results = store.search(query, top_k=4)

    assert len(results) == 4
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert results[0]["chunk_id"] == "c0"
    assert results[1]["chunk_id"] == "c1"
    assert results[2]["chunk_id"] == "c2"
    assert results[3]["chunk_id"] == "c3"


# ============================================================================
# 7. Returning the correct chunk ID
# ============================================================================


def test_search_returns_correct_chunk_id() -> None:
    """7. Search returns the exact chunk_id associated with the matched vector."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {"chunk_id": "target_id_alpha", "text": "Alpha text", "metadata": {"tag": "a"}},
        {"chunk_id": "target_id_beta", "text": "Beta text", "metadata": {"tag": "b"}},
    ]
    embeddings = np.vstack([_make_unit_vector(384, 10), _make_unit_vector(384, 20)])
    store.add_chunks(chunks, embeddings)

    res = store.search(_make_unit_vector(384, 20), top_k=1)
    assert res[0]["chunk_id"] == "target_id_beta"


# ============================================================================
# 8. Returning the correct text
# ============================================================================


def test_search_returns_correct_text() -> None:
    """8. Search result preserves the full text corresponding to the stored chunk."""
    store = FAISSVectorStore(dimension=384)
    expected_text = "Detailed cybersecurity guideline on multi-factor authentication."
    chunks = [
        {"chunk_id": "sec_1", "text": expected_text, "metadata": {"topic": "security"}},
        {"chunk_id": "hr_1", "text": "Vacation policy details.", "metadata": {"topic": "hr"}},
    ]
    embeddings = np.vstack([_make_unit_vector(384, 5), _make_unit_vector(384, 6)])
    store.add_chunks(chunks, embeddings)

    res = store.search(_make_unit_vector(384, 5), top_k=1)
    assert res[0]["text"] == expected_text


# ============================================================================
# 9. Preserving filename metadata
# ============================================================================


def test_search_preserves_filename_metadata() -> None:
    """9. Search results retain the original source document filename in metadata."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {
            "chunk_id": "whitepaper.pdf_p1_c1",
            "text": "Enterprise cloud security review.",
            "metadata": {"filename": "whitepaper.pdf", "file_type": "pdf", "page_number": 1},
        }
    ]
    store.add_chunks(chunks, np.array([_make_unit_vector(384, 1)]))

    res = store.search(_make_unit_vector(384, 1), top_k=1)
    assert res[0]["metadata"]["filename"] == "whitepaper.pdf"
    assert res[0]["metadata"]["file_type"] == "pdf"


# ============================================================================
# 10. Preserving page numbers
# ============================================================================


def test_search_preserves_page_numbers() -> None:
    """10. Search results retain the original physical/logical page number."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {
            "chunk_id": "guide.pdf_p42_c1",
            "text": "Specific instructions on page forty-two.",
            "metadata": {"filename": "guide.pdf", "page_number": 42, "chunk_index": 1},
        }
    ]
    store.add_chunks(chunks, np.array([_make_unit_vector(384, 7)]))

    res = store.search(_make_unit_vector(384, 7), top_k=1)
    assert res[0]["metadata"]["page_number"] == 42


# ============================================================================
# 11. Searching across multiple documents
# ============================================================================


def test_search_across_multiple_documents() -> None:
    """11. A single vector store accurately indexes and retrieves chunks from multiple files."""
    store = FAISSVectorStore(dimension=384)

    # Document A: handbook.pdf (pages 1 and 2)
    doc_a_chunks = [
        {
            "chunk_id": "handbook.pdf_page_1_chunk_1",
            "text": "Employee onboarding and orientation overview.",
            "metadata": {"filename": "handbook.pdf", "page_number": 1, "chunk_index": 1},
        },
        {
            "chunk_id": "handbook.pdf_page_2_chunk_1",
            "text": "Working hours, lunch breaks, and remote work policies.",
            "metadata": {"filename": "handbook.pdf", "page_number": 2, "chunk_index": 1},
        },
    ]
    doc_a_vecs = np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)])
    store.add_chunks(doc_a_chunks, doc_a_vecs)

    # Document B: policies.pdf (pages 1 and 3)
    doc_b_chunks = [
        {
            "chunk_id": "policies.pdf_page_1_chunk_1",
            "text": "Information security and acceptable equipment usage.",
            "metadata": {"filename": "policies.pdf", "page_number": 1, "chunk_index": 1},
        },
        {
            "chunk_id": "policies.pdf_page_3_chunk_1",
            "text": "Incident reporting procedure for suspected malware or breaches.",
            "metadata": {"filename": "policies.pdf", "page_number": 3, "chunk_index": 1},
        },
    ]
    doc_b_vecs = np.vstack([_make_unit_vector(384, 2), _make_unit_vector(384, 3)])
    store.add_chunks(doc_b_chunks, doc_b_vecs)

    assert store.total_vectors == 4

    # Query closest to Document B, page 3
    results = store.search(_make_unit_vector(384, 3), top_k=4)

    assert len(results) == 4
    # Top result must be policies.pdf page 3
    assert results[0]["chunk_id"] == "policies.pdf_page_3_chunk_1"
    assert results[0]["metadata"]["filename"] == "policies.pdf"
    assert results[0]["metadata"]["page_number"] == 3

    # All source documents are preserved and accessible
    filenames = {r["metadata"]["filename"] for r in results}
    assert filenames == {"handbook.pdf", "policies.pdf"}


# ============================================================================
# 12. Searching an empty index
# ============================================================================


def test_search_empty_index_returns_empty_list() -> None:
    """12. Searching an empty index returns [] safely without crashing or raising errors."""
    store = FAISSVectorStore(dimension=384)
    query = _make_unit_vector(384, 0)

    results = store.search(query, top_k=5)

    assert results == []
    assert isinstance(results, list)


# ============================================================================
# 13. top_k larger than the number of stored vectors
# ============================================================================


def test_top_k_larger_than_total_vectors() -> None:
    """13. When top_k exceeds total vectors, only available records are returned."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {"chunk_id": "c1", "text": "Text 1", "metadata": {"page": 1}},
        {"chunk_id": "c2", "text": "Text 2", "metadata": {"page": 2}},
    ]
    embeddings = np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)])
    store.add_chunks(chunks, embeddings)

    # Request top_k=10 on a store with only 2 vectors
    results = store.search(_make_unit_vector(384, 0), top_k=10)

    assert len(results) == 2
    # Ensure no FAISS -1 sentinel IDs are leaked
    for r in results:
        assert r["chunk_id"] in {"c1", "c2"}


# ============================================================================
# 14. Invalid top_k values
# ============================================================================


def test_invalid_top_k_values_rejected() -> None:
    """14. Non-positive, non-integer, and boolean top_k values raise ValueError."""
    store = FAISSVectorStore(dimension=384)
    query = _make_unit_vector(384, 0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search(query, top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search(query, top_k=-3)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search(query, top_k="5")  # type: ignore

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search(query, top_k=2.5)  # type: ignore

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        store.search(query, top_k=True)  # type: ignore


# ============================================================================
# 15. Wrong embedding dimensions
# ============================================================================


def test_wrong_embedding_dimensions_rejected() -> None:
    """15. Adding vectors with dimensionality different from store dimension is rejected."""
    store = FAISSVectorStore(dimension=384)
    chunks = [{"chunk_id": "c1", "text": "Sample text", "metadata": {}}]

    # Vector with 128 dimensions instead of 384
    wrong_dim_vecs = np.zeros((1, 128), dtype=np.float32)
    wrong_dim_vecs[0, 0] = 1.0

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add_chunks(chunks, wrong_dim_vecs)

    # Store remains empty
    assert store.is_empty is True


# ============================================================================
# 16. Mismatched chunk and embedding counts
# ============================================================================


def test_mismatched_chunk_and_embedding_counts_rejected() -> None:
    """16. Mismatched chunk list length and embedding row count raises ValueError."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {"chunk_id": "c1", "text": "T1", "metadata": {}},
        {"chunk_id": "c2", "text": "T2", "metadata": {}},
        {"chunk_id": "c3", "text": "T3", "metadata": {}},
    ]
    # Only 2 vectors for 3 chunks
    embeddings = np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)])

    with pytest.raises(ValueError, match="Mismatched counts"):
        store.add_chunks(chunks, embeddings)

    assert store.is_empty is True


# ============================================================================
# 17. Invalid chunk records
# ============================================================================


def test_invalid_chunk_records_rejected() -> None:
    """17. Malformed chunk records (missing/empty keys, invalid types) are rejected."""
    store = FAISSVectorStore(dimension=384)
    vec = np.array([_make_unit_vector(384, 0)])

    # None chunks
    with pytest.raises(ValueError, match="chunks list cannot be None"):
        store.add_chunks(None, vec)  # type: ignore

    # Non-list chunks
    with pytest.raises(TypeError, match="must be a list"):
        store.add_chunks({"not": "a list"}, vec)  # type: ignore

    # Non-dict chunk item
    with pytest.raises(ValueError, match="must be a dictionary"):
        store.add_chunks(["not a dict"], vec)  # type: ignore

    # Missing chunk_id
    with pytest.raises(ValueError, match="missing required 'chunk_id'"):
        store.add_chunks([{"text": "hi", "metadata": {}}], vec)

    # Empty chunk_id
    with pytest.raises(ValueError, match="empty or whitespace-only 'chunk_id'"):
        store.add_chunks([{"chunk_id": "   ", "text": "hi", "metadata": {}}], vec)

    # Non-string chunk_id
    with pytest.raises(TypeError, match="'chunk_id' must be a string"):
        store.add_chunks([{"chunk_id": 12345, "text": "hi", "metadata": {}}], vec)  # type: ignore

    # Missing text
    with pytest.raises(ValueError, match="missing required 'text'"):
        store.add_chunks([{"chunk_id": "c1", "metadata": {}}], vec)

    # Non-string text
    with pytest.raises(TypeError, match="'text' must be a string"):
        store.add_chunks([{"chunk_id": "c1", "text": 42, "metadata": {}}], vec)  # type: ignore

    # Missing metadata
    with pytest.raises(ValueError, match="missing required 'metadata'"):
        store.add_chunks([{"chunk_id": "c1", "text": "valid text"}], vec)

    # Non-dict metadata
    with pytest.raises(TypeError, match="'metadata' must be a dict"):
        store.add_chunks([{"chunk_id": "c1", "text": "valid", "metadata": "not a dict"}], vec)  # type: ignore

    # Confirm index remains pristine
    assert store.is_empty is True


# ============================================================================
# 18. Duplicate chunk IDs
# ============================================================================


def test_duplicate_chunk_ids_rejected() -> None:
    """18. Duplicate chunk IDs within batch or across batches are rejected."""
    store = FAISSVectorStore(dimension=384)

    # Duplicate within same batch
    dup_batch = [
        {"chunk_id": "dup_1", "text": "First", "metadata": {}},
        {"chunk_id": "dup_1", "text": "Second", "metadata": {}},
    ]
    vecs = np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)])
    with pytest.raises(ValueError, match="Duplicate chunk_id detected within batch"):
        store.add_chunks(dup_batch, vecs)
    assert store.is_empty is True

    # Add initial chunk successfully
    store.add_chunks(
        [{"chunk_id": "existing_id", "text": "Initial", "metadata": {}}],
        np.array([_make_unit_vector(384, 0)]),
    )
    assert store.total_vectors == 1

    # Attempt to add colliding chunk ID in a new batch
    with pytest.raises(ValueError, match="Duplicate chunk_id detected: 'existing_id' already exists"):
        store.add_chunks(
            [{"chunk_id": "existing_id", "text": "Colliding", "metadata": {}}],
            np.array([_make_unit_vector(384, 1)]),
        )

    # Verify original state preserved
    assert store.total_vectors == 1


# ============================================================================
# 19. NaN and infinite vector values
# ============================================================================


def test_nan_and_infinite_vectors_rejected() -> None:
    """19. Embeddings with NaN, Inf, or -Inf are rejected with clear errors."""
    store = FAISSVectorStore(dimension=384)
    chunks = [{"chunk_id": "c1", "text": "T1", "metadata": {}}]

    # NaN vector
    nan_vec = np.zeros((1, 384), dtype=np.float32)
    nan_vec[0, 10] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.add_chunks(chunks, nan_vec)

    # Positive Inf vector
    inf_vec = np.zeros((1, 384), dtype=np.float32)
    inf_vec[0, 20] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.add_chunks(chunks, inf_vec)

    # Negative Inf vector
    ninf_vec = np.zeros((1, 384), dtype=np.float32)
    ninf_vec[0, 30] = -np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.add_chunks(chunks, ninf_vec)

    assert store.is_empty is True


# ============================================================================
# 20. Invalid query embeddings
# ============================================================================


def test_invalid_query_embeddings_rejected() -> None:
    """20. Malformed query embeddings (wrong dimension, NaN/Inf, zero magnitude) are rejected."""
    store = FAISSVectorStore(dimension=384)
    store.add_chunks(
        [{"chunk_id": "c1", "text": "Valid", "metadata": {}}],
        np.array([_make_unit_vector(384, 0)]),
    )

    # None query
    with pytest.raises(ValueError, match="query_embedding cannot be None"):
        store.search(None)  # type: ignore

    # Wrong dimension
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.search(np.zeros(128, dtype=np.float32))

    # Query with NaN
    nan_query = _make_unit_vector(384, 0)
    nan_query[5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.search(nan_query)

    # Query with Inf
    inf_query = _make_unit_vector(384, 0)
    inf_query[5] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        store.search(inf_query)

    # Zero magnitude query
    with pytest.raises(ValueError, match="Zero-magnitude query embedding"):
        store.search(np.zeros(384, dtype=np.float32))

    # 3D array query
    with pytest.raises(ValueError, match="must be 1D"):
        store.search(np.zeros((1, 1, 384), dtype=np.float32))


# ============================================================================
# 21. Clearing the index
# ============================================================================


def test_clear_index_resets_store_and_metadata() -> None:
    """21. clear() flushes the FAISS index, records, and chunk_ids mapping."""
    store = FAISSVectorStore(dimension=384)
    chunks = [
        {"chunk_id": "c1", "text": "T1", "metadata": {"page": 1}},
        {"chunk_id": "c2", "text": "T2", "metadata": {"page": 2}},
    ]
    store.add_chunks(chunks, np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)]))

    assert store.total_vectors == 2
    assert store.has_chunk("c1") is True

    store.clear()

    assert store.total_vectors == 0
    assert len(store) == 0
    assert store.is_empty is True
    assert store.has_chunk("c1") is False
    assert store.get_chunk("c1") is None
    assert store.search(_make_unit_vector(384, 0)) == []


# ============================================================================
# 22. Adding chunks after clearing
# ============================================================================


def test_add_chunks_after_clear() -> None:
    """22. Vector store functions properly when adding new chunks after clear()."""
    store = FAISSVectorStore(dimension=384)
    old_chunks = [{"chunk_id": "old_1", "text": "Old text", "metadata": {}}]
    store.add_chunks(old_chunks, np.array([_make_unit_vector(384, 0)]))
    store.clear()

    # Re-insert new chunk
    new_chunks = [{"chunk_id": "new_1", "text": "New text", "metadata": {"page": 1}}]
    store.add_chunks(new_chunks, np.array([_make_unit_vector(384, 5)]))

    assert store.total_vectors == 1
    assert store.has_chunk("new_1") is True
    assert store.has_chunk("old_1") is False

    results = store.search(_make_unit_vector(384, 5), top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "new_1"
    assert results[0]["text"] == "New text"


# ============================================================================
# 23. Normalization policy
# ============================================================================


def test_normalization_policy() -> None:
    """23. Normalization policy validates unnormalized vs normalized vector handling."""
    # A. Default store: normalize_vectors=True
    store_auto = FAISSVectorStore(dimension=4, normalize_vectors=True)
    # Pass unnormalized vector with length 5.0
    unnorm_vec = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    chunks = [{"chunk_id": "unnorm_1", "text": "Vector length 5", "metadata": {}}]
    store_auto.add_chunks(chunks, unnorm_vec)

    # Search with unit query [0.6, 0.8, 0.0, 0.0]
    query = np.array([0.6, 0.8, 0.0, 0.0], dtype=np.float32)
    res = store_auto.search(query, top_k=1)
    # Cosine similarity must be ~1.0
    assert pytest.approx(res[0]["similarity_score"], abs=1e-5) == 1.0

    # B. Strict store: normalize_vectors=False
    store_strict = FAISSVectorStore(dimension=4, normalize_vectors=False)
    # Reject unnormalized input
    with pytest.raises(ValueError, match="Embeddings are not L2-normalized"):
        store_strict.add_chunks(chunks, unnorm_vec)

    # Accept valid unit-normalized input
    norm_vec = np.array([[0.6, 0.8, 0.0, 0.0]], dtype=np.float32)
    store_strict.add_chunks(chunks, norm_vec)
    assert store_strict.total_vectors == 1

    # Search with non-normalized query on strict store
    with pytest.raises(ValueError, match="Query embedding is not L2-normalized"):
        store_strict.search(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))


# ============================================================================
# 24. Atomic batch insertion prevents partial data corruption
# ============================================================================


def test_failed_batch_insertion_atomicity() -> None:
    """24. An invalid batch does not leave the vector store in a partially modified state."""
    store = FAISSVectorStore(dimension=384)

    # Add 2 initial valid chunks
    initial_chunks = [
        {"chunk_id": "init_1", "text": "Initial 1", "metadata": {}},
        {"chunk_id": "init_2", "text": "Initial 2", "metadata": {}},
    ]
    initial_vecs = np.vstack([_make_unit_vector(384, 0), _make_unit_vector(384, 1)])
    store.add_chunks(initial_chunks, initial_vecs)
    assert store.total_vectors == 2

    # Attempt to add batch of 2: first item valid, second item duplicate of init_1
    corrupt_batch = [
        {"chunk_id": "valid_candidate", "text": "Candidate", "metadata": {}},
        {"chunk_id": "init_1", "text": "Duplicate ID", "metadata": {}},
    ]
    corrupt_vecs = np.vstack([_make_unit_vector(384, 2), _make_unit_vector(384, 3)])

    with pytest.raises(ValueError, match="Duplicate chunk_id detected"):
        store.add_chunks(corrupt_batch, corrupt_vecs)

    # Verify no partial insertion occurred: candidate was NOT stored
    assert store.total_vectors == 2
    assert store.has_chunk("valid_candidate") is False
    assert len(store._records) == 2


# ============================================================================
# 25. Embedded chunks format (embeddings passed within chunk dicts)
# ============================================================================


def test_embedded_chunks_format_support() -> None:
    """25. Chunks produced by EmbeddingService.embed_chunks() can be added directly."""
    store = FAISSVectorStore(dimension=384)
    embedded_chunks = [
        {
            "chunk_id": "embedded_chunk_1",
            "text": "Embedded chunk text",
            "metadata": {"filename": "doc.pdf", "page_number": 1},
            "embedding": _make_unit_vector(384, 0),
        }
    ]

    # Call add_chunks with embeddings=None
    count = store.add_chunks(embedded_chunks)

    assert count == 1
    assert store.total_vectors == 1
    assert store.has_chunk("embedded_chunk_1") is True


# ============================================================================
# 26. Factory function create_vector_store
# ============================================================================


def test_create_vector_store_factory() -> None:
    """26. create_vector_store initializes FAISSVectorStore with correct defaults."""
    store = create_vector_store()
    assert isinstance(store, FAISSVectorStore)
    assert store.dimension == DEFAULT_EMBEDDING_DIMENSION
    assert store.is_empty is True
