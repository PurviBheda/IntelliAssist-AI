"""Automated unit tests for IntelliAssist AI Text Embeddings Module.

Tests coverage:
  1. Empty text list returns the expected empty array shape without loading the model.
  2. Invalid inputs are rejected.
  3. The model is initialized lazily.
  4. Model initialization is reused instead of repeated unnecessarily.
  5. A single text produces a one-dimensional vector.
  6. Multiple texts produce a two-dimensional array.
  7. All embeddings have consistent dimensions.
  8. The default model produces 384-dimensional vectors.
  9. Normalization behavior is correct if normalization is enabled.
  10. The chunk embedding helper preserves chunk IDs and metadata.
  11. Model-loading failures produce useful errors.

All unit tests use mock models to ensure fast, deterministic execution without
requiring internet access or downloading weights during automated testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.embedding_service import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_MODEL_NAME,
    EmbeddingService,
    embed_chunks,
    embed_text,
    embed_texts,
)


def _create_mock_st_model(dim: int = DEFAULT_EMBEDDING_DIMENSION) -> MagicMock:
    """Create a mock SentenceTransformer instance producing valid vectors."""
    mock_model = MagicMock()

    def mock_encode(
        sentences,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        if isinstance(sentences, str):
            rng = np.random.default_rng(seed=abs(hash(sentences)) % (2**32))
            vec = rng.standard_normal(dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec
        elif isinstance(sentences, list):
            rows = []
            for s in sentences:
                rng = np.random.default_rng(seed=abs(hash(s)) % (2**32))
                vec = rng.standard_normal(dim).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                rows.append(vec)
            if not rows:
                return np.empty((0, dim), dtype=np.float32)
            return np.vstack(rows)

    mock_model.encode.side_effect = mock_encode
    return mock_model


# ============================================================================
# 1. Empty text list returns (0, 384) without loading model
# ============================================================================


def test_empty_text_list_returns_expected_shape_without_loading_model() -> None:
    """1. Empty text list returns (0, 384) without initializing the model."""
    service = EmbeddingService()
    assert not service.is_model_loaded

    result = service.embed_texts([])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, DEFAULT_EMBEDDING_DIMENSION)
    assert result.dtype == np.float32
    assert not service.is_model_loaded


# ============================================================================
# 2. Invalid inputs are rejected
# ============================================================================


def test_invalid_single_text_rejected() -> None:
    """2a. Invalid single text inputs raise appropriate exceptions."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    # None input
    with pytest.raises(ValueError, match="cannot be None"):
        service.embed_text(None)  # type: ignore

    # Non-string input
    with pytest.raises(TypeError, match="must be a string"):
        service.embed_text(12345)  # type: ignore

    # Empty string
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.embed_text("")

    # Whitespace only
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.embed_text("   \n\t  ")


def test_invalid_texts_list_rejected() -> None:
    """2b. Invalid texts list inputs raise appropriate exceptions."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    # None texts list
    with pytest.raises(ValueError, match="cannot be None"):
        service.embed_texts(None)  # type: ignore

    # Non-list input
    with pytest.raises(TypeError, match="must be a list"):
        service.embed_texts("just a string")  # type: ignore

    # List containing None
    with pytest.raises(ValueError, match="cannot be None"):
        service.embed_texts(["Valid text", None])  # type: ignore

    # List containing non-string
    with pytest.raises(TypeError, match="must be a string"):
        service.embed_texts(["Valid text", 42])  # type: ignore

    # List containing whitespace-only text
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.embed_texts(["Valid text", "   "])

    # Invalid batch_size
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.embed_texts(["Valid text"], batch_size=0)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        service.embed_texts(["Valid text"], batch_size=-5)


# ============================================================================
# 3. Model is initialized lazily
# ============================================================================


def test_model_is_initialized_lazily() -> None:
    """3. Model is not loaded on instantiation, but loads on first embedding call."""
    mock_model = _create_mock_st_model()

    with patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_model
    ) as mock_st_cls:
        service = EmbeddingService()
        assert not service.is_model_loaded
        mock_st_cls.assert_not_called()

        # First call triggers lazy loading
        vec = service.embed_text("Lazy loading test")
        assert service.is_model_loaded
        mock_st_cls.assert_called_once_with(DEFAULT_MODEL_NAME)
        assert vec.shape == (DEFAULT_EMBEDDING_DIMENSION,)


# ============================================================================
# 4. Model initialization is reused instead of repeated
# ============================================================================


def test_model_initialization_is_reused() -> None:
    """4. Repeated calls to embed reuse the same loaded model instance."""
    mock_model = _create_mock_st_model()

    with patch(
        "sentence_transformers.SentenceTransformer", return_value=mock_model
    ) as mock_st_cls:
        service = EmbeddingService()

        service.embed_text("First query")
        service.embed_text("Second query")
        service.embed_texts(["Batch item 1", "Batch item 2"])

        # Model constructor should be called exactly once
        assert mock_st_cls.call_count == 1


# ============================================================================
# 5. Single text produces a 1D vector
# ============================================================================


def test_single_text_produces_one_dimensional_vector() -> None:
    """5. embed_text returns a 1D vector of shape (384,)."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    vec = service.embed_text("IntelliAssist AI automated vector representation.")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape == (384,)
    assert vec.dtype == np.float32


# ============================================================================
# 6. Multiple texts produce a 2D array
# ============================================================================


def test_multiple_texts_produce_two_dimensional_array() -> None:
    """6. embed_texts returns a 2D array of shape (N, 384)."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    texts = [
        "First document text segment",
        "Second document text segment",
        "Third document text segment",
    ]
    arr = service.embed_texts(texts, batch_size=2)
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.shape == (3, 384)
    assert arr.dtype == np.float32


# ============================================================================
# 7. All embeddings have consistent dimensions
# ============================================================================


def test_embeddings_have_consistent_dimensions() -> None:
    """7. Single and batch embeddings produce consistent vector dimensions."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    single_vec = service.embed_text("Sample paragraph")
    batch_vecs = service.embed_texts(["Batch sentence 1", "Batch sentence 2"])

    assert single_vec.shape[0] == batch_vecs.shape[1]
    assert single_vec.shape[0] == 384


# ============================================================================
# 8. Default model produces 384-dimensional vectors
# ============================================================================


def test_default_model_produces_384_dimensions() -> None:
    """8. Verifies default model configuration uses 384 dimensions."""
    service = EmbeddingService()
    assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert service.embedding_dimension == 384


# ============================================================================
# 9. Normalization behavior is correct if normalization enabled
# ============================================================================


def test_normalization_behavior() -> None:
    """9. L2 normalization produces vectors with Euclidean norm approximately 1.0."""
    service = EmbeddingService(normalize_embeddings=True)
    service._model = _create_mock_st_model()

    vec = service.embed_text("Cosine similarity test")
    norm = np.linalg.norm(vec)
    assert np.isclose(norm, 1.0, atol=1e-5)

    matrix = service.embed_texts(["Text A", "Text B", "Text C"])
    row_norms = np.linalg.norm(matrix, axis=1)
    for r_norm in row_norms:
        assert np.isclose(r_norm, 1.0, atol=1e-5)


# ============================================================================
# 10. Chunk embedding helper preserves chunk IDs and metadata
# ============================================================================


def test_chunk_embedding_helper_preserves_chunk_ids_and_metadata() -> None:
    """10. embed_chunks preserves original text, chunk_id, and metadata intact."""
    service = EmbeddingService()
    service._model = _create_mock_st_model()

    chunks = [
        {
            "chunk_id": "report.pdf_page_1_chunk_1",
            "text": "First page initial paragraph summary.",
            "metadata": {
                "filename": "report.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "report.pdf_page_2_chunk_1",
            "text": "Second page follow-up findings.",
            "metadata": {
                "filename": "report.pdf",
                "file_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        },
    ]

    embedded = service.embed_chunks(chunks)

    assert len(embedded) == 2
    for idx, item in enumerate(embedded):
        assert item["chunk_id"] == chunks[idx]["chunk_id"]
        assert item["text"] == chunks[idx]["text"]
        assert item["metadata"] == chunks[idx]["metadata"]
        assert isinstance(item["embedding"], np.ndarray)
        assert item["embedding"].shape == (384,)
        assert item["embedding"].dtype == np.float32

    # Verify empty chunks list
    assert service.embed_chunks([]) == []

    # Verify malformed chunk entries
    with pytest.raises(ValueError, match="missing 'chunk_id'"):
        service.embed_chunks([{"text": "abc", "metadata": {}}])


# ============================================================================
# 11. Model-loading failures produce useful errors
# ============================================================================


def test_model_loading_failures_produce_useful_errors() -> None:
    """11. Model-loading failures raise descriptive RuntimeError with context."""
    with patch(
        "sentence_transformers.SentenceTransformer",
        side_effect=OSError("Network connection refused during model download"),
    ):
        service = EmbeddingService()
        with pytest.raises(RuntimeError, match="Failed to load sentence-transformer model"):
            service.embed_text("Will fail to load")


# ============================================================================
# Convenience Functions (Module-level wrappers)
# ============================================================================


def test_module_level_convenience_functions() -> None:
    """Verifies that module-level embed_text, embed_texts, and embed_chunks work properly."""
    with patch(
        "app.embedding_service.get_default_embedding_service"
    ) as mock_get_svc:
        mock_svc = MagicMock()
        mock_get_svc.return_value = mock_svc

        embed_text("Query text")
        mock_svc.embed_text.assert_called_once_with("Query text")

        embed_texts(["Text 1", "Text 2"], batch_size=16)
        mock_svc.embed_texts.assert_called_once_with(["Text 1", "Text 2"], batch_size=16)

        embed_chunks([{"chunk_id": "1", "text": "a", "metadata": {}}], batch_size=16)
        mock_svc.embed_chunks.assert_called_once_with(
            [{"chunk_id": "1", "text": "a", "metadata": {}}], batch_size=16
        )
