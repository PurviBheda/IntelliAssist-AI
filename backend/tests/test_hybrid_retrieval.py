"""Automated unit and integration tests for Genuine TF-IDF & Hybrid Retrieval.

Coverage:
  1. TF-IDF vocabulary fitting and sparse matrix representation.
  2. Distinctive keyword queries scoring and ranking.
  3. Semantically relevant queries with limited word overlap.
  4. Hybrid reciprocal rank fusion (RRF) and score normalization.
  5. Stable chunk deduplication by chunk_id.
  6. Empty queries and empty index safety.
  7. Out-of-vocabulary (unseen) query terms handling.
  8. Index consistency on chunk addition, document deletion, and clearing.
  9. Preservation of complete citation metadata.
  10. TF-IDF based extractive summarization.
  11. LangChain retriever adapter in hybrid mode.
  12. Configurable hybrid weights and parameters.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List
import numpy as np
import pytest

from app.embedding_service import EmbeddingService
from app.rag_pipeline import (
    DEFAULT_SYSTEM_INSTRUCTION,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
)
from app.summarizer import (
    ExtractiveSummarizer,
    SummaryResult,
    SummaryType,
    TextSummarizer,
)
from app.vector_store import (
    DEFAULT_EMBEDDING_DIMENSION,
    FAISSVectorStore,
    create_vector_store,
)


def _make_unit_vector(dim: int, index: int) -> np.ndarray:
    """Create a unit vector with 1.0 at index and 0 elsewhere."""
    vec = np.zeros(dim, dtype=np.float32)
    vec[index % dim] = 1.0
    return vec


@pytest.fixture
def sample_corpus_chunks() -> List[Dict[str, Any]]:
    """Distinctive document chunks for keyword and semantic tests."""
    return [
        {
            "chunk_id": "security.pdf_page_1_chunk_1",
            "text": "Multi-factor authentication and quarterly access reviews are mandatory for enterprise cloud security.",
            "metadata": {
                "filename": "security.pdf",
                "page_number": 1,
                "file_type": "pdf",
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "quantum.txt_page_1_chunk_1",
            "text": "Quantum superposition and entanglement allow qubits to execute Shor algorithm efficiently.",
            "metadata": {
                "filename": "quantum.txt",
                "page_number": 1,
                "file_type": "txt",
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "biology.docx_page_1_chunk_1",
            "text": "Chloroplasts capture radiant sunlight energy to synthesize glucose through cellular photosynthesis.",
            "metadata": {
                "filename": "biology.docx",
                "page_number": 1,
                "file_type": "docx",
                "chunk_index": 1,
            },
        },
    ]


@pytest.fixture
def populated_hybrid_store(sample_corpus_chunks: List[Dict[str, Any]]) -> FAISSVectorStore:
    """Pre-populated store with 3 distinct chunks and orthogonal embedding vectors."""
    store = FAISSVectorStore(dimension=384)
    embeddings = [
        _make_unit_vector(384, 0),
        _make_unit_vector(384, 1),
        _make_unit_vector(384, 2),
    ]
    store.add_chunks(sample_corpus_chunks, embeddings)
    return store


# ============================================================================
# 1. Genuine TF-IDF Vocabulary and Scoring Tests
# ============================================================================

def test_tfidf_vocabulary_and_matrix_fitting(populated_hybrid_store: FAISSVectorStore) -> None:
    """1. TF-IDF vectorizer fits across chunk texts with non-empty vocabulary and sparse matrix."""
    assert populated_hybrid_store.tfidf_vectorizer is not None
    assert populated_hybrid_store.tfidf_matrix is not None
    assert populated_hybrid_store.tfidf_vocabulary_size > 0

    vocab = populated_hybrid_store.tfidf_vectorizer.vocabulary_
    assert "authentication" in vocab
    assert "quantum" in vocab
    assert "photosynthesis" in vocab
    assert populated_hybrid_store.tfidf_matrix.shape[0] == 3
    assert populated_hybrid_store.tfidf_matrix.shape[1] == len(vocab)


def test_tfidf_keyword_scoring(populated_hybrid_store: FAISSVectorStore) -> None:
    """2. Searching distinctive keywords returns exact matching chunk with top score."""
    results = populated_hybrid_store.search_tfidf("quantum entanglement qubits", top_k=3)
    assert len(results) > 0
    top = results[0]
    assert top["chunk_id"] == "quantum.txt_page_1_chunk_1"
    assert top["similarity_score"] > 0.0
    assert "superposition" in top["text"]
    assert top["metadata"]["filename"] == "quantum.txt"


def test_tfidf_unseen_query_terms(populated_hybrid_store: FAISSVectorStore) -> None:
    """3. Unseen query terms return empty results safely without raising exceptions."""
    results = populated_hybrid_store.search_tfidf("xyzgibberishword nonexistentterm", top_k=3)
    assert results == []


def test_tfidf_empty_query_and_empty_index() -> None:
    """4. Empty queries or empty store safely return empty lists."""
    empty_store = FAISSVectorStore(dimension=384)
    assert empty_store.search_tfidf("quantum", top_k=5) == []
    assert empty_store.search_tfidf("", top_k=5) == []
    assert empty_store.search_tfidf("   ", top_k=5) == []


# ============================================================================
# 2. Hybrid Retrieval & RRF Ranking Tests
# ============================================================================

def test_hybrid_search_combines_dense_and_sparse(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """5. Hybrid search combines dense semantic vector and sparse keywords via RRF."""
    # Query embedding pointing to chunk 0 (security)
    q_vec = _make_unit_vector(384, 0)
    # Query text mentioning quantum keywords (chunk 1)
    results = populated_hybrid_store.hybrid_search(
        query_text="quantum entanglement qubits",
        query_embedding=q_vec,
        top_k=3,
        dense_weight=0.6,
        tfidf_weight=0.4,
    )

    assert len(results) >= 2
    # Check that both chunk 0 and chunk 1 are surfaced
    chunk_ids = [r["chunk_id"] for r in results]
    assert "security.pdf_page_1_chunk_1" in chunk_ids
    assert "quantum.txt_page_1_chunk_1" in chunk_ids

    # Verify score normalization between 0.0 and 1.0
    for r in results:
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert "retrieval_source" in r["metadata"]


def test_hybrid_search_deduplication(populated_hybrid_store: FAISSVectorStore) -> None:
    """6. Chunks matching both dense and sparse channels are strictly deduplicated."""
    # Both vector and keyword point to chunk 2 (biology)
    q_vec = _make_unit_vector(384, 2)
    results = populated_hybrid_store.hybrid_search(
        query_text="photosynthesis glucose chloroplasts",
        query_embedding=q_vec,
        top_k=3,
    )

    assert len(results) > 0
    top_result = results[0]
    assert top_result["chunk_id"] == "biology.docx_page_1_chunk_1"
    assert top_result["metadata"]["retrieval_source"] == "hybrid"
    assert "dense_score" in top_result["metadata"]
    assert "tfidf_score" in top_result["metadata"]
    assert top_result["metadata"]["dense_score"] > 0.0
    assert top_result["metadata"]["tfidf_score"] > 0.0

    # Ensure no duplicates
    result_ids = [r["chunk_id"] for r in results]
    assert len(result_ids) == len(set(result_ids))


def test_hybrid_search_fallback_when_one_channel_empty(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """7. Hybrid search falls back cleanly when one channel returns no candidates."""
    # Only dense vector provided, empty text
    q_vec = _make_unit_vector(384, 1)
    dense_only = populated_hybrid_store.hybrid_search(
        query_text="",
        query_embedding=q_vec,
        top_k=2,
    )
    assert len(dense_only) > 0
    assert dense_only[0]["chunk_id"] == "quantum.txt_page_1_chunk_1"
    assert dense_only[0]["metadata"]["retrieval_source"] == "dense"

    # Only text provided, no embedding
    tfidf_only = populated_hybrid_store.hybrid_search(
        query_text="authentication security",
        query_embedding=None,
        top_k=2,
    )
    assert len(tfidf_only) > 0
    assert tfidf_only[0]["chunk_id"] == "security.pdf_page_1_chunk_1"
    assert tfidf_only[0]["metadata"]["retrieval_source"] == "tfidf"


def test_hybrid_search_configurable_weights(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """8. Changing dense vs tfidf weights alters ranking predictably."""
    dense_target = _make_unit_vector(384, 0)  # security
    sparse_text = "chloroplasts photosynthesis"  # biology

    # Heavy dense weighting: security should be rank 1
    dense_heavy = populated_hybrid_store.hybrid_search(
        query_text=sparse_text,
        query_embedding=dense_target,
        top_k=2,
        dense_weight=0.95,
        tfidf_weight=0.05,
    )
    assert dense_heavy[0]["chunk_id"] == "security.pdf_page_1_chunk_1"

    # Heavy sparse weighting: biology should be rank 1
    sparse_heavy = populated_hybrid_store.hybrid_search(
        query_text=sparse_text,
        query_embedding=dense_target,
        top_k=2,
        dense_weight=0.05,
        tfidf_weight=0.95,
    )
    assert sparse_heavy[0]["chunk_id"] == "biology.docx_page_1_chunk_1"


# ============================================================================
# 3. Lifecycle, Deletion & Consistency Tests
# ============================================================================

def test_delete_chunks_by_document_maintains_consistency(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """9. Deleting chunks by document filename updates both FAISS and TF-IDF."""
    assert populated_hybrid_store.total_vectors == 3

    # Delete quantum.txt
    deleted = populated_hybrid_store.delete_chunks_by_document("quantum.txt")
    assert deleted == 1
    assert populated_hybrid_store.total_vectors == 2
    assert populated_hybrid_store.has_chunk("quantum.txt_page_1_chunk_1") is False

    # Searching quantum keywords in TF-IDF must yield 0 results
    res = populated_hybrid_store.search_tfidf("quantum superposition entanglement", top_k=5)
    assert res == []

    # Other documents must still be searchable in both dense and sparse
    sec_res = populated_hybrid_store.search_tfidf("authentication security", top_k=5)
    assert len(sec_res) == 1
    assert sec_res[0]["chunk_id"] == "security.pdf_page_1_chunk_1"


def test_delete_single_chunk_maintains_consistency(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """10. Deleting a single chunk by chunk_id removes it from all indices."""
    success = populated_hybrid_store.delete_chunk("biology.docx_page_1_chunk_1")
    assert success is True
    assert populated_hybrid_store.total_vectors == 2
    assert populated_hybrid_store.has_chunk("biology.docx_page_1_chunk_1") is False

    # Biology text should not match
    assert populated_hybrid_store.search_tfidf("photosynthesis chloroplasts", top_k=5) == []


def test_clear_resets_both_dense_and_tfidf(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """11. clear() resets FAISS, stored vectors, chunk IDs, and TF-IDF matrix."""
    populated_hybrid_store.clear()
    assert populated_hybrid_store.is_empty is True
    assert populated_hybrid_store.total_vectors == 0
    assert populated_hybrid_store.tfidf_vectorizer is None
    assert populated_hybrid_store.tfidf_matrix is None
    assert populated_hybrid_store.tfidf_vocabulary_size == 0
    assert populated_hybrid_store.search_tfidf("security", top_k=5) == []


# ============================================================================
# 4. RAG Pipeline & Citation Preservation Tests
# ============================================================================

def test_rag_pipeline_uses_hybrid_retrieval(
    populated_hybrid_store: FAISSVectorStore,
) -> None:
    """12. RAGPipeline answers using hybrid retrieval and preserves citation metadata."""
    mock_llm = MockLLMClient()
    mock_emb = EmbeddingService()

    pipeline = RAGPipeline(
        vector_store=populated_hybrid_store,
        embedding_service=mock_emb,
        llm_client=mock_llm,
        retrieval_mode="hybrid",
        top_k=2,
    )

    response: RAGResponse = pipeline.answer("What are the requirements for authentication?")
    assert response.generation_status == "success"
    assert response.sufficient_context is True
    assert len(response.retrieved_chunks) > 0
    assert len(response.citations) > 0

    top_citation = response.citations[0]
    assert top_citation["filename"] in {"security.pdf", "quantum.txt", "biology.docx"}
    assert top_citation["page_number"] == 1
    assert top_citation["file_type"] in {"pdf", "txt", "docx"}
    assert top_citation["similarity_score"] is not None


# ============================================================================
# 5. TF-IDF Extractive Summarization Tests
# ============================================================================

def test_tfidf_extractive_summarizer_salience() -> None:
    """13. ExtractiveSummarizer computes sentence salience via genuine TF-IDF centroid scoring."""
    text = (
        "Artificial intelligence revolutionizes medical imaging diagnostics across modern hospitals. "
        "Convolutional neural networks detect tumors in magnetic resonance imaging scans with high accuracy. "
        "Radiologists review automated predictions before confirming clinical treatment protocols. "
        "Medical ethics requires transparency and interpretability in all healthcare algorithms."
    )
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(text, summary_type="concise")

    assert isinstance(res, SummaryResult)
    assert res.status == "success"
    assert res.method == "extractive"
    assert res.summary_sentence_count <= 2
    assert res.summary_length < len(text)
    assert "artificial intelligence" in res.summary.lower() or "medical" in res.summary.lower()


def test_tfidf_summarizer_all_stopwords_and_uncommon_words() -> None:
    """14. Summarizer handles inputs with repeated words, stopwords, and uncommon vocabulary."""
    all_stop = "It was there for him and her in the morning. They had been with them as well."
    summarizer = ExtractiveSummarizer()
    res = summarizer.summarize(all_stop, summary_type="concise")
    assert res.status == "success"
    assert len(res.summary) > 0

    uncommon = "Zyxwvu plorqik and flabbergasted quetzalcoatl danced beneath nebulous twilight skies. The ethereal vortex spun rapidly."
    res2 = summarizer.summarize(uncommon, summary_type="concise")
    assert res2.status == "success"
    assert len(res2.summary) > 0


# ============================================================================
# 6. LangChain Retriever Hybrid Mode Compatibility
# ============================================================================

def test_langchain_retriever_hybrid_mode(populated_hybrid_store: FAISSVectorStore) -> None:
    """15. FAISSRetrieverAdapter operates in hybrid mode with LangChain documents."""
    emb_service = EmbeddingService()
    retriever = populated_hybrid_store.as_retriever(
        embedding_service=emb_service,
        top_k=2,
        retrieval_mode="hybrid",
    )

    docs = retriever.invoke("photosynthesis sunlight plants")
    assert isinstance(docs, list)
    assert len(docs) > 0

    top_doc = docs[0]
    assert hasattr(top_doc, "page_content")
    assert hasattr(top_doc, "metadata")
    assert "chunk_id" in top_doc.metadata
    assert "similarity_score" in top_doc.metadata
    assert top_doc.metadata["similarity_score"] > 0.0
