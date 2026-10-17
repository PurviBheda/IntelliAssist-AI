"""Automated deterministic unit tests for IntelliAssist AI RAG Pipeline.

Verifies:
  1. A valid question retrieves context and generates an answer.
  2. Empty questions are rejected.
  3. Whitespace-only questions are rejected.
  4. Retrieved context is passed to the LLM.
  5. The LLM cannot receive unrelated, unretrieved documents.
  6. Citations preserve real filenames and page numbers.
  7. Chunk IDs are preserved.
  8. Empty retrieval results do not generate an unsupported answer.
  9. Insufficient context produces an appropriate response.
  10. Missing API configuration produces a clear error.
  11. LLM failures are handled safely.
  12. Embedding failures are handled safely.
  13. Retrieval failures are handled safely.
  14. Top-k configuration is respected.
  15. Tests do not make real network requests or paid API calls.
  16. Subscript and dictionary compatibility of RAGResponse.
  17. Logical page preservation for TXT and DOCX files.
  18. Context length truncation defense.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.embedding_service import EmbeddingService
from app.rag_pipeline import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_SYSTEM_INSTRUCTION,
    DEFAULT_TOP_K,
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    Citation,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    answer_question,
    build_citations,
    build_rag_prompt,
    is_context_insufficient_response,
    validate_question,
)
from app.vector_store import FAISSVectorStore


# ============================================================================
# Test Fixtures & Factories
# ============================================================================

def make_deterministic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate a deterministic unit-length vector seeded by text hash."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def create_mock_embedding_service(dim: int = 384) -> EmbeddingService:
    """Create a mock EmbeddingService producing deterministic vectors without model loading."""
    service = EmbeddingService(embedding_dimension=dim)
    mock_model = MagicMock()

    def mock_encode(sentences, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        if isinstance(sentences, str):
            return make_deterministic_embedding(sentences, dim)
        elif isinstance(sentences, list):
            return np.vstack([make_deterministic_embedding(s, dim) for s in sentences])
        return make_deterministic_embedding(str(sentences), dim)

    mock_model.encode.side_effect = mock_encode
    service._model = mock_model
    return service


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    return create_mock_embedding_service()


@pytest.fixture
def sample_chunks() -> List[Dict[str, Any]]:
    """Diverse multi-document chunks across PDF, DOCX, and TXT formats."""
    return [
        {
            "chunk_id": "employee_handbook.pdf_page_1_chunk_1",
            "text": "Standard working hours are 9:00 AM to 5:00 PM with core collaboration hours from 10:00 AM to 3:00 PM.",
            "metadata": {
                "filename": "employee_handbook.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "employee_handbook.pdf_page_2_chunk_1",
            "text": "Employees receive 20 days of paid time off (PTO) annually, accrued monthly.",
            "metadata": {
                "filename": "employee_handbook.pdf",
                "file_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "security_policy.docx_page_1_chunk_1",
            "text": "All staff must enable multi-factor authentication (MFA) and choose passwords of at least 14 characters.",
            "metadata": {
                "filename": "security_policy.docx",
                "file_type": "docx",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "security_policy.docx_page_1_chunk_2",
            "text": "Data is categorized into four classification tiers: Public, Internal, Confidential, and Restricted.",
            "metadata": {
                "filename": "security_policy.docx",
                "file_type": "docx",
                "page_number": 1,
                "chunk_index": 2,
            },
        },
        {
            "chunk_id": "notes.txt_page_1_chunk_1",
            "text": "Expense reports must be submitted within 30 days of incurring the expense with itemized receipts attached.",
            "metadata": {
                "filename": "notes.txt",
                "file_type": "txt",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
    ]


@pytest.fixture
def populated_vector_store(
    sample_chunks: List[Dict[str, Any]],
    mock_embedding_service: EmbeddingService,
) -> FAISSVectorStore:
    """FAISS vector store populated with sample chunks and deterministic embeddings."""
    store = FAISSVectorStore(dimension=384, normalize_vectors=True)
    embeddings = [
        mock_embedding_service.embed_text(c["text"])
        for c in sample_chunks
    ]
    store.add_chunks(sample_chunks, embeddings)
    return store


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    return MockLLMClient()


# ============================================================================
# Unit Tests
# ============================================================================

class TestRAGPipeline:
    """Test suite covering the 15 required verification targets for the RAG pipeline."""

    def test_01_valid_question_retrieves_context_and_generates_answer(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """1. A valid question retrieves context and generates an answer."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=3,
        )

        response = pipeline.answer("What is the password requirement?")

        assert isinstance(response, RAGResponse)
        assert response.generation_status == "success"
        assert response.sufficient_context is True
        assert response.retrieval_count == 3
        assert len(response.retrieved_chunks) == 3
        assert len(response.citations) == 3
        # Ensure answer is grounded in one of the retrieved sources
        assert any(c["filename"] in response.answer for c in response.citations)
        assert mock_llm_client.call_count == 1

    def test_02_empty_questions_are_rejected(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """2. Empty questions are rejected."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
        )

        with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
            pipeline.answer("")

        # Also verify validate_question directly
        with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
            validate_question("")

        # Verify None is rejected
        with pytest.raises(ValueError, match="Question cannot be None"):
            validate_question(None)

    def test_03_whitespace_only_questions_are_rejected(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """3. Whitespace-only questions are rejected."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
        )

        with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
            pipeline.answer("   \t  \n  ")

        with pytest.raises(ValueError, match="Question cannot be empty or whitespace-only"):
            validate_question("     ")

    def test_04_retrieved_context_is_passed_to_the_llm(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """4. Retrieved context is passed to the LLM."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=2,
        )

        response = pipeline.answer("What are standard working hours?")

        assert mock_llm_client.call_count == 1
        last_prompt = mock_llm_client.last_call["prompt"]

        # Ensure retrieved chunk content is embedded in the prompt sent to LLM
        for chunk in response.retrieved_chunks:
            assert chunk["text"] in last_prompt
            assert chunk["metadata"]["filename"] in last_prompt

    def test_05_unrelated_unretrieved_documents_not_passed_to_llm(
        self,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """5. The LLM cannot receive unrelated, unretrieved documents."""
        store = FAISSVectorStore(dimension=384, normalize_vectors=True)
        chunk_a = {
            "chunk_id": "allowed.pdf_page_1_chunk_1",
            "text": "CONFIDENTIAL PROJECT ALPHA: Authorized eyes only.",
            "metadata": {"filename": "allowed.pdf", "page_number": 1},
        }
        chunk_b = {
            "chunk_id": "unrelated.pdf_page_1_chunk_1",
            "text": "TOP SECRET PROJECT OMEGA: Nuclear submarine blueprint specs.",
            "metadata": {"filename": "unrelated.pdf", "page_number": 1},
        }

        # Seed embeddings such that chunk_a matches query perfectly
        v_a = make_deterministic_embedding("target_query", 384)
        v_b = make_deterministic_embedding("completely_different_seed", 384)
        store.add_chunks([chunk_a, chunk_b], [v_a, v_b])

        # Mock query embedding to match v_a
        mock_service = MagicMock(spec=EmbeddingService)
        mock_service.embed_text.return_value = v_a

        pipeline = RAGPipeline(
            vector_store=store,
            embedding_service=mock_service,
            llm_client=mock_llm_client,
            top_k=1,  # Only retrieve 1 chunk
        )

        response = pipeline.answer("target_query")

        assert response.retrieval_count == 1
        prompt = mock_llm_client.last_call["prompt"]
        assert "PROJECT ALPHA" in prompt
        assert "PROJECT OMEGA" not in prompt
        assert "unrelated.pdf" not in prompt

    def test_06_citations_preserve_real_filenames_and_page_numbers(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """6. Citations preserve real filenames and page numbers."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=3,
        )

        response = pipeline.answer("How much paid time off is provided?")

        assert len(response.citations) > 0
        known_filenames = {"employee_handbook.pdf", "security_policy.docx", "notes.txt"}

        for citation in response.citations:
            assert citation["filename"] in known_filenames
            assert isinstance(citation["page_number"], int)
            assert citation["page_number"] >= 1
            # Real page numbers are preserved, not invented strings
            assert citation["page_number"] in (1, 2)

    def test_07_chunk_ids_are_preserved(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """7. Chunk IDs are preserved."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=2,
        )

        response = pipeline.answer("What is the expense submission window?")

        for citation, chunk in zip(response.citations, response.retrieved_chunks):
            assert citation["chunk_id"] == chunk["chunk_id"]
            assert citation["chunk_id"].endswith("_chunk_1") or citation["chunk_id"].endswith("_chunk_2")

    def test_08_empty_retrieval_results_do_not_generate_unsupported_answer(
        self,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """8. Empty retrieval results do not generate an unsupported answer."""
        empty_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

        pipeline = RAGPipeline(
            vector_store=empty_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
        )

        response = pipeline.answer("What is the refund policy?")

        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert response.retrieval_count == 0
        assert response.citations == []
        assert response.retrieved_chunks == []
        assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
        # LLM must NOT be called when retrieval is empty
        assert mock_llm_client.call_count == 0

    def test_09_insufficient_context_produces_appropriate_response(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """9. Insufficient context produces an appropriate response."""
        # LLM states it cannot answer based on context
        refusal_llm = MockLLMClient(
            default_response="The available documents do not contain enough information to answer this question."
        )

        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=refusal_llm,
        )

        response = pipeline.answer("What is the CEO's personal home address?")

        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert "do not contain enough information" in response.answer

        # Also test similarity threshold filtering causing insufficient context
        high_threshold_pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=refusal_llm,
            similarity_threshold=0.999,  # Unrealistic threshold filters everything
        )
        threshold_response = high_threshold_pipeline.answer("Any question")
        assert threshold_response.sufficient_context is False
        assert threshold_response.retrieval_count == 0
        assert threshold_response.answer == INSUFFICIENT_CONTEXT_MESSAGE

    def test_10_missing_api_configuration_produces_clear_error(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """10. Missing API configuration produces a clear error."""
        # Ensure API key variables are unset
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        # Direct client instantiation fails with clear error
        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiLLMClient(api_key=None)

        # RAG pipeline with lazy client initialization catches missing key safely
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=None,  # Triggers lazy Gemini client init
        )

        response = pipeline.answer("What are standard hours?")

        assert response.generation_status == "error"
        assert response.sufficient_context is False
        assert "GEMINI_API_KEY" in response.answer
        assert response.error is not None
        assert "Gemini API key is missing" in response.error

    def test_11_llm_failures_are_handled_safely(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """11. LLM failures are handled safely."""
        failing_llm = MagicMock(spec=BaseLLMClient)
        failing_llm.model_name = "failing-gemini"
        failing_llm.generate_response.side_effect = RuntimeError("Quota exceeded or 503 Service Unavailable")

        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=failing_llm,
        )

        response = pipeline.answer("What is the PTO accrual rate?")

        assert response.generation_status == "error"
        assert response.sufficient_context is False
        assert "Unable to generate answer due to language model failure" in response.answer
        assert "Quota exceeded" in response.error
        # Preserves retrieved chunks and citations for debugging
        assert response.retrieval_count > 0
        assert len(response.citations) > 0

    def test_12_embedding_failures_are_handled_safely(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """12. Embedding failures are handled safely."""
        failing_service = MagicMock(spec=EmbeddingService)
        failing_service.embed_text.side_effect = RuntimeError("CUDA out of memory or model tensor error")

        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=failing_service,
            llm_client=mock_llm_client,
        )

        response = pipeline.answer("What is the policy?")

        assert response.generation_status == "error"
        assert response.sufficient_context is False
        assert "Unable to process question due to embedding failure" in response.answer
        assert "CUDA out of memory" in response.error
        assert mock_llm_client.call_count == 0

    def test_13_retrieval_failures_are_handled_safely(
        self,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """13. Retrieval failures are handled safely."""
        failing_store = MagicMock(spec=FAISSVectorStore)
        failing_store.is_empty = False
        failing_store.search.side_effect = RuntimeError("FAISS search internal index segmentation fault")

        pipeline = RAGPipeline(
            vector_store=failing_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
        )

        response = pipeline.answer("What is the policy?")

        assert response.generation_status == "error"
        assert response.sufficient_context is False
        assert "Unable to retrieve relevant documents due to search failure" in response.answer
        assert "FAISS search internal index" in response.error
        assert mock_llm_client.call_count == 0

    def test_14_top_k_configuration_is_respected(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """14. Top-k configuration is respected."""
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=2,
        )

        # Baseline top_k=2 from pipeline init
        res_k2 = pipeline.answer("working hours")
        assert res_k2.retrieval_count == 2
        assert len(res_k2.retrieved_chunks) == 2
        assert len(res_k2.citations) == 2

        # Override top_k=4 in call
        res_k4 = pipeline.answer("working hours", top_k=4)
        assert res_k4.retrieval_count == 4
        assert len(res_k4.retrieved_chunks) == 4
        assert len(res_k4.citations) == 4

        # Invalid top_k rejected
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.answer("test", top_k=0)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            pipeline.answer("test", top_k=-1)

    def test_15_tests_do_not_make_real_network_requests(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """15. Tests do not make real network requests or paid API calls."""
        with patch("socket.socket.connect") as mock_connect:
            mock_connect.side_effect = AssertionError("Network call attempted during test!")

            pipeline = RAGPipeline(
                vector_store=populated_vector_store,
                embedding_service=mock_embedding_service,
                llm_client=mock_llm_client,
                top_k=3,
            )

            response = pipeline.answer("What is the security classification?")
            assert response.generation_status == "success"
            assert mock_connect.call_count == 0


class TestAdditionalRAGCapabilities:
    """Additional unit tests verifying schemas, citations, and functional helpers."""

    def test_rag_response_dictionary_access(self) -> None:
        """Verify RAGResponse supports both attribute and subscript dict access."""
        resp = RAGResponse(
            question="What is X?",
            answer="X is 42.",
            citations=[{"chunk_id": "c1", "filename": "doc.pdf", "page_number": 1}],
            retrieved_chunks=[{"chunk_id": "c1"}],
            retrieval_count=1,
            sufficient_context=True,
        )

        # Subscript access
        assert resp["question"] == "What is X?"
        assert resp["answer"] == "X is 42."
        assert resp["retrieval_count"] == 1
        assert resp["sufficient_context"] is True

        # Dict get method
        assert resp.get("answer") == "X is 42."
        assert resp.get("nonexistent", "fallback") == "fallback"

        # to_dict
        d = resp.to_dict()
        assert isinstance(d, dict)
        assert d["answer"] == "X is 42."

    def test_txt_and_docx_logical_page_preservation(self) -> None:
        """Verify that TXT and DOCX logical page 1 is preserved in citations."""
        chunks = [
            {
                "chunk_id": "policy.docx_page_1_chunk_1",
                "metadata": {"filename": "policy.docx", "file_type": "docx", "page_number": 1},
                "text": "DOCX content",
            },
            {
                "chunk_id": "readme.txt_page_1_chunk_1",
                "metadata": {"filename": "readme.txt", "file_type": "txt", "page_number": 1},
                "text": "TXT content",
            },
        ]
        citations = build_citations(chunks, answer_text="Refer to policy.docx")

        assert len(citations) == 2
        assert citations[0]["filename"] == "policy.docx"
        assert citations[0]["page_number"] == 1
        assert citations[0]["is_cited"] is True

        assert citations[1]["filename"] == "readme.txt"
        assert citations[1]["page_number"] == 1
        assert citations[1]["is_cited"] is False

    def test_context_character_limit_truncation(self) -> None:
        """Verify build_rag_prompt truncates context when exceeding max_context_chars."""
        chunks = [
            {
                "chunk_id": f"doc.pdf_page_1_chunk_{i}",
                "text": "A" * 500,
                "metadata": {"filename": "doc.pdf", "page_number": 1},
            }
            for i in range(10)
        ]

        # Restrict context budget to 1200 characters (~2 chunks)
        prompt, included = build_rag_prompt("Question?", chunks, max_context_chars=1200)

        assert len(included) < 10
        assert len(included) >= 1
        assert "[Source 1]" in prompt

    def test_functional_answer_question_interface(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Verify the functional convenience wrapper answer_question()."""
        response = answer_question(
            question="What is the data tier?",
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            top_k=2,
        )

        assert isinstance(response, RAGResponse)
        assert response.generation_status == "success"
        assert response.retrieval_count == 2
        assert len(response.citations) == 2

    def test_is_context_insufficient_detection(self) -> None:
        """Verify detection of refusal phrases in generated answers."""
        assert is_context_insufficient_response(
            "The available documents do not contain enough information to answer this question."
        ) is True

        assert is_context_insufficient_response(
            "I cannot answer based on the provided documents."
        ) is True

        assert is_context_insufficient_response("") is True

        assert is_context_insufficient_response(
            "According to employee_handbook.pdf (Page 1), standard hours are 9 to 5."
        ) is False
