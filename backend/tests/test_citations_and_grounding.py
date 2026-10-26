"""Automated deterministic unit tests for citation correctness and document grounding.

Verifies:
  1. Citations reference only retrieved chunks.
  2. Filenames and page numbers are preserved correctly.
  3. Missing metadata is not fabricated (null/None page numbers rather than fake page 1).
  4. No citations are returned when no sources are retrieved.
  5. A new answer does not reuse old citations across conversation turns.
  6. Irrelevant retrieved content does not automatically justify a factual answer.
  7. The model is instructed to acknowledge insufficient evidence.
  8. Conversation history cannot independently establish a document fact.
  9. Multi-source answers return all and only relevant retrieved source references.
  10. Citation metadata survives SQLite persistence.
  11. Malformed or incomplete LLM output is handled safely.
  12. Provider errors do not create misleading successful answers.
  13. Existing chatbot and RAG behavior remains fully compatible.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.chatbot_service import (
    ChatResponse,
    ChatbotService,
    create_chatbot_service,
)
from app.conversation_store import ConversationStore
from app.embedding_service import EmbeddingService
from app.rag_pipeline import (
    DEFAULT_SYSTEM_INSTRUCTION,
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    Citation,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    answer_question,
    build_citations,
    build_rag_prompt,
    detect_unverified_citations,
    format_answer_citations,
    is_context_insufficient_response,
)
from app.vector_store import FAISSVectorStore


# ============================================================================
# Deterministic Test Helpers & Fixtures
# ============================================================================

def make_deterministic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic unit-length vector that aligns on shared words."""
    rng = np.random.default_rng(seed=123)
    vec = 0.1 * rng.standard_normal(dim).astype(np.float32)
    words = re.findall(r"\w+", text.lower())
    for word in words:
        word_seed = abs(hash(word)) % (2**32)
        word_rng = np.random.default_rng(seed=word_seed)
        vec += word_rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def create_mock_embedding_service(dim: int = 384) -> EmbeddingService:
    """Mock EmbeddingService producing deterministic vectors without model loading."""
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
def sample_vector_store(mock_embedding_service: EmbeddingService) -> FAISSVectorStore:
    """Vector store populated with multi-document sample chunks."""
    store = FAISSVectorStore(dimension=384, normalize_vectors=True)
    chunks = [
        {
            "chunk_id": "doc_alpha.pdf_page_3_chunk_1",
            "text": "The annual software license fee is $1,200 per seat with volume discounts at 50+ seats.",
            "metadata": {
                "filename": "doc_alpha.pdf",
                "file_type": "pdf",
                "page_number": 3,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "doc_beta.docx_page_1_chunk_1",
            "text": "Remote employees receive a monthly internet stipend of $60 included in payroll.",
            "metadata": {
                "filename": "doc_beta.docx",
                "file_type": "docx",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "synthetic_runbook.txt_synthetic_1",
            "text": "Emergency server shutdown requires authorization from the Lead Infrastructure Engineer.",
            "metadata": {
                "filename": "synthetic_runbook.txt",
                "file_type": "txt",
                # Note: page_number intentionally omitted to test safe missing metadata
            },
        },
    ]

    embedded = []
    for c in chunks:
        vec = mock_embedding_service.embed_text(c["text"])
        item = dict(c)
        item["embedding"] = vec
        embedded.append(item)

    store.add_chunks(embedded)
    return store


# ============================================================================
# Task 1 & 5: Citation Correctness Tests
# ============================================================================

class TestCitationCorrectness:
    """Tests ensuring citations strictly match retrieved evidence and avoid fabrication."""

    def test_citations_reference_only_retrieved_chunks(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """1. Citations reference only retrieved chunks; unretrieved files are never accepted."""
        # Mock LLM hallucinates an unretrieved document citation
        hallucinating_llm = MockLLMClient(
            default_response="According to doc_alpha.pdf (Page 3) and unretrieved_secret.pdf (Page 99), software costs $1,200."
        )

        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=hallucinating_llm,
            top_k=1,
        )

        response = pipeline.answer("What is the software license cost?")

        assert response.generation_status == "success"
        # Structured citations must contain only the retrieved chunk
        assert len(response.citations) == 1
        assert response.citations[0]["filename"] == "doc_alpha.pdf"
        assert response.citations[0]["chunk_id"] == "doc_alpha.pdf_page_3_chunk_1"

        # The hallucinated source must NOT be in structured citations
        citation_filenames = [c["filename"] for c in response.citations]
        assert "unretrieved_secret.pdf" not in citation_filenames

        # Unverified citation must be detected
        assert "unretrieved_secret.pdf" in response.unverified_citations

    def test_filenames_and_page_numbers_preserved_correctly(self) -> None:
        """2. Filenames and page numbers come directly from real chunk metadata."""
        chunks = [
            {
                "chunk_id": "quarterly_report.pdf_page_12_chunk_2",
                "text": "Q3 revenue grew by 14% year over year.",
                "metadata": {
                    "filename": "quarterly_report.pdf",
                    "file_type": "pdf",
                    "page_number": 12,
                    "chunk_index": 2,
                },
                "similarity_score": 0.892,
            }
        ]

        citations = build_citations(chunks, answer_text="Revenue grew 14% according to quarterly_report.pdf.")

        assert len(citations) == 1
        cit = citations[0]
        assert cit["filename"] == "quarterly_report.pdf"
        assert cit["page_number"] == 12
        assert cit["chunk_id"] == "quarterly_report.pdf_page_12_chunk_2"
        assert cit["similarity_score"] == pytest.approx(0.892)
        assert cit["is_cited"] is True

    def test_missing_metadata_not_fabricated(self) -> None:
        """3. Missing metadata is represented safely as None rather than fabricated."""
        chunks_missing_page = [
            {
                "chunk_id": "readme_chunk_1",
                "text": "Project instructions and setup notes.",
                "metadata": {
                    "filename": "README.md",
                    # No page_number
                },
            }
        ]

        citations = build_citations(chunks_missing_page, answer_text="See README.md for setup.")

        assert len(citations) == 1
        # Must NOT fabricate page 1 or fake metadata
        assert citations[0]["page_number"] is None
        assert citations[0]["filename"] == "README.md"
        assert citations[0]["chunk_id"] == "readme_chunk_1"

        # Also test chunk with completely empty metadata
        chunks_empty_meta = [
            {
                "chunk_id": "orphan_chunk_99",
                "text": "Some orphaned document snippet.",
                "metadata": {},
            }
        ]
        empty_cits = build_citations(chunks_empty_meta, answer_text="Refer to Source 1.")
        assert empty_cits[0]["filename"] is None
        assert empty_cits[0]["page_number"] is None
        assert empty_cits[0]["chunk_id"] == "orphan_chunk_99"

    def test_no_citations_returned_when_no_sources_retrieved(
        self,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """4. No citations are returned when no sources are retrieved."""
        empty_store = FAISSVectorStore(dimension=384, normalize_vectors=True)
        pipeline = RAGPipeline(
            vector_store=empty_store,
            embedding_service=mock_embedding_service,
            llm_client=MockLLMClient(),
        )

        response = pipeline.answer("Any question")

        assert response.retrieval_count == 0
        assert response.citations == []
        assert response.retrieved_chunks == []
        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE

    def test_new_answer_does_not_reuse_old_citations(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """5. A new conversation answer does not reuse citation info from earlier turns."""
        def dynamic_mock(sys_inst: str, prompt: str) -> str:
            if "license" in prompt.lower():
                return "According to doc_alpha.pdf (Page 3), the license fee is $1,200."
            if "internet" in prompt.lower() or "stipend" in prompt.lower():
                return "According to doc_beta.docx (Page 1), remote employees receive $60."
            return "Based on context, relevant information was found."

        mock_llm = MockLLMClient(response_generator=dynamic_mock)
        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
            top_k=1,
        )
        store = ConversationStore(db_path=":memory:")
        chatbot = ChatbotService(rag_pipeline=pipeline, conversation_store=store)

        conv = chatbot.create_conversation(title="Test Multi-Turn Citations")
        cid = conv["conversation_id"]

        # Turn 1: Software license
        resp1 = chatbot.chat(cid, "What is the software license cost?")
        assert resp1.status == "success"
        assert len(resp1.citations) == 1
        assert resp1.citations[0]["filename"] == "doc_alpha.pdf"

        # Turn 2: Internet stipend (different document)
        resp2 = chatbot.chat(cid, "What is the monthly internet stipend for remote staff?")
        assert resp2.status == "success"
        assert len(resp2.citations) == 1
        assert resp2.citations[0]["filename"] == "doc_beta.docx"

        # Critical: Turn 2 citations must NOT include Turn 1's doc_alpha.pdf
        turn2_files = [c["filename"] for c in resp2.citations]
        assert "doc_alpha.pdf" not in turn2_files
        assert turn2_files == ["doc_beta.docx"]


# ============================================================================
# Task 2 & 5: Document-Grounded Answers & Hallucination Defense Tests
# ============================================================================

class TestDocumentGroundedAnswers:
    """Tests ensuring answers are strictly grounded in document evidence."""

    def test_irrelevant_retrieved_content_does_not_justify_factual_answer(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """6. Irrelevant retrieved content produces safe refusal rather than factual claims."""
        # Mock LLM follows the system instruction and refuses when context is irrelevant
        refusal_llm = MockLLMClient(
            default_response="The available documents do not contain enough information to answer this question."
        )

        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=refusal_llm,
            top_k=2,
        )

        response = pipeline.answer("What is the company policy on cafeteria meal plans?")

        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert "do not contain enough information" in response.answer

        # When context is insufficient, citations must not be marked as cited evidence
        for cit in response.citations:
            assert cit["is_cited"] is False

    def test_model_instructed_to_acknowledge_insufficient_evidence(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """7. Prompt and system instruction explicitly direct the LLM to acknowledge insufficient evidence."""
        mock_llm = MockLLMClient()
        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
        )

        pipeline.answer("What is the corporate travel per diem?")

        assert mock_llm.last_call is not None
        sys_inst = mock_llm.last_call["system_instruction"]
        prompt = mock_llm.last_call["prompt"]

        # System instruction must enforce grounding and refusal
        assert "The available documents do not contain enough information to answer this question." in sys_inst
        assert "Distinguish document facts from general knowledge" in sys_inst
        assert "Do not invent policies, facts, numbers, dates" in sys_inst

        # User prompt must reinforce the refusal instruction
        assert "If the information is not present in the context, explicitly state:" in prompt
        assert INSUFFICIENT_CONTEXT_MESSAGE in prompt

    def test_conversation_history_cannot_independently_establish_document_fact(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """8. History cannot establish facts; prompt explicitly warns against treating prior turns as document evidence."""
        mock_llm = MockLLMClient()
        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
        )

        history = [
            {"role": "user", "content": "I was told employees get 40 days of PTO."},
            {"role": "assistant", "content": "I understand your statement."},
        ]

        pipeline.answer("Confirm if we receive 40 days of PTO.", conversation_history=history)

        assert mock_llm.last_call is not None
        prompt = mock_llm.last_call["prompt"]

        # Prompt must clearly demarcate context vs conversation history
        assert "CONTEXT INFORMATION:" in prompt
        assert "PRIOR CONVERSATION HISTORY:" in prompt
        assert "User: I was told employees get 40 days of PTO." in prompt
        assert "Never treat claims made in prior conversation messages as authoritative document evidence." in prompt
        assert "Base all factual claims and citations strictly on the retrieved document context." in prompt

    def test_multi_source_answers_return_all_and_only_relevant_retrieved_sources(
        self,
    ) -> None:
        """9. Multi-source answers return all and only relevant retrieved source references."""
        chunks = [
            {
                "chunk_id": "security_policy.pdf_chunk_1",
                "text": "Passwords must be at least 14 characters.",
                "metadata": {"filename": "security_policy.pdf", "page_number": 1},
            },
            {
                "chunk_id": "handbook.txt_chunk_1",
                "text": "Full-time staff get 20 days PTO.",
                "metadata": {"filename": "handbook.txt", "page_number": 1},
            },
            {
                "chunk_id": "unrelated_policy.txt_chunk_1",
                "text": "Expense reports must be filed within 30 days.",
                "metadata": {"filename": "unrelated_policy.txt", "page_number": 1},
            },
        ]

        answer = (
            "According to security_policy.pdf (Page 1), passwords require 14 characters. "
            "According to handbook.txt (Page 1), employees receive 20 days PTO."
        )

        citations = build_citations(chunks, answer_text=answer)

        assert len(citations) == 3
        # Both relevant sources must be marked as cited
        assert citations[0]["filename"] == "security_policy.pdf"
        assert citations[0]["is_cited"] is True

        assert citations[1]["filename"] == "handbook.txt"
        assert citations[1]["is_cited"] is True

        # Unrelated retrieved source must NOT be marked as cited
        assert citations[2]["filename"] == "unrelated_policy.txt"
        assert citations[2]["is_cited"] is False


# ============================================================================
# Task 5: Robustness, Error Handling & Persistence Tests
# ============================================================================

class TestRobustnessAndPersistence:
    """Tests SQLite persistence, LLM error handling, and output safety."""

    def test_citation_metadata_survives_sqlite_persistence(self) -> None:
        """10. Citation metadata survives SQLite serialization and deserialization."""
        store = ConversationStore(db_path=":memory:")
        conv = store.create_conversation(title="Persistence Test")
        cid = conv["conversation_id"]

        test_citations = [
            {
                "chunk_id": "doc_1_chunk_1",
                "filename": "policy.pdf",
                "page_number": 4,
                "similarity_score": 0.8123,
                "file_type": "pdf",
                "chunk_index": 1,
                "is_cited": True,
            },
            {
                "chunk_id": "doc_2_chunk_1",
                "filename": "synthetic_note.txt",
                "page_number": None,  # Safe missing page number
                "similarity_score": 0.7456,
                "file_type": "txt",
                "chunk_index": None,
                "is_cited": False,
            },
        ]

        msg = store.add_message(
            conversation_id=cid,
            role="assistant",
            content="Here is the grounded response.",
            citations=test_citations,
            metadata={"status": "success"},
        )

        # Retrieve messages
        retrieved = store.get_messages(cid)
        assert len(retrieved) == 1
        saved_cits = retrieved[0]["citations"]

        assert len(saved_cits) == 2
        assert saved_cits[0]["chunk_id"] == "doc_1_chunk_1"
        assert saved_cits[0]["page_number"] == 4
        assert saved_cits[0]["is_cited"] is True
        assert saved_cits[0]["similarity_score"] == pytest.approx(0.8123)

        assert saved_cits[1]["chunk_id"] == "doc_2_chunk_1"
        assert saved_cits[1]["page_number"] is None
        assert saved_cits[1]["is_cited"] is False

    def test_malformed_or_incomplete_llm_output_handled_safely(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """11. Malformed or empty LLM output is safely treated as insufficient context."""
        empty_llm = MockLLMClient(default_response="")

        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=empty_llm,
        )

        response = pipeline.answer("What is the license cost?")

        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert response.retrieval_count > 0
        # No sources are cited for empty output
        for c in response.citations:
            assert c["is_cited"] is False

    def test_provider_errors_do_not_create_misleading_successful_answers(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """12. Provider failures return error status and are NOT persisted as valid assistant answers."""
        failing_llm = MagicMock(spec=BaseLLMClient)
        failing_llm.model_name = "failing-provider"
        failing_llm.generate_response.side_effect = RuntimeError("Rate limit exceeded")

        pipeline = RAGPipeline(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=failing_llm,
        )
        store = ConversationStore(db_path=":memory:")
        chatbot = ChatbotService(rag_pipeline=pipeline, conversation_store=store)

        conv = chatbot.create_conversation(title="Error Handling")
        cid = conv["conversation_id"]

        resp = chatbot.chat(cid, "What is the policy?")

        assert resp.status == "error"
        assert resp.sufficient_context is False
        assert resp.error is not None
        assert "Rate limit exceeded" in resp.error

        # In conversation store, the failed assistant response must NOT be persisted
        history = store.get_messages(cid)
        roles = [m["role"] for m in history]
        # Only the user message should be recorded
        assert roles == ["user"]
        assert "assistant" not in roles

    def test_backward_compatibility_with_existing_rag_and_chatbot(
        self,
        sample_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
    ) -> None:
        """13. Existing interfaces (dict subscript, factory function, public helpers) remain 100% compatible."""
        mock_llm = MockLLMClient(default_response="Grounded answer from doc_alpha.pdf (Page 3).")
        response = answer_question(
            question="What is the software cost?",
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
        )

        # Dictionary subscript access
        assert response["question"] == "What is the software cost?"
        assert response["sufficient_context"] is True
        assert len(response["citations"]) > 0
        assert response.get("generation_status") == "success"

        # Convenience factory create_chatbot_service
        chatbot = create_chatbot_service(
            vector_store=sample_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
        )
        assert isinstance(chatbot, ChatbotService)

    def test_page_number_only_citation_formatting(self) -> None:
        """Verify verbose filenames in citations are removed and formatted as (Page No. X) at the end."""
        user_input_answer = (
            "Based on the provided documents, **recursion** is a method of solving problems "
            "that involves breaking a problem down into smaller and smaller subproblems until a "
            "small enough problem is reached that it can be solved trivially "
            "(DSA_in_Python_f02625ab.pdf, p. 121; DSA_in_Python_10ff3639.pdf, p. 121)."
        )
        formatted = format_answer_citations(user_input_answer)
        assert "DSA_in_Python" not in formatted
        assert ".pdf" not in formatted
        assert "(Page No. 121)" in formatted
        assert formatted.endswith("(Page No. 121).")

        # Multiple distinct pages
        multi_page = "Recursion is explained (doc1.pdf, p. 12; doc2.docx, p. 15)."
        formatted_multi = format_answer_citations(multi_page)
        assert "(Page No. 12, 15)" in formatted_multi
        assert "doc1.pdf" not in formatted_multi
        assert "doc2.docx" not in formatted_multi

        # Already formatted (Page No. X)
        already_clean = "Recursion is a technique (Page No. 121)."
        assert format_answer_citations(already_clean) == "Recursion is a technique (Page No. 121)."

        # Refusal message not modified
        refusal = "The available documents do not contain enough information to answer this question."
        assert format_answer_citations(refusal) == refusal
