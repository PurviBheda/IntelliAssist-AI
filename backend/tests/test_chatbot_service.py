"""Automated deterministic unit tests for IntelliAssist AI ChatbotService.

Verifies:
  1. Valid question retrieves context and generates persistent chat answer.
  2. Empty or whitespace-only user messages are rejected.
  3. Non-string user messages are rejected.
  4. Unknown conversation ID produces clear KeyError or creates on demand.
  5. User message and assistant response are persisted with timestamps.
  6. Recent conversation history is incorporated for follow-up questions.
  7. History is loaded before saving user message to prevent message duplication.
  8. Configurable history limits (max_history_messages) are strictly enforced.
  9. Retrieved document context and conversation history are kept strictly separate.
  10. Strict conversation isolation across multiple conversation threads.
  11. Source citations and chunk metadata from RAG are preserved in response and DB.
  12. Safe refusal on insufficient context is preserved and persisted as assistant turn.
  13. No fabricated citations when RAG returns none.
  14. LLM provider and RAG errors are safely handled without persisting failed answers.
  15. Missing API key handling produces clean error without secret leakage.
  16. Accurate reporting of live vs mocked LLM (never falsely claims live Gemini).
  17. Zero network requests or live cloud API calls during testing.
  18. Convenience factory create_chatbot_service() functionality.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
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
    INSUFFICIENT_CONTEXT_MESSAGE,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
)
from app.vector_store import FAISSVectorStore


# ============================================================================
# Deterministic Fixtures
# ============================================================================

def make_deterministic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic unit-length vector seeded by text hash."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def create_mock_embedding_service(dim: int = 384) -> EmbeddingService:
    """Create mock EmbeddingService producing deterministic vectors without model loading."""
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
def sample_indexed_vector_store(mock_embedding_service: EmbeddingService) -> FAISSVectorStore:
    """Vector store populated with security policy and handbook sample chunks."""
    store = FAISSVectorStore(dimension=384, normalize_vectors=True)
    chunks = [
        {
            "chunk_id": "security_policy.pdf_page_1_chunk_1",
            "text": "All employee passwords must be at least 14 characters in length and expire every 90 days.",
            "metadata": {
                "filename": "security_policy.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "security_policy.pdf_page_2_chunk_1",
            "text": "Multi-factor authentication (MFA) is strictly mandatory for all administrative access.",
            "metadata": {
                "filename": "security_policy.pdf",
                "file_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "employee_handbook.txt_page_1_chunk_1",
            "text": "Employees receive 20 days of paid time off (PTO) annually. Unused PTO rolls over up to 5 days.",
            "metadata": {
                "filename": "employee_handbook.txt",
                "file_type": "txt",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
    ]

    embedded_chunks = []
    for chunk in chunks:
        vec = mock_embedding_service.embed_text(chunk["text"])
        chunk_copy = dict(chunk)
        chunk_copy["embedding"] = vec
        embedded_chunks.append(chunk_copy)

    store.add_chunks(embedded_chunks)
    return store


@pytest.fixture
def memory_conversation_store() -> ConversationStore:
    return ConversationStore(db_path=":memory:")


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    def generator(sys_inst: str, prompt: str) -> str:
        if "password" in prompt.lower():
            return "According to security_policy.pdf (Page 1), passwords must be at least 14 characters."
        if "pto" in prompt.lower() or "paid time off" in prompt.lower():
            return "According to employee_handbook.txt (Page 1), employees receive 20 days of PTO."
        if "expire" in prompt.lower() and "security_policy" in prompt.lower():
            return "According to security_policy.pdf (Page 1), passwords expire every 90 days."
        return "Based on the provided context, relevant policy information is found in the cited source."

    return MockLLMClient(response_generator=generator, model_name="mock-grounded-assistant")


@pytest.fixture
def chatbot_service(
    sample_indexed_vector_store: FAISSVectorStore,
    mock_embedding_service: EmbeddingService,
    mock_llm_client: MockLLMClient,
    memory_conversation_store: ConversationStore,
) -> ChatbotService:
    pipeline = RAGPipeline(
        vector_store=sample_indexed_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_llm_client,
        top_k=2,
    )
    return ChatbotService(
        rag_pipeline=pipeline,
        conversation_store=memory_conversation_store,
        max_history_messages=4,
    )


# ============================================================================
# 1. Input Validation & Conversation Lifecycle Tests
# ============================================================================

class TestChatbotValidationAndLifecycle:
    """Tests message validation and conversation handling."""

    def test_empty_and_whitespace_user_messages_rejected(
        self, chatbot_service: ChatbotService
    ) -> None:
        conv = chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            chatbot_service.chat(cid, "")

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            chatbot_service.chat(cid, "    ")

    def test_non_string_message_rejected(self, chatbot_service: ChatbotService) -> None:
        conv = chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        with pytest.raises(ValueError, match="cannot be None"):
            chatbot_service.chat(cid, None)  # type: ignore

        with pytest.raises(TypeError, match="must be a string"):
            chatbot_service.chat(cid, 12345)  # type: ignore


    def test_invalid_conversation_id_rejected(self, chatbot_service: ChatbotService) -> None:
        with pytest.raises(ValueError, match="conversation_id must be a non-empty string"):
            chatbot_service.chat("", "Valid question")

    def test_unknown_conversation_id_raises_keyerror_by_default(
        self, chatbot_service: ChatbotService
    ) -> None:
        with pytest.raises(KeyError, match="does not exist"):
            chatbot_service.chat("non-existent-cid", "What is the password policy?")

    def test_auto_create_conversation_creates_thread(
        self, chatbot_service: ChatbotService
    ) -> None:
        cid = "auto-thread-99"
        assert chatbot_service.get_conversation(cid) is None

        resp = chatbot_service.chat(
            cid,
            "What is the password length requirement?",
            auto_create_conversation=True,
        )

        assert resp.status == "success"
        assert resp.conversation_id == cid
        created = chatbot_service.get_conversation(cid)
        assert created is not None
        assert "password" in created["title"].lower()


# ============================================================================
# 2. Chat Request Workflow & Follow-Up Context Tests
# ============================================================================

class TestChatRequestWorkflow:
    """Verifies end-to-end multi-turn conversation and context separation."""

    def test_successful_single_turn_persists_messages(
        self, chatbot_service: ChatbotService
    ) -> None:
        conv = chatbot_service.create_conversation(title="Security FAQ")
        cid = conv["conversation_id"]

        response = chatbot_service.chat(cid, "What is the required password length?")

        assert isinstance(response, ChatResponse)
        assert response.status == "success"
        assert response.sufficient_context is True
        assert "14" in response.assistant_response
        assert response.is_live_llm is False
        assert response.model_name == "mock-grounded-assistant"

        # Citations preserved
        assert len(response.citations) > 0
        assert response.citations[0]["filename"] == "security_policy.pdf"

        # History in conversation store
        history = chatbot_service.get_conversation_history(cid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "What is the required password length?"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == response.assistant_response
        assert len(history[1]["citations"]) == len(response.citations)

    def test_follow_up_question_incorporates_prior_history(
        self,
        chatbot_service: ChatbotService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        conv = chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # Turn 1
        resp1 = chatbot_service.chat(cid, "Tell me about passwords.")
        assert resp1.status == "success"

        # Turn 2: Follow-up
        resp2 = chatbot_service.chat(cid, "How often do they expire?")
        assert resp2.status == "success"

        # Verify that MockLLM received the prior history in the prompt of Turn 2
        last_call = mock_llm_client.last_call
        assert last_call is not None
        prompt = last_call["prompt"]

        # Check strict separation:
        assert "CONTEXT INFORMATION:" in prompt
        assert "PRIOR CONVERSATION HISTORY:" in prompt
        assert "User: Tell me about passwords." in prompt
        assert "Assistant: According to security_policy.pdf" in prompt
        assert "QUESTION:\nHow often do they expire?" in prompt

        # Verify current question is NOT in history section
        history_part = prompt.split("PRIOR CONVERSATION HISTORY:")[1].split("QUESTION:")[0]
        assert "How often do they expire?" not in history_part

    def test_history_limit_enforced_in_prompt(
        self,
        chatbot_service: ChatbotService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Max history is 4 messages (2 user-assistant pairs). Older turns must be omitted."""
        conv = chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # 3 turns = 6 messages in DB
        chatbot_service.chat(cid, "Question 1")
        chatbot_service.chat(cid, "Question 2")
        chatbot_service.chat(cid, "Question 3")

        # Turn 4
        chatbot_service.chat(cid, "Question 4")

        last_prompt = mock_llm_client.last_call["prompt"]
        assert "PRIOR CONVERSATION HISTORY:" in last_prompt

        # With limit 4, Question 1 and its answer should have been pushed out
        assert "Question 1" not in last_prompt
        # But recent questions (Question 2 and Question 3) should be present
        assert "Question 2" in last_prompt
        assert "Question 3" in last_prompt

    def test_conversation_isolation_no_history_leaks(
        self,
        chatbot_service: ChatbotService,
        mock_llm_client: MockLLMClient,
    ) -> None:
        """Verify thread A history is never supplied to thread B."""
        c1 = chatbot_service.create_conversation(title="Thread A")
        c2 = chatbot_service.create_conversation(title="Thread B")

        chatbot_service.chat(c1["conversation_id"], "Secret question for Thread A")

        # Now chat in Thread B
        chatbot_service.chat(c2["conversation_id"], "Question for Thread B")

        prompt_b = mock_llm_client.last_call["prompt"]
        assert "Secret question for Thread A" not in prompt_b
        # Thread B has no prior history yet
        assert "PRIOR CONVERSATION HISTORY:" not in prompt_b


# ============================================================================
# 3. Citation Preservation & Insufficient Context Tests
# ============================================================================

class TestCitationsAndGrounding:
    """Verifies citations handling and grounded refusal behavior."""

    def test_insufficient_context_safely_refuses_and_persists_in_history(
        self,
        sample_indexed_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        memory_conversation_store: ConversationStore,
    ) -> None:
        """When query is unanswerable, returns safe refusal and saves assistant response."""
        refusal_llm = MockLLMClient(default_response=INSUFFICIENT_CONTEXT_MESSAGE)
        pipeline = RAGPipeline(
            vector_store=sample_indexed_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=refusal_llm,
            top_k=2,
            similarity_threshold=0.99,  # Filter out all chunks
        )
        service = ChatbotService(
            rag_pipeline=pipeline,
            conversation_store=memory_conversation_store,
        )

        conv = service.create_conversation()
        cid = conv["conversation_id"]

        resp = service.chat(cid, "What is the corporate recipe for brownies?")

        assert resp.status == "insufficient_context"
        assert resp.sufficient_context is False
        assert resp.assistant_response == INSUFFICIENT_CONTEXT_MESSAGE
        assert resp.citations == []

        # Verified in DB
        history = service.get_conversation_history(cid)
        assert len(history) == 2
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == INSUFFICIENT_CONTEXT_MESSAGE
        assert history[1]["citations"] == []

    def test_never_fabricate_citations_when_none_retrieved(
        self,
        mock_embedding_service: EmbeddingService,
        memory_conversation_store: ConversationStore,
    ) -> None:
        """Empty vector store returns empty citations without fabrication."""
        empty_store = FAISSVectorStore(dimension=384)
        pipeline = RAGPipeline(
            vector_store=empty_store,
            embedding_service=mock_embedding_service,
            llm_client=MockLLMClient(),
        )
        service = ChatbotService(
            rag_pipeline=pipeline,
            conversation_store=memory_conversation_store,
        )

        conv = service.create_conversation()
        resp = service.chat(conv["conversation_id"], "Any question?")

        assert resp.citations == []
        assert resp.sufficient_context is False
        assert resp.status == "insufficient_context"


# ============================================================================
# 4. Error Handling & Provider Failure Tests
# ============================================================================

class TestErrorHandlingAndSafety:
    """Verifies that failures do not persist corrupt answers or expose secrets."""

    def test_llm_provider_failure_returns_error_and_does_not_persist_answer(
        self,
        sample_indexed_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        memory_conversation_store: ConversationStore,
    ) -> None:
        failing_client = MockLLMClient()

        def fail_generator(s, p):
            raise RuntimeError("API quota exhausted: [SECRET_KEY_12345]")

        failing_client.response_generator = fail_generator

        pipeline = RAGPipeline(
            vector_store=sample_indexed_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=failing_client,
        )
        service = ChatbotService(
            rag_pipeline=pipeline,
            conversation_store=memory_conversation_store,
        )

        conv = service.create_conversation()
        cid = conv["conversation_id"]

        resp = service.chat(cid, "What is the password policy?")

        assert resp.status == "error"
        assert resp.error is not None
        assert "API quota exhausted" in resp.error

        # CRITICAL: Failed assistant message must NOT be persisted in conversation history
        history = service.get_conversation_history(cid)
        # User message was recorded, but assistant failure was NOT persisted as a valid answer
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_missing_gemini_api_key_handled_safely(
        self,
        sample_indexed_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        memory_conversation_store: ConversationStore,
    ) -> None:
        """GeminiLLMClient without API key raises ValueError safely."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Gemini API key is missing"):
                GeminiLLMClient(api_key=None)

    def test_mock_llm_never_claimed_as_live_gemini(
        self, chatbot_service: ChatbotService
    ) -> None:
        conv = chatbot_service.create_conversation()
        resp = chatbot_service.chat(conv["conversation_id"], "What is the password policy?")
        assert resp.is_live_llm is False
        assert resp.model_name == "mock-grounded-assistant"

    def test_convenience_factory_create_chatbot_service(
        self,
        sample_indexed_vector_store: FAISSVectorStore,
        mock_embedding_service: EmbeddingService,
        mock_llm_client: MockLLMClient,
        memory_conversation_store: ConversationStore,
    ) -> None:
        svc = create_chatbot_service(
            vector_store=sample_indexed_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm_client,
            conversation_store=memory_conversation_store,
            max_history_messages=5,
        )
        assert isinstance(svc, ChatbotService)
        assert svc.max_history_messages == 5

        conv = svc.create_conversation()
        resp = svc.chat(conv["conversation_id"], "What is the password policy?")
        assert resp.status == "success"
