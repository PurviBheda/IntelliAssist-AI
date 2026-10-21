"""Automated integration tests for IntelliAssist AI ChatbotService with Intent Detection.

Verifies end-to-end integration:
  1. Greetings route to conversational handler without invoking RAG or vector retrieval.
  2. Sentiment requests invoke SentimentAnalyzer and return calibrated metrics.
  3. Inline text summarization invokes TextSummarizer.
  4. Document summarization retrieves relevant chunks and preserves source citations.
  5. Conversation history summarization recaps prior conversational turns.
  6. Out-of-scope requests receive polite scope explanations without crashing.
  7. Document questions continue using the grounded RAG pipeline.
  8. Refusal on insufficient context is preserved for factual document questions.
  9. Grounding requirements are never bypassed for document inquiries.
  10. Multi-turn dialogue seamlessly mixing multiple intents in a single thread.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pytest
from unittest.mock import MagicMock

from app.chatbot_service import (
    ChatResponse,
    ChatbotService,
    create_chatbot_service,
)
from app.conversation_store import ConversationStore
from app.embedding_service import EmbeddingService
from app.intent_detector import IntentDetector, IntentType
from app.rag_pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    MockLLMClient,
    RAGPipeline,
)
from app.sentiment_analyzer import SentimentAnalyzer
from app.summarizer import TextSummarizer
from app.vector_store import FAISSVectorStore


# ============================================================================
# Test Fixtures
# ============================================================================

def make_deterministic_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic unit vector seeded by text hash."""
    import hashlib
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    service = EmbeddingService(embedding_dimension=384)
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda sentences, **kw: (
        make_deterministic_embedding(sentences)
        if isinstance(sentences, str)
        else np.vstack([make_deterministic_embedding(s) for s in sentences])
    )
    service._model = mock_model
    return service


@pytest.fixture
def indexed_vector_store(mock_embedding_service: EmbeddingService) -> FAISSVectorStore:
    """Vector store with security policy and handbook sample chunks."""
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

    embedded = []
    for c in chunks:
        copy_c = dict(c)
        copy_c["embedding"] = mock_embedding_service.embed_text(c["text"])
        embedded.append(copy_c)

    store.add_chunks(embedded)
    return store


@pytest.fixture
def mock_rag_llm() -> MockLLMClient:
    import re

    def generator(sys_inst: str, prompt: str) -> str:
        q_match = re.search(r"QUESTION:\s*([^\n]+)", prompt)
        q_lower = q_match.group(1).lower() if q_match else prompt.lower()
        if "pto" in q_lower or "paid time off" in q_lower:
            return "According to employee_handbook.txt (Page 1), employees receive 20 days of PTO."
        if "password" in q_lower or "14" in q_lower:
            return "According to security_policy.pdf (Page 1), passwords must be at least 14 characters."
        return "Based on the provided context, policies are detailed in security_policy.pdf."

    return MockLLMClient(response_generator=generator, model_name="mock-grounded-assistant")


@pytest.fixture
def integrated_chatbot_service(
    indexed_vector_store: FAISSVectorStore,
    mock_embedding_service: EmbeddingService,
    mock_rag_llm: MockLLMClient,
) -> ChatbotService:
    conv_store = ConversationStore(db_path=":memory:")
    pipeline = RAGPipeline(
        vector_store=indexed_vector_store,
        embedding_service=mock_embedding_service,
        llm_client=mock_rag_llm,
        top_k=2,
    )
    return ChatbotService(
        rag_pipeline=pipeline,
        conversation_store=conv_store,
        max_history_messages=6,
    )


# ============================================================================
# 1. Intent Routing Tests
# ============================================================================

class TestChatbotIntentRouting:
    """Verifies that user prompts are routed to the appropriate functional handler."""

    def test_chatbot_routes_greeting_without_calling_rag(
        self, integrated_chatbot_service: ChatbotService, mock_rag_llm: MockLLMClient
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        resp = integrated_chatbot_service.chat(cid, "Hello! Good morning.")

        assert resp.status == "success"
        assert resp.intent == IntentType.GREETING.value
        assert resp.citations == []
        assert resp.retrieved_chunks == []
        assert "IntelliAssist AI" in resp.assistant_response
        # Verify RAG LLM was NOT invoked
        assert mock_rag_llm.call_count == 0

        # Verify persisted in conversation store
        history = integrated_chatbot_service.get_conversation_history(cid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_chatbot_routes_sentiment_inline_text(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        prompt = "Analyze the sentiment of: 'The new system is remarkably fast and very helpful!'"
        resp = integrated_chatbot_service.chat(cid, prompt)

        assert resp.status == "success"
        assert resp.intent == IntentType.SENTIMENT_ANALYSIS.value
        assert resp.citations == []
        assert "Positive" in resp.assistant_response
        assert resp.sentiment_result is not None
        assert resp.sentiment_result["label"] == "positive"
        assert resp.sentiment_result["confidence"] > 0.60

    def test_chatbot_routes_sentiment_prior_message(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # Turn 1: user asks a document question
        integrated_chatbot_service.chat(cid, "What is the password requirement?")

        # Turn 2: asks for sentiment of previous message
        resp = integrated_chatbot_service.chat(cid, "What is the sentiment of the last response?")

        assert resp.status == "success"
        assert resp.intent == IntentType.SENTIMENT_ANALYSIS.value
        assert resp.sentiment_result is not None

    def test_chatbot_routes_summarize_inline_text(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        prompt = (
            "Summarize this: 'Employees must complete annual cybersecurity awareness training by December. "
            "Managers must review team compliance reports monthly. "
            "Failure to complete training will result in access revocation.'"
        )
        resp = integrated_chatbot_service.chat(cid, prompt)

        assert resp.status == "success"
        assert resp.intent == IntentType.SUMMARIZE.value
        assert len(resp.assistant_response) > 0
        assert resp.summary_result is not None

    def test_chatbot_routes_summarize_document_with_citations(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # Request summarization of indexed security policy
        resp = integrated_chatbot_service.chat(cid, "Summarize the security policy document")

        assert resp.status == "success"
        assert resp.intent == IntentType.SUMMARIZE.value
        assert len(resp.assistant_response) > 0
        assert resp.summary_result is not None
        # Citations must be preserved!
        assert len(resp.citations) > 0
        assert resp.citations[0]["filename"] == "security_policy.pdf"

    def test_chatbot_routes_summarize_conversation_history(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # Turn 1
        integrated_chatbot_service.chat(cid, "What is the password length requirement?")
        # Turn 2
        integrated_chatbot_service.chat(cid, "How many days of PTO do employees receive?")

        # Turn 3: Request conversation recap
        resp = integrated_chatbot_service.chat(cid, "Can you summarize our conversation?")

        assert resp.status == "success"
        assert resp.intent == IntentType.SUMMARIZE.value
        assert resp.summary_result is not None
        assert "conversation" in resp.assistant_response.lower()

    def test_chatbot_routes_out_of_scope_politely(
        self, integrated_chatbot_service: ChatbotService, mock_rag_llm: MockLLMClient
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        resp = integrated_chatbot_service.chat(cid, "Write a python script to download videos from youtube")

        assert resp.status == "success"
        assert resp.intent == IntentType.OUT_OF_SCOPE.value
        assert resp.citations == []
        assert "outside my scope" in resp.assistant_response.lower()
        # RAG LLM was not invoked for out-of-scope query
        assert mock_rag_llm.call_count == 0


# ============================================================================
# 2. Document Questions & Grounding Isolation Tests
# ============================================================================

class TestChatbotDocumentQuestionGrounding:
    """Verifies that factual document questions strictly enforce RAG grounding and citations."""

    def test_document_question_continues_using_rag_pipeline(
        self, integrated_chatbot_service: ChatbotService, mock_rag_llm: MockLLMClient
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        resp = integrated_chatbot_service.chat(cid, "What is the required password length?")

        assert resp.status == "success"
        assert resp.intent == IntentType.DOCUMENT_QUESTION.value
        assert resp.sufficient_context is True
        assert "14" in resp.assistant_response
        assert len(resp.citations) > 0
        assert resp.citations[0]["filename"] == "security_policy.pdf"
        assert mock_rag_llm.call_count == 1

    def test_insufficient_context_refusal_preserved(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation()
        cid = conv["conversation_id"]

        # Ask about a topic definitely not in the vector store
        resp = integrated_chatbot_service.chat(
            cid,
            "What is the corporate policy regarding pets in the workplace?",
            similarity_threshold=0.99,
        )

        assert resp.status == "insufficient_context"
        assert resp.sufficient_context is False
        assert resp.assistant_response == INSUFFICIENT_CONTEXT_MESSAGE
        assert resp.citations == []


# ============================================================================
# 3. Multi-Turn Mixed-Intent Conversation Test
# ============================================================================

class TestMultiTurnMixedIntentDialogue:
    """Verifies a cohesive multi-turn session switching smoothly across intents."""

    def test_multi_turn_mixed_intent_conversation(
        self, integrated_chatbot_service: ChatbotService
    ) -> None:
        conv = integrated_chatbot_service.create_conversation(title="Mixed Intent Session")
        cid = conv["conversation_id"]

        # Turn 1: Greeting
        t1 = integrated_chatbot_service.chat(cid, "Hi there!")
        assert t1.intent == IntentType.GREETING.value
        assert t1.status == "success"

        # Turn 2: Document Question
        t2 = integrated_chatbot_service.chat(cid, "How many days of PTO do employees receive?")
        assert t2.intent == IntentType.DOCUMENT_QUESTION.value
        assert "20" in t2.assistant_response
        assert len(t2.citations) > 0

        # Turn 3: Summarize Document
        t3 = integrated_chatbot_service.chat(cid, "Summarize the employee handbook")
        assert t3.intent == IntentType.SUMMARIZE.value
        assert len(t3.citations) > 0

        # Turn 4: Sentiment Analysis
        t4 = integrated_chatbot_service.chat(
            cid, "Analyze the sentiment of: 'The benefits package is outstanding and wonderful!'"
        )
        assert t4.intent == IntentType.SENTIMENT_ANALYSIS.value
        assert t4.sentiment_result["label"] == "positive"

        # Turn 5: Farewell
        t5 = integrated_chatbot_service.chat(cid, "Thank you, goodbye!")
        assert t5.intent == IntentType.GREETING.value
        assert "goodbye" in t5.assistant_response.lower() or "day" in t5.assistant_response.lower()

        # Verify all 10 messages persisted in conversation store chronologically
        history = integrated_chatbot_service.get_conversation_history(cid)
        assert len(history) == 10
        expected_roles = ["user", "assistant"] * 5
        assert [m["role"] for m in history] == expected_roles
