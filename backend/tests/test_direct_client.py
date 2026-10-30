"""Unit and integration tests for DirectServiceAdapter (single-process Streamlit mode).

Verifies:
1. Public method signatures and return models match IntelliAssistApiClient.
2. User-safe error handling (ApiStatusError, ApiClientError, ValueError).
3. Session isolation between independent DirectServiceAdapter instances.
4. Process-wide caching of shared models (SentenceTransformers, LLM).
5. Document upload, vector indexing, chat grounding, citations, and temporary file cleanup.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from app.chatbot_service import create_chatbot_service
from app.conversation_store import ConversationStore
from app.embedding_service import EmbeddingService
from app.rag_pipeline import MockLLMClient
from app.vector_store import FAISSVectorStore
from streamlit_app.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiPayloadError,
    ApiStatusError,
)
from streamlit_app.direct_client import (
    DirectServiceAdapter,
    get_direct_client,
    get_shared_embedding_service,
    get_shared_llm_client,
    is_in_streamlit_session,
)


@pytest.fixture
def mock_llm():
    """Deterministic Mock LLM returning grounded answers for testing."""
    return MockLLMClient(default_response="The corporate wifi password is BlueSky2026.")


@pytest.fixture
def adapter(mock_llm):
    """DirectServiceAdapter configured with MockLLM and transient in-memory stores."""
    # Shared embedding service
    emb_svc = get_shared_embedding_service()
    vs = FAISSVectorStore(dimension=384, normalize_vectors=True)
    cs = ConversationStore(db_path=":memory:")
    bot = create_chatbot_service(
        vector_store=vs,
        embedding_service=emb_svc,
        llm_client=mock_llm,
        conversation_store=cs,
    )
    return DirectServiceAdapter(
        vector_store=vs,
        conversation_store=cs,
        chatbot_service=bot,
        embedding_service=emb_svc,
        llm_client=mock_llm,
    )


# ============================================================================
# 1. Initialization and Health Tests
# ============================================================================

def test_adapter_initialization_defaults():
    """Verify default initialization without explicit arguments."""
    ad = DirectServiceAdapter(llm_client=MockLLMClient())
    assert ad.base_url == "direct://in-process"
    assert ad.vector_store is not None
    assert ad.conversation_store is not None
    assert ad.chatbot_service is not None
    assert ad.is_available() is True


def test_check_health_schema(adapter):
    """Verify health check returns complete expected dictionary schema."""
    health = adapter.check_health()
    assert health["status"] == "healthy"
    assert health["service"] == "IntelliAssist AI"
    assert "is_live_llm" in health
    assert "model_name" in health
    assert "indexed_chunks" in health
    assert health["indexed_chunks"] == 0
    assert "timestamp" in health


def test_is_available(adapter):
    """Verify is_available returns True for healthy operational adapter."""
    assert adapter.is_available() is True


# ============================================================================
# 2. Conversation Management Tests
# ============================================================================

def test_create_and_list_conversations(adapter):
    """Verify creating threads and retrieving the list of conversations."""
    conv1 = adapter.create_conversation(title="Meeting Notes Discussion")
    assert conv1["conversation_id"] is not None
    assert conv1["title"] == "Meeting Notes Discussion"
    assert conv1["documents"] == []

    conv2 = adapter.create_conversation(title="Quarterly Review")
    conv_list = adapter.list_conversations(limit=10)
    assert len(conv_list) == 2
    titles = [c["title"] for c in conv_list]
    assert "Meeting Notes Discussion" in titles
    assert "Quarterly Review" in titles


def test_get_conversation_success_and_not_found(adapter):
    """Verify retrieving thread details and 404 error when not found."""
    conv = adapter.create_conversation(title="Product Spec")
    cid = conv["conversation_id"]

    retrieved = adapter.get_conversation(cid)
    assert retrieved["conversation_id"] == cid
    assert retrieved["title"] == "Product Spec"
    assert "messages" in retrieved
    assert "documents" in retrieved

    # Test not found raises ApiStatusError with 404
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.get_conversation("nonexistent-conv-id")
    assert exc_info.value.status_code == 404


def test_delete_conversation_success_and_not_found(adapter):
    """Verify deleting conversation and 404 error when nonexistent."""
    conv = adapter.create_conversation(title="Temporary Thread")
    cid = conv["conversation_id"]

    del_resp = adapter.delete_conversation(cid)
    assert del_resp["status"] == "success"
    assert del_resp["deleted"] is True

    # Confirm thread is gone
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.get_conversation(cid)
    assert exc_info.value.status_code == 404

    # Deleting again raises 404
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.delete_conversation(cid)
    assert exc_info.value.status_code == 404


def test_star_conversation(adapter):
    """Verify toggling and explicitly setting starred conversation state."""
    conv = adapter.create_conversation(title="Starred Conversation")
    cid = conv["conversation_id"]

    # Toggle to True
    res1 = adapter.star_conversation(cid)
    assert res1["is_starred"] is True

    # Toggle to False
    res2 = adapter.star_conversation(cid)
    assert res2["is_starred"] is False

    # Explicit set to True
    res3 = adapter.star_conversation(cid, is_starred=True)
    assert res3["is_starred"] is True

    # Nonexistent conversation raises 404
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.star_conversation("missing-conv-id")
    assert exc_info.value.status_code == 404


# ============================================================================
# 3. Document Ingestion & Vector Indexing Tests
# ============================================================================

def test_upload_document_txt_success(adapter):
    """Verify uploading a text document, indexing chunks, and linking to thread."""
    conv = adapter.create_conversation(title="Security Policies")
    cid = conv["conversation_id"]

    text_content = (
        "Internal Corporate Network Security Standard.\n"
        "Section 1: The primary internal Wi-Fi SSID is CorpSecure-5G.\n"
        "Section 2: The corporate wifi password is BlueSky2026 for all approved staff.\n"
        "Section 3: Guest network access expires after 8 hours of continuous connection."
    )
    raw_bytes = text_content.encode("utf-8")

    upload_res = adapter.upload_document(
        filename="network_security.txt",
        file_bytes=raw_bytes,
        conversation_id=cid,
        content_type="text/plain",
    )

    assert upload_res["status"] == "success"
    assert "document" in upload_res
    doc = upload_res["document"]
    assert doc["filename"] == "network_security.txt"
    assert doc["file_type"] == "txt"
    assert doc["total_pages"] == 1
    assert doc["chunk_count"] > 0
    assert adapter.vector_store.total_vectors > 0

    # Verify conversation now includes this document
    conv_data = adapter.get_conversation(cid)
    assert len(conv_data["documents"]) == 1
    assert conv_data["documents"][0]["filename"] == "network_security.txt"


def test_upload_document_validation_errors(adapter):
    """Verify validation checks on filename, file bytes, and unsupported extensions."""
    # Empty filename
    with pytest.raises(ValueError):
        adapter.upload_document(filename="", file_bytes=b"content")

    # Empty bytes
    with pytest.raises(ValueError):
        adapter.upload_document(filename="test.txt", file_bytes=b"")

    # Unsupported file extension (e.g. .exe)
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.upload_document(filename="virus.exe", file_bytes=b"some bytes")
    assert exc_info.value.status_code == 400
    assert "Unsupported file format" in exc_info.value.detail

    # Oversized document (> 25MB)
    with patch("streamlit_app.direct_client.MAX_FILE_SIZE_BYTES", 10):
        with pytest.raises(ApiStatusError) as exc_info:
            adapter.upload_document(filename="huge.txt", file_bytes=b"0123456789ABCDEF")
        assert exc_info.value.status_code == 400
        assert "exceeds maximum allowed size" in exc_info.value.detail


def test_upload_tempfile_cleanup(adapter):
    """Verify temporary files are completely cleaned up after upload processing."""
    created_temp_paths = []
    real_named_temp = tempfile.NamedTemporaryFile

    def recording_tempfile(*args, **kwargs):
        res = real_named_temp(*args, **kwargs)
        created_temp_paths.append(Path(res.name))
        return res

    with patch("streamlit_app.direct_client.tempfile.NamedTemporaryFile", side_effect=recording_tempfile):
        adapter.upload_document(
            filename="temp_check.txt",
            file_bytes=b"Simple temporary document test.",
        )

    assert len(created_temp_paths) > 0
    for p in created_temp_paths:
        assert not p.exists(), f"Temporary file {p} was not cleaned up!"


# ============================================================================
# 4. Chat Message & RAG Pipeline Tests
# ============================================================================

def test_send_chat_message_greeting(adapter):
    """Verify chat handles polite greetings without querying vector store."""
    conv = adapter.create_conversation(title="Chat Session")
    cid = conv["conversation_id"]

    resp = adapter.send_chat_message(
        conversation_id=cid,
        message="Hello",
    )
    assert resp["conversation_id"] == cid
    assert "IntelliAssist AI" in resp["assistant_response"]
    assert resp["intent"] in {"greeting", "bot_persona"}


def test_send_chat_message_with_document_grounding(adapter):
    """Verify asking questions about uploaded documents retrieves citations and answers."""
    conv = adapter.create_conversation(title="QA Thread")
    cid = conv["conversation_id"]

    policy_text = (
        "Project Orion Travel Reimbursement Guidelines.\n"
        "Hotel lodging expenses are capped at $250 per night in major metropolitan areas.\n"
        "Meal allowances are fixed at $75 per diem without itemized receipts required.\n"
        "All expense reports must be submitted within 30 days of trip conclusion."
    )
    adapter.upload_document(
        filename="travel_policy.txt",
        file_bytes=policy_text.encode("utf-8"),
        conversation_id=cid,
    )

    resp = adapter.send_chat_message(
        conversation_id=cid,
        message="What is the hotel lodging cap per night?",
    )
    assert resp["conversation_id"] == cid
    assert resp["assistant_response"] is not None
    assert len(resp["assistant_response"]) > 0
    assert len(resp["citations"]) > 0
    assert resp["citations"][0]["filename"] == "travel_policy.txt"


def test_send_chat_message_validation(adapter):
    """Verify validation on empty conversation_id and prompt message."""
    with pytest.raises(ValueError):
        adapter.send_chat_message(conversation_id="", message="Hello")

    with pytest.raises(ValueError):
        adapter.send_chat_message(conversation_id="conv-1", message="   ")

    with pytest.raises(ApiStatusError) as exc_info:
        adapter.send_chat_message(
            conversation_id="missing-id",
            message="Test",
            auto_create_conversation=False,
        )
    assert exc_info.value.status_code == 404


# ============================================================================
# 5. Document Removal Tests
# ============================================================================

def test_remove_document_success_and_not_found(adapter):
    """Verify removing a document detaches it from the conversation and syncs vectors."""
    conv = adapter.create_conversation(title="Doc Removal Test")
    cid = conv["conversation_id"]

    upload_res = adapter.upload_document(
        filename="file_to_remove.txt",
        file_bytes=b"This text will be indexed then removed.",
        conversation_id=cid,
    )
    doc_id = upload_res["document"]["document_id"]
    assert adapter.vector_store.total_vectors > 0

    rem_resp = adapter.remove_document(conversation_id=cid, document_id=doc_id)
    assert rem_resp["status"] == "success"
    assert rem_resp["document_id"] == doc_id

    # Verify document no longer appears in conversation
    conv_data = adapter.get_conversation(cid)
    assert len(conv_data["documents"]) == 0

    # Removing again raises 404
    with pytest.raises(ApiStatusError) as exc_info:
        adapter.remove_document(conversation_id=cid, document_id=doc_id)
    assert exc_info.value.status_code == 404


# ============================================================================
# 6. Session Isolation Verification Tests
# ============================================================================

def test_strict_session_isolation():
    """Verify that two independent DirectServiceAdapter sessions never leak data."""
    # Simulate User Session 1
    session_1 = DirectServiceAdapter(llm_client=MockLLMClient())
    conv_1 = session_1.create_conversation(title="Session 1 Secret Thread")
    cid_1 = conv_1["conversation_id"]

    session_1.upload_document(
        filename="confidential_session1.txt",
        file_bytes=b"CONFIDENTIAL: The vault combination is 44-88-22.",
        conversation_id=cid_1,
    )
    assert session_1.vector_store.total_vectors > 0

    # Simulate User Session 2
    session_2 = DirectServiceAdapter(llm_client=MockLLMClient())

    # 1. Verify Session 2 has an empty conversation list
    s2_convs = session_2.list_conversations()
    assert len(s2_convs) == 0
    assert not any(c["conversation_id"] == cid_1 for c in s2_convs)

    # 2. Verify Session 2 cannot access Session 1's conversation
    with pytest.raises(ApiStatusError) as exc_info:
        session_2.get_conversation(cid_1)
    assert exc_info.value.status_code == 404

    # 3. Verify Session 2 vector store is completely empty
    assert session_2.vector_store.total_vectors == 0

    # 4. Verify Session 2 cannot retrieve Session 1's vault combination
    conv_2 = session_2.create_conversation(title="Session 2 Thread")
    cid_2 = conv_2["conversation_id"]
    query_resp = session_2.send_chat_message(
        conversation_id=cid_2,
        message="What is the vault combination?",
    )
    # Session 2 has no documents, so sufficient_context must be False
    assert query_resp["sufficient_context"] is False
    assert "44-88-22" not in query_resp["assistant_response"]


def test_shared_resource_caching():
    """Verify that sentence-transformers and LLM instances are cached process-wide."""
    emb1 = get_shared_embedding_service()
    emb2 = get_shared_embedding_service()
    assert emb1 is emb2, "EmbeddingService was not cached process-wide!"

    llm1 = get_shared_llm_client()
    llm2 = get_shared_llm_client()
    assert llm1 is llm2, "LLMClient was not cached process-wide!"

