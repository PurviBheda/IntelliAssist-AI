"""Unit and integration tests for IntelliAssist AI API Server."""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import pytest
from fastapi.testclient import TestClient

from app.server import app, conversation_store, vector_store, uploads_dir


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client: TestClient):
    """Verify health check returns service information."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "IntelliAssist AI"
    assert "indexed_chunks" in data
    assert "is_live_llm" in data


def test_create_and_list_conversations(client: TestClient):
    """Verify conversation creation and retrieval via API."""
    create_resp = client.post(
        "/api/conversations",
        json={"title": "API Test Conversation"},
    )
    assert create_resp.status_code == 200
    conv = create_resp.json()
    cid = conv["conversation_id"]
    assert conv["title"] == "API Test Conversation"
    assert "documents" in conv

    # List conversations
    list_resp = client.get("/api/conversations")
    assert list_resp.status_code == 200
    conversations = list_resp.json()
    assert any(c["conversation_id"] == cid for c in conversations)

    # Get conversation details
    detail_resp = client.get(f"/api/conversations/{cid}")
    assert detail_resp.status_code == 200
    details = detail_resp.json()
    assert details["conversation_id"] == cid
    assert "messages" in details
    assert "documents" in details


def test_upload_document_txt(client: TestClient):
    """Verify uploading a plain text document, indexing, and linking to conversation."""
    # Create conversation first
    c_resp = client.post("/api/conversations", json={"title": "Doc QA Thread"})
    cid = c_resp.json()["conversation_id"]

    sample_content = (
        "Project Andromeda Security Protocol.\n"
        "All engineers must complete security training annually by December 15.\n"
        "Multi-factor authentication is mandatory on all company laptops and mobile devices."
    )
    file_bytes = io.BytesIO(sample_content.encode("utf-8"))

    upload_resp = client.post(
        "/api/upload",
        files={"file": ("security_protocol.txt", file_bytes, "text/plain")},
        data={"conversation_id": cid},
    )
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    assert data["status"] == "success"
    assert "document" in data
    doc = data["document"]
    assert doc["filename"] == "security_protocol.txt"
    assert doc["file_type"] == "txt"
    assert doc["total_pages"] == 1
    assert doc["chunk_count"] > 0

    # Verify document is linked to conversation
    detail_resp = client.get(f"/api/conversations/{cid}")
    conv_docs = detail_resp.json()["documents"]
    assert len(conv_docs) > 0
    assert any(d["filename"] == "security_protocol.txt" for d in conv_docs)


def test_upload_invalid_file_extension(client: TestClient):
    """Verify rejection of unsupported extensions (e.g. .exe, .jpg)."""
    fake_bytes = io.BytesIO(b"binary data")
    upload_resp = client.post(
        "/api/upload",
        files={"file": ("malicious.exe", fake_bytes, "application/octet-stream")},
    )
    assert upload_resp.status_code == 400
    assert "Unsupported file format" in upload_resp.json()["detail"]


def test_upload_empty_file(client: TestClient):
    """Verify rejection of empty file (0 bytes)."""
    empty_bytes = io.BytesIO(b"")
    upload_resp = client.post(
        "/api/upload",
        files={"file": ("empty.txt", empty_bytes, "text/plain")},
    )
    assert upload_resp.status_code == 400
    assert "empty" in upload_resp.json()["detail"].lower()


def test_chat_turn_and_history(client: TestClient):
    """Verify conversational turn, RAG or intent detection, and message persistence."""
    conv_resp = client.post("/api/conversations", json={"title": "Chat Test"})
    cid = conv_resp.json()["conversation_id"]

    # Send a greeting
    chat_resp = client.post(
        "/api/chat",
        json={"conversation_id": cid, "message": "Hello! How can you help me?"},
    )
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert chat_data["conversation_id"] == cid
    assert len(chat_data["assistant_response"]) > 0

    # Check conversation messages
    detail_resp = client.get(f"/api/conversations/{cid}")
    messages = detail_resp.json()["messages"]
    assert len(messages) >= 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_delete_conversation(client: TestClient):
    """Verify deleting a conversation removes it completely."""
    conv_resp = client.post("/api/conversations", json={"title": "To Delete"})
    cid = conv_resp.json()["conversation_id"]

    del_resp = client.delete(f"/api/conversations/{cid}")
    assert del_resp.status_code == 200

    # Getting deleted conversation returns 404
    get_resp = client.get(f"/api/conversations/{cid}")
    assert get_resp.status_code == 404
