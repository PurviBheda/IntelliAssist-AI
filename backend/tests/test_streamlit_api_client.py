"""Unit tests for IntelliAssist AI Streamlit API Client.

These tests use mocked HTTP transports/responses and do NOT make live network or Gemini calls.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from streamlit_app.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiPayloadError,
    ApiStatusError,
    IntelliAssistApiClient,
)


@pytest.fixture
def mock_session():
    """Create a mock requests.Session."""
    return MagicMock(spec=requests.Session)


@pytest.fixture
def client(mock_session):
    """Create an IntelliAssistApiClient wired to mock_session."""
    return IntelliAssistApiClient(base_url="http://127.0.0.1:8000", session=mock_session)


def create_mock_response(status_code: int = 200, json_data: Any = None, text: str = "") -> requests.Response:
    """Helper to synthesize a mock requests.Response."""
    resp = requests.Response()
    resp.status_code = status_code
    resp.encoding = "utf-8"

    if json_data is not None:
        import json
        resp._content = json.dumps(json_data).encode("utf-8")
    else:
        resp._content = text.encode("utf-8")
    return resp


# ============================================================================
# 1. Initialization and Configuration Tests
# ============================================================================

def test_client_initialization_defaults():
    """Verify default base URL and environment variable fallback."""
    with patch.dict(os.environ, {}, clear=True):
        c = IntelliAssistApiClient()
        assert c.base_url == "http://127.0.0.1:8000"

    with patch.dict(os.environ, {"INTELLIASSIST_API_URL": "http://api.internal:9000/"}):
        c = IntelliAssistApiClient()
        assert c.base_url == "http://api.internal:9000"

    c_custom = IntelliAssistApiClient(base_url="https://remote.server:8443///")
    assert c_custom.base_url == "https://remote.server:8443"


# ============================================================================
# 2. Health Check Tests
# ============================================================================

def test_check_health_success(client, mock_session):
    """Verify health check extracts status dictionary."""
    mock_session.request.return_value = create_mock_response(
        200,
        {
            "status": "healthy",
            "service": "IntelliAssist AI",
            "is_live_llm": True,
            "model_name": "gemini-2.5-flash",
            "indexed_chunks": 14,
        },
    )

    data = client.check_health()
    assert data["status"] == "healthy"
    assert data["is_live_llm"] is True
    assert data["indexed_chunks"] == 14

    mock_session.request.assert_called_once()
    args, kwargs = mock_session.request.call_args
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "http://127.0.0.1:8000/api/health"


def test_is_available_helper(client, mock_session):
    """Verify is_available returns True on healthy, False on error."""
    mock_session.request.return_value = create_mock_response(200, {"status": "healthy"})
    assert client.is_available() is True

    mock_session.request.side_effect = requests.exceptions.ConnectionError("refused")
    assert client.is_available() is False


# ============================================================================
# 3. Document Upload Tests
# ============================================================================

def test_upload_document_success(client, mock_session):
    """Verify multipart file upload and document record extraction."""
    mock_session.request.return_value = create_mock_response(
        200,
        {
            "status": "success",
            "message": "Processed test.pdf",
            "document": {
                "document_id": "doc-123",
                "filename": "test.pdf",
                "file_type": "pdf",
                "total_pages": 3,
                "chunk_count": 6,
            },
        },
    )

    fake_bytes = b"%PDF-1.4 test document content"
    res = client.upload_document(
        filename="test.pdf",
        file_bytes=fake_bytes,
        conversation_id="conv-abc",
        content_type="application/pdf",
    )

    assert res["status"] == "success"
    assert res["document"]["filename"] == "test.pdf"
    assert res["document"]["chunk_count"] == 6

    mock_session.request.assert_called_once()
    _, kwargs = mock_session.request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://127.0.0.1:8000/api/upload"
    assert "file" in kwargs["files"]
    assert kwargs["files"]["file"] == ("test.pdf", fake_bytes, "application/pdf")
    assert kwargs["data"] == {"conversation_id": "conv-abc"}


def test_upload_document_validation(client):
    """Verify rejection of empty filename or empty file content."""
    with pytest.raises(ValueError, match="filename must be a non-empty string"):
        client.upload_document("", b"data")

    with pytest.raises(ValueError, match="file_bytes cannot be empty"):
        client.upload_document("test.pdf", b"")


# ============================================================================
# 4. Chat Message Tests
# ============================================================================

def test_send_chat_message_success(client, mock_session):
    """Verify sending a chat prompt returns full structured assistant response."""
    mock_session.request.return_value = create_mock_response(
        200,
        {
            "conversation_id": "conv-1",
            "user_message": "What is the password policy?",
            "assistant_response": "Based on provided documents, passwords must be 14 characters.",
            "citations": [
                {
                    "filename": "security_policy.txt",
                    "page_number": 1,
                    "similarity_score": 0.88,
                }
            ],
            "sufficient_context": True,
            "status": "success",
            "intent": "document_question",
            "sentiment_result": None,
            "summary_result": None,
        },
    )

    resp = client.send_chat_message(
        conversation_id="conv-1",
        message="What is the password policy?",
        auto_create_conversation=True,
    )

    assert resp["conversation_id"] == "conv-1"
    assert "14 characters" in resp["assistant_response"]
    assert len(resp["citations"]) == 1
    assert resp["citations"][0]["filename"] == "security_policy.txt"
    assert resp["intent"] == "document_question"

    mock_session.request.assert_called_once()
    _, kwargs = mock_session.request.call_args
    assert kwargs["method"] == "POST"
    assert kwargs["url"] == "http://127.0.0.1:8000/api/chat"
    assert kwargs["json"] == {
        "conversation_id": "conv-1",
        "message": "What is the password policy?",
        "auto_create_conversation": True,
    }


def test_send_chat_message_validation(client):
    """Verify validation of conversation_id and message."""
    with pytest.raises(ValueError, match="conversation_id must be a non-empty string"):
        client.send_chat_message("", "hello")

    with pytest.raises(ValueError, match="message must be a non-empty string"):
        client.send_chat_message("c1", "   ")


# ============================================================================
# 5. Conversation Management Tests
# ============================================================================

def test_list_conversations_success(client, mock_session):
    """Verify listing conversation threads with query parameters."""
    mock_session.request.return_value = create_mock_response(
        200,
        [
            {"conversation_id": "c1", "title": "Security Query", "message_count": 2},
            {"conversation_id": "c2", "title": "Policy Summary", "message_count": 4},
        ],
    )

    convs = client.list_conversations(limit=10, offset=0, starred=True, shared=False)
    assert len(convs) == 2
    assert convs[0]["conversation_id"] == "c1"

    _, kwargs = mock_session.request.call_args
    assert kwargs["method"] == "GET"
    assert kwargs["url"] == "http://127.0.0.1:8000/api/conversations"
    assert kwargs["params"] == {
        "limit": 10,
        "offset": 0,
        "starred": "true",
        "shared": "false",
    }


def test_get_conversation_success(client, mock_session):
    """Verify retrieving full conversation thread."""
    mock_session.request.return_value = create_mock_response(
        200,
        {
            "conversation_id": "c1",
            "title": "Security Query",
            "documents": [{"filename": "doc.pdf"}],
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
        },
    )

    conv = client.get_conversation("c1")
    assert conv["conversation_id"] == "c1"
    assert len(conv["messages"]) == 2


def test_create_conversation_success(client, mock_session):
    """Verify creating a conversation thread."""
    mock_session.request.return_value = create_mock_response(
        200,
        {"conversation_id": "c9", "title": "New Topic", "documents": []},
    )

    conv = client.create_conversation(title="New Topic")
    assert conv["conversation_id"] == "c9"
    assert conv["title"] == "New Topic"


def test_delete_and_star_conversation(client, mock_session):
    """Verify delete and star conversation endpoints."""
    mock_session.request.return_value = create_mock_response(
        200, {"status": "success", "conversation_id": "c1", "deleted": True}
    )
    del_res = client.delete_conversation("c1")
    assert del_res["deleted"] is True

    mock_session.request.return_value = create_mock_response(
        200, {"conversation_id": "c1", "is_starred": True}
    )
    star_res = client.star_conversation("c1", is_starred=True)
    assert star_res["is_starred"] is True


# ============================================================================
# 6. Error and Failure Handling Tests
# ============================================================================

def test_connection_failure_raises_api_connection_error(client, mock_session):
    """Verify connection refusal raises ApiConnectionError with user-safe message."""
    mock_session.request.side_effect = requests.exceptions.ConnectionError("Connection refused")

    with pytest.raises(ApiConnectionError) as exc_info:
        client.check_health()

    assert "Could not connect to IntelliAssist backend" in str(exc_info.value)
    assert "8000" in str(exc_info.value)


def test_timeout_raises_api_connection_error(client, mock_session):
    """Verify connection timeout raises ApiConnectionError."""
    mock_session.request.side_effect = requests.exceptions.Timeout("Read timed out")

    with pytest.raises(ApiConnectionError) as exc_info:
        client.check_health()

    assert "timed out" in str(exc_info.value).lower()


def test_http_400_error_raises_api_status_error(client, mock_session):
    """Verify 400 bad request returns detailed detail message from server."""
    mock_session.request.return_value = create_mock_response(
        400,
        {"detail": "Unsupported file format '.exe'. Supported formats are: .docx, .pdf, .txt"},
    )

    with pytest.raises(ApiStatusError) as exc_info:
        client.upload_document("malicious.exe", b"binary")

    err = exc_info.value
    assert err.status_code == 400
    assert "Unsupported file format" in err.detail


def test_http_404_error_raises_api_status_error(client, mock_session):
    """Verify 404 not found raises ApiStatusError."""
    mock_session.request.return_value = create_mock_response(
        404,
        {"detail": "Conversation 'c-missing' not found."},
    )

    with pytest.raises(ApiStatusError) as exc_info:
        client.get_conversation("c-missing")

    err = exc_info.value
    assert err.status_code == 404
    assert "not found" in err.detail.lower()


def test_http_500_with_plain_text(client, mock_session):
    """Verify 500 internal server error with plain text response."""
    mock_session.request.return_value = create_mock_response(500, text="Internal Server Error")

    with pytest.raises(ApiStatusError) as exc_info:
        client.check_health()

    assert exc_info.value.status_code == 500
    assert "Internal Server Error" in str(exc_info.value)


def test_invalid_json_payload_raises_api_payload_error(client, mock_session):
    """Verify malformed JSON payload raises ApiPayloadError."""
    resp = requests.Response()
    resp.status_code = 200
    resp._content = b"{malformed json"
    mock_session.request.return_value = resp

    with pytest.raises(ApiPayloadError) as exc_info:
        client.check_health()

    assert "Failed to decode server JSON response" in str(exc_info.value)


def test_unexpected_payload_structure_raises_api_payload_error(client, mock_session):
    """Verify unexpected schema structure raises ApiPayloadError."""
    # List instead of dict for chat
    mock_session.request.return_value = create_mock_response(200, ["unexpected", "list"])

    with pytest.raises(ApiPayloadError, match="Server response did not include assistant response"):
        client.send_chat_message("c1", "hello")
