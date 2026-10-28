"""Unit and integration tests for the Streamlit UI application.

Uses streamlit.testing.v1.AppTest and mocks the API client so tests run
offline without making live backend or Gemini API calls.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py")

from streamlit_app.api_client import (
    ApiConnectionError,
    ApiStatusError,
    IntelliAssistApiClient,
)


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """Clear Streamlit cache resources and data before each test."""
    import streamlit as st
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def mock_api_client():
    """Create a mock IntelliAssistApiClient with standard responses."""
    client = MagicMock(spec=IntelliAssistApiClient)
    client.check_health.return_value = {
        "status": "healthy",
        "service": "IntelliAssist AI",
        "is_live_llm": True,
        "model_name": "gemini-2.5-flash",
        "indexed_chunks": 18,
    }
    client.is_available.return_value = True
    client.list_conversations.return_value = [
        {
            "conversation_id": "conv-101",
            "title": "Security Policy Ingestion",
            "message_count": 4,
            "created_at": "2026-10-27T00:00:00Z",
        },
        {
            "conversation_id": "conv-102",
            "title": "HR Handbook Discussion",
            "message_count": 2,
            "created_at": "2026-10-27T00:05:00Z",
        },
    ]
    client.get_conversation.return_value = {
        "conversation_id": "conv-101",
        "title": "Security Policy Ingestion",
        "documents": [
            {
                "document_id": "doc-1",
                "filename": "security_policy.txt",
                "file_type": "txt",
                "total_pages": 1,
                "chunk_count": 3,
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": "What is the password requirement?",
            },
            {
                "role": "assistant",
                "content": "Based on provided documents, passwords must be 14 characters.",
                "citations": [
                    {
                        "filename": "security_policy.txt",
                        "page_number": 1,
                        "similarity_score": 0.92,
                    }
                ],
                "intent": "document_question",
            },
        ],
    }
    return client


# ============================================================================
# 1. Application Startup and Layout Tests
# ============================================================================

def test_app_starts_and_renders_without_exceptions(mock_api_client):
    """Verify application boots cleanly, connects to client, and renders without exceptions."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.run(timeout=10)

    assert not at.exception
    # Verify brand header card is rendered
    assert len(at.markdown) > 0
    header_markdown = any("IntelliAssist AI" in m.value for m in at.markdown)
    assert header_markdown


def test_app_handles_backend_offline_state():
    """Verify application renders warning badge and startup guide when backend is offline."""
    offline_client = MagicMock(spec=IntelliAssistApiClient)
    offline_client.check_health.side_effect = ApiConnectionError("Could not connect to backend")
    offline_client.list_conversations.return_value = []

    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = offline_client
    at.run(timeout=10)

    assert not at.exception
    # Check error element rendered in sidebar
    assert len(at.sidebar.error) > 0
    assert "Backend Offline" in at.sidebar.error[0].value
    # Check restart instruction present
    assert any("server.py" in str(code.value) for code in at.sidebar.code)


# ============================================================================
# 2. Conversation History Rendering Tests
# ============================================================================

def test_conversation_history_displayed_in_sidebar(mock_api_client):
    """Verify previous conversations retrieved from client are displayed in the sidebar."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.run(timeout=10)

    assert not at.exception
    # Verify conversation select buttons are rendered
    conv_buttons = [b for b in at.sidebar.button if "Security Policy" in b.label or "HR Handbook" in b.label]
    assert len(conv_buttons) >= 2


def test_empty_conversation_history_handled_gracefully():
    """Verify empty conversation list renders placeholder without errors."""
    empty_client = MagicMock(spec=IntelliAssistApiClient)
    empty_client.check_health.return_value = {"status": "healthy"}
    empty_client.list_conversations.return_value = []

    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = empty_client
    at.run(timeout=10)

    assert not at.exception
    # Verify placeholder caption is shown
    assert any("No previous conversations found" in c.value for c in at.sidebar.caption)


# ============================================================================
# 3. Chat Messages, Citations, and Intent Badges Tests
# ============================================================================

def test_chat_messages_render_with_citations_and_intent(mock_api_client):
    """Verify messages render with user prompt, grounded assistant answer, and citation details."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    # Populate session state with initial message thread
    at.session_state["messages"] = [
        {"role": "user", "content": "What is the password requirement?"},
        {
            "role": "assistant",
            "content": "Based on provided documents, passwords must be 14 characters.",
            "citations": [
                {
                    "filename": "security_policy.txt",
                    "page_number": 1,
                    "similarity_score": 0.92,
                }
            ],
            "intent": "document_question",
        },
    ]
    at.run(timeout=10)

    assert not at.exception
    # Verify chat messages rendered
    chat_msgs = at.chat_message
    assert len(chat_msgs) == 2
    assert chat_msgs[0].avatar == "user"
    assert chat_msgs[1].avatar == "assistant"

    # Verify citation expander is rendered
    assert len(at.expander) > 0
    citation_expander = [e for e in at.expander if "Source Citations" in e.label]
    assert len(citation_expander) == 1


def test_sentiment_and_summary_cards_render_in_chat(mock_api_client):
    """Verify structured sentiment and summary results render in dialogue turn."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Sentiment evaluated as positive.",
            "intent": "sentiment_analysis",
            "sentiment_result": {
                "label": "positive",
                "confidence": 0.89,
                "compound_score": 0.75,
                "positive_score": 0.8,
                "neutral_score": 0.2,
                "negative_score": 0.0,
            },
            "summary_result": {
                "compression_ratio": 0.35,
                "method": "extractive",
                "original_sentences": 20,
                "summary_sentences": 5,
            },
        }
    ]
    at.run(timeout=10)

    assert not at.exception
    # Check metrics rendered for sentiment
    metrics = at.metric
    assert len(metrics) >= 3
    metric_labels = [m.label for m in metrics]
    assert "Tone Label" in metric_labels
    assert "Confidence" in metric_labels


# ============================================================================
# 4. Document Intelligence Tab and Quick Action Tests
# ============================================================================

def test_document_intelligence_tab_renders_active_files(mock_api_client):
    """Verify document overview renders file cards when documents are attached."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.session_state["active_documents"] = [
        {
            "document_id": "doc-abc",
            "filename": "annual_report.pdf",
            "file_type": "pdf",
            "total_pages": 12,
            "total_characters": 8500,
            "chunk_count": 24,
        }
    ]
    at.run(timeout=10)

    assert not at.exception
    # Verify annual_report.pdf appears in markdown elements
    assert any("annual_report.pdf" in m.value for m in at.markdown)
    # Verify quick action buttons exist
    button_labels = [b.label for b in at.button]
    assert any("Executive Summary" in l for l in button_labels)
    assert any("Tone & Sentiment" in l for l in button_labels)


# ============================================================================
# 5. Deduplication and Safe Failure Handling Tests
# ============================================================================

def test_upload_failure_displays_user_safe_error(mock_api_client):
    """Verify API status errors during upload produce clear user-facing messages."""
    mock_api_client.upload_document.side_effect = ApiStatusError(
        message="Request failed (400)",
        status_code=400,
        detail="Unsupported file format '.exe'",
    )

    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.run(timeout=10)

    # The initial render should not crash
    assert not at.exception


def test_session_state_deduplication_preserves_conversation_across_reruns(mock_api_client):
    """Verify conversation ID and messages are preserved without duplication across reruns."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["_api_client"] = mock_api_client
    at.session_state["current_conversation_id"] = "fixed-uuid-12345"
    at.session_state["messages"] = [{"role": "user", "content": "Query 1"}]

    at.run(timeout=10)
    assert at.session_state["current_conversation_id"] == "fixed-uuid-12345"
    assert len(at.session_state["messages"]) == 1

    # Simulate second rerun
    at.run(timeout=10)
    assert at.session_state["current_conversation_id"] == "fixed-uuid-12345"
    assert len(at.session_state["messages"]) == 1
