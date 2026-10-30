"""IntelliAssist AI - Clean Minimal ChatGPT-Style Streamlit Application.

Plain white main workspace with ~800px centered conversation column,
anchored unified composer with attached [+] upload popover and [↑] send button,
and a simple light-gray sidebar (~250px).
Connected to backend services via DirectServiceAdapter (default) or IntelliAssistApiClient (HTTP mode).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Union
import uuid

import streamlit as st

# Ensure backend root is at index 0 of sys.path so 'app' package resolves to backend/app
backend_dir = str(Path(__file__).resolve().parent.parent)
while backend_dir in sys.path:
    sys.path.remove(backend_dir)
sys.path.insert(0, backend_dir)

from streamlit_app.api_client import IntelliAssistApiClient
from streamlit_app.components.chat_view import render_chat_view
from streamlit_app.components.sidebar import render_sidebar


# ============================================================================
# Page Configuration & Clean Minimal CSS
# ============================================================================

st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Global reset to plain white */
    .stApp {
        background-color: #FFFFFF !important;
        color: #262626 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Main central column constrained to ~800px and centered */
    .main .block-container {
        max-width: 800px !important;
        margin: 0 auto !important;
        padding-top: 1.5rem !important;
        padding-bottom: 7.5rem !important; /* Space for anchored composer */
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        background-color: #FFFFFF !important;
    }

    /* Minimal light-gray sidebar (~250px) */
    [data-testid="stSidebar"] {
        background-color: #F9F9F9 !important;
        border-right: 1px solid #E5E5E5 !important;
        min-width: 250px !important;
        max-width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* Typography */
    h1, h2, h3, h4 {
        color: #262626 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.015em !important;
    }
    p, span, label, div {
        color: #262626;
    }

    /* Chat Messages - ChatGPT style */
    .stChatMessage {
        border-radius: 0px !important;
        padding: 8px 0px !important;
        margin-bottom: 8px !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    /* User message bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background-color: #F4F4F4 !important;
        border-radius: 18px !important;
        padding: 10px 16px !important;
        margin-left: auto !important;
        max-width: 80% !important;
    }
    /* Assistant plain response */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background-color: transparent !important;
        padding: 6px 0px 12px 0px !important;
        width: 100% !important;
    }

    /* Subtle avatars */
    div[data-testid="stChatMessageAvatar"] {
        background-color: #EFEFEF !important;
        color: #525252 !important;
        border-radius: 50% !important;
    }

    /* Anchored Bottom Container - Pinned composer styling */
    div[data-testid="stBottomBlockContainer"] {
        background-color: #FFFFFF !important;
        padding-bottom: 1.2rem !important;
        padding-top: 0.5rem !important;
        border-top: 1px solid #F0F0F0 !important;
    }
    div[data-testid="stBottomBlockContainer"] > div {
        max-width: 800px !important;
        margin: 0 auto !important;
    }

    /* The [+] attachment button in st.bottom */
    div[data-testid="stBottomBlockContainer"] div[data-testid="stPopover"] > button {
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.15rem !important;
        border: 1px solid #E5E5E5 !important;
        background-color: #FFFFFF !important;
        color: #525252 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
    }
    div[data-testid="stBottomBlockContainer"] div[data-testid="stPopover"] > button:hover {
        background-color: #F4F4F4 !important;
        border-color: #D4D4D4 !important;
        color: #171717 !important;
    }

    /* Native Chat Input Box */
    .stChatInputContainer {
        border-radius: 24px !important;
        border: 1px solid #E5E5E5 !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 6px rgba(0, 0, 0, 0.03) !important;
        padding: 2px 6px !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #8E8E8E !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06) !important;
    }
    .stChatInputContainer button[data-testid="stChatInputSubmitButton"] {
        border-radius: 50% !important;
        background-color: #171717 !important;
        color: #FFFFFF !important;
        width: 32px !important;
        height: 32px !important;
    }
    .stChatInputContainer button[data-testid="stChatInputSubmitButton"]:hover {
        background-color: #000000 !important;
    }

    /* Citations expander */
    .streamlit-expanderHeader {
        background-color: #FAFAFA !important;
        border: 1px solid #E5E5E5 !important;
        border-radius: 8px !important;
        color: #525252 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        padding: 6px 12px !important;
    }
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E5E5 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 10px 12px !important;
    }

    /* Sidebar New chat button */
    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #171717 !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        border: 1px solid #171717 !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #333333 !important;
    }

    /* Sidebar conversation item buttons */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #262626 !important;
        border: none !important;
        text-align: left !important;
        padding: 6px 8px !important;
        font-size: 0.86rem !important;
        border-radius: 6px !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #EFEFEF !important;
    }

    /* Small attachment chip */
    .doc-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #F4F4F4;
        border: 1px solid #E5E5E5;
        border-radius: 14px;
        padding: 2px 8px;
        font-size: 0.76rem;
        color: #404040;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Client Provider Singleton (Direct In-Process by Default / HTTP Optional)
# ============================================================================

def get_api_client() -> Any:
    """Initialize and retrieve the client instance for this user session.

    Defaults to in-process DirectServiceAdapter for single-process cloud or zero-cost hosting.
    HTTP mode (IntelliAssistApiClient) can be explicitly selected by setting
    INTELLIASSIST_CLIENT_MODE=http (or INTELLIASSIST_MODE=http) or INTELLIASSIST_API_URL.
    Honors pre-injected session_state['_api_client'] for isolated testing.
    """
    client = st.session_state.get("_api_client")
    if client is None:
        client_mode = os.getenv("INTELLIASSIST_CLIENT_MODE", os.getenv("INTELLIASSIST_MODE", "")).strip().lower()
        if client_mode in ("http", "api", "server", "fastapi") or (
            os.getenv("INTELLIASSIST_API_URL") and client_mode != "direct"
        ):
            client = IntelliAssistApiClient()
        else:
            from streamlit_app.direct_client import DirectServiceAdapter
            client = DirectServiceAdapter()
        st.session_state["_api_client"] = client
    return client


def init_session_state() -> None:
    """Initialize session state variables."""
    if "current_conversation_id" not in st.session_state:
        st.session_state["current_conversation_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "active_documents" not in st.session_state:
        st.session_state["active_documents"] = []
    if "processed_upload_signatures" not in st.session_state:
        st.session_state["processed_upload_signatures"] = set()
    if "pending_query" not in st.session_state:
        st.session_state["pending_query"] = None


# ============================================================================
# Main Application Render
# ============================================================================

def main() -> None:
    """Main application loop."""
    init_session_state()
    client = get_api_client()

    # Render Minimal Sidebar (Logo, + New chat, Search, Recent History, Status)
    render_sidebar(client)

    # Render Centered Flowing Conversation Workspace & Bottom Unified Composer
    render_chat_view(client)


if __name__ == "__main__":
    main()
