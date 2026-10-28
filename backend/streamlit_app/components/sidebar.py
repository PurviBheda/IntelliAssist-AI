"""Streamlit Sidebar Component for IntelliAssist AI.

Simple, compact light-gray sidebar:
- Top: Small icon + IntelliAssist AI text
- + New chat button
- Search conversations field
- Recent conversation list (clean text buttons, delete action)
- Compact Document Insights expander
- Bottom: Small backend connection indicator
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import uuid

import streamlit as st

from streamlit_app.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiStatusError,
    IntelliAssistApiClient,
)
from streamlit_app.components.doc_summary import render_doc_summary

logger = logging.getLogger("intelliassist.streamlit_sidebar")


def _get_backend_status(client: IntelliAssistApiClient) -> Dict[str, Any]:
    """Retrieve backend health with a short in-memory cache to avoid redundant network calls."""
    now = time.time()
    cached = st.session_state.get("_cached_backend_health")
    cached_time = st.session_state.get("_cached_backend_health_time", 0)

    if cached is not None and (now - cached_time) < 10.0:
        return cached

    try:
        health_data = client.check_health(timeout=3.0)
        st.session_state["_cached_backend_health"] = health_data
        st.session_state["_cached_backend_health_time"] = now
        st.session_state["backend_connected"] = True
        return health_data
    except Exception as err:
        failure_data = {"status": "offline", "error": str(err)}
        st.session_state["_cached_backend_health"] = failure_data
        st.session_state["_cached_backend_health_time"] = now
        st.session_state["backend_connected"] = False
        return failure_data


def render_sidebar(client: IntelliAssistApiClient) -> None:
    """Render the simple, compact light-gray sidebar."""
    with st.sidebar:
        # 1. Branding: Small icon + Text
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; font-weight:600; font-size:0.95rem; color:#262626; padding: 2px 0 10px 0;">
                <span style="font-size:1.05rem;">✦</span>
                <span>IntelliAssist AI</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. + New chat button
        if st.button("＋ New chat", use_container_width=True, type="primary"):
            try:
                created = client.create_conversation(title="New Conversation")
                st.session_state["current_conversation_id"] = created["conversation_id"]
                st.session_state["messages"] = []
                st.session_state["active_documents"] = []
                st.session_state["pending_query"] = None
                st.rerun()
            except Exception:
                st.session_state["current_conversation_id"] = str(uuid.uuid4())
                st.session_state["messages"] = []
                st.session_state["active_documents"] = []
                st.session_state["pending_query"] = None
                st.rerun()

        # 3. Search conversations input
        search_query = st.text_input(
            "Search conversations",
            placeholder="Search conversations...",
            key="conv_search_query",
            label_visibility="collapsed",
        )

        # 4. Document Insights & Attached Files (Simple compact expander)
        with st.expander("📑 Document Insights", expanded=False):
            render_doc_summary(client)

        st.markdown(
            "<div style='font-size:0.75rem; font-weight:600; color:#737373; margin:14px 0 6px 2px;'>Recent</div>",
            unsafe_allow_html=True,
        )

        # 5. Recent Conversations List
        try:
            conversations = client.list_conversations(limit=40)
        except Exception:
            conversations = []

        if search_query and search_query.strip():
            sq = search_query.strip().lower()
            conversations = [c for c in conversations if sq in (c.get("title") or "").lower()]

        if not conversations:
            st.caption("No previous conversations found.")
        else:
            current_cid = st.session_state.get("current_conversation_id")

            for conv in conversations:
                cid = conv.get("conversation_id", "")
                raw_title = conv.get("title", "Conversation")
                msg_count = conv.get("message_count", 0)
                is_active = (cid == current_cid)

                display_title = raw_title if len(raw_title) <= 24 else f"{raw_title[:22]}..."
                button_label = f"{display_title} ({msg_count})"

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    if st.button(
                        button_label,
                        key=f"conv_select_{cid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state["current_conversation_id"] = cid
                        try:
                            details = client.get_conversation(cid)
                            st.session_state["messages"] = details.get("messages", [])
                            st.session_state["active_documents"] = details.get("documents", [])
                        except Exception as load_err:
                            st.error(f"Failed to load: {load_err}")
                        st.rerun()

                with col_del:
                    if st.button("✕", key=f"conv_del_{cid}", help="Delete"):
                        try:
                            client.delete_conversation(cid)
                            if cid == current_cid:
                                st.session_state["current_conversation_id"] = None
                                st.session_state["messages"] = []
                                st.session_state["active_documents"] = []
                            st.rerun()
                        except Exception as del_err:
                            st.error(f"Could not delete: {del_err}")

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # 6. Bottom Backend Status Indicator
        health = _get_backend_status(client)

        if health.get("status") == "healthy":
            model_name = health.get("model_name", "gemini")
            st.markdown(
                f"<div style='font-size:0.75rem; color:#737373; padding:8px 0; border-top:1px solid #E5E5E5;'>"
                f"<span style='color:#16A34A;'>●</span> Connected ({model_name})"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error(
                "**Backend Offline**  \n"
                "Cannot connect to `http://127.0.0.1:8000`",
                icon="🔴",
            )
            with st.expander("🛠️ Startup command"):
                st.code(
                    "cd \"d:\\IntelliAssist AI\\backend\"\n"
                    ".\\.venv\\Scripts\\python.exe server.py",
                    language="powershell",
                )
            if st.button("🔄 Retry", use_container_width=True):
                st.session_state.pop("_cached_backend_health", None)
                st.rerun()
