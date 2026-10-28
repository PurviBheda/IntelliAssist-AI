"""Streamlit Document Insights Component for IntelliAssist AI.

Provides a clean, minimal document metadata overview and quick actions
for summarization and tone analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import uuid

import streamlit as st

from streamlit_app.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiStatusError,
    IntelliAssistApiClient,
)

logger = logging.getLogger("intelliassist.streamlit_doc_summary")


def render_doc_summary(client: IntelliAssistApiClient) -> None:
    """Render document overview, attached files, and one-click summary/sentiment triggers."""
    cid = st.session_state.get("current_conversation_id")
    active_docs: List[Dict[str, Any]] = st.session_state.get("active_documents", [])

    if not active_docs:
        st.caption("No documents attached to this conversation. Use the attachment button to upload files.")
        return

    st.markdown("<div style='font-size:0.8rem; font-weight:600; color:#525252; margin-bottom:6px;'>Attached Files</div>", unsafe_allow_html=True)

    for doc in active_docs:
        fname = doc.get("filename", "Document")
        ftype = doc.get("file_type", "").upper()
        pages = doc.get("total_pages", 1)
        chunks = doc.get("chunk_count", 0)
        doc_id = doc.get("document_id", "")

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**📄 {fname}**  \n<span style='color:#737373; font-size:0.75rem;'>{ftype} • {pages} pg • {chunks} chunks</span>", unsafe_allow_html=True)
        with col2:
            if st.button("✕", key=f"del_doc_{doc_id}", help="Detach document"):
                if cid and doc_id:
                    try:
                        client.remove_document(conversation_id=cid, document_id=doc_id)
                        st.session_state["active_documents"] = [
                            d for d in active_docs if d.get("document_id") != doc_id
                        ]
                        st.rerun()
                    except Exception as err:
                        st.error(f"Could not remove document: {err}")

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # Quick Intelligence Actions
    col_sum, col_sent = st.columns(2)

    with col_sum:
        if st.button("📝 Executive Summary", use_container_width=True, type="secondary"):
            if not cid:
                cid = str(uuid.uuid4())
                st.session_state["current_conversation_id"] = cid

            prompt_text = "Please provide a concise summary of the key findings and contents of the uploaded document(s)."
            st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt_text})

            with st.spinner("Synthesizing summary..."):
                try:
                    resp = client.send_chat_message(
                        conversation_id=cid,
                        message=prompt_text,
                        auto_create_conversation=True,
                    )
                    asst_msg = {
                        "role": "assistant",
                        "content": resp.get("assistant_response", ""),
                        "citations": resp.get("citations", []),
                        "intent": resp.get("intent"),
                        "summary_result": resp.get("summary_result"),
                    }
                    st.session_state["messages"].append(asst_msg)
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to generate summary: {err}")

    with col_sent:
        if st.button("📊 Tone & Sentiment", use_container_width=True, type="secondary"):
            if not cid:
                cid = str(uuid.uuid4())
                st.session_state["current_conversation_id"] = cid

            prompt_text = "Analyze the overall sentiment, tone, and emotional valence of the uploaded document content."
            st.session_state.setdefault("messages", []).append({"role": "user", "content": prompt_text})

            with st.spinner("Analyzing tone..."):
                try:
                    resp = client.send_chat_message(
                        conversation_id=cid,
                        message=prompt_text,
                        auto_create_conversation=True,
                    )
                    asst_msg = {
                        "role": "assistant",
                        "content": resp.get("assistant_response", ""),
                        "citations": resp.get("citations", []),
                        "intent": resp.get("intent"),
                        "sentiment_result": resp.get("sentiment_result"),
                    }
                    st.session_state["messages"].append(asst_msg)
                    st.rerun()
                except Exception as err:
                    st.error(f"Failed to evaluate sentiment: {err}")
