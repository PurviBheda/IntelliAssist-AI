"""Streamlit Chat View Component for IntelliAssist AI.

Implements:
- Modest welcome heading centered horizontally and vertically when empty
- Chronological, vertically flowing ~800px conversation column
- Unified bottom-anchored composer: [ + ] [ Ask anything about your documents... ] [ ↑ ]
- Small attachment chips
- Plain assistant responses with compact expandable Source Citations
- User-friendly error messaging on model failures
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

logger = logging.getLogger("intelliassist.streamlit_chat")


def _render_citations(citations: List[Dict[str, Any]]) -> None:
    """Render compact, collapsible source citation metadata."""
    if not citations:
        return

    with st.expander(f"📚 Source Citations ({len(citations)})", expanded=False):
        for idx, cite in enumerate(citations, 1):
            fname = cite.get("filename", "Unknown Document")
            page = cite.get("page_number", 1)
            score = cite.get("similarity_score")
            score_text = f" — {score:.1%} match" if score is not None else ""

            st.markdown(
                f"<div style='font-size:0.83rem; color:#525252; padding:3px 0;'>"
                f"[{idx}] <b>{fname}</b> (Page {page}){score_text}</div>",
                unsafe_allow_html=True,
            )


def _render_sentiment_card(sent: Dict[str, Any]) -> None:
    """Render structured sentiment evaluation metrics if returned."""
    if not sent or not isinstance(sent, dict):
        return

    label = sent.get("label", "").capitalize()
    confidence = sent.get("confidence", 0.0)
    compound = sent.get("compound_score", 0.0)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tone Label", label)
    col2.metric("Confidence", f"{confidence:.1%}")
    col3.metric("Compound Polarity", f"{compound:+.2f}")


def _render_summary_card(summary: Dict[str, Any]) -> None:
    """Render structured text summarization metrics if returned."""
    if not summary or not isinstance(summary, dict):
        return

    comp_ratio = summary.get("compression_ratio", 0.0)
    method = summary.get("method", "extractive").capitalize()
    orig_len = summary.get("original_sentences") or summary.get("original_characters", 0)
    sum_len = summary.get("summary_sentences") or summary.get("summary_characters", 0)

    st.info(
        f"**Summary Metrics ({method})**: "
        f"Compressed to **{comp_ratio:.1%}** of original text "
        f"({sum_len} sentences from {orig_len} original).",
        icon="📝",
    )


def render_chat_view(client: IntelliAssistApiClient) -> None:
    """Render the main chat interface, past messages, welcome heading, and input composer."""
    messages: List[Dict[str, Any]] = st.session_state.setdefault("messages", [])
    cid = st.session_state.get("current_conversation_id")
    active_docs: List[Dict[str, Any]] = st.session_state.get("active_documents", [])

    # 1. Homepage State (Only when no messages exist) - Vertically and horizontally centered
    if not messages:
        st.markdown(
            """
            <div style="min-height: 46vh; display: flex; align-items: center; justify-content: center; text-align: center;">
                <h2 style="font-size: 1.75rem; font-weight: 600; color: #262626; margin: 0; letter-spacing: -0.02em;">
                    How can I help you today?
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Conversation Turns (Chronological, vertically flowing column)
    for msg in messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        metadata = msg.get("metadata", {}) or {}

        with st.chat_message(role):
            if role == "assistant":
                intent_val = metadata.get("intent") or msg.get("intent")
                if intent_val and intent_val != "document_question":
                    st.caption(f"`{intent_val.replace('_', ' ').capitalize()}`")

                # Plain assistant response
                st.markdown(content)

                # Source Citations
                citations = msg.get("citations") or metadata.get("citations") or []
                _render_citations(citations)

                # Sentiment & Summary metrics if requested
                sent = metadata.get("sentiment_result") or msg.get("sentiment_result")
                if sent:
                    _render_sentiment_card(sent)

                summary = metadata.get("summary_result") or msg.get("summary_result")
                if summary:
                    _render_summary_card(summary)
            else:
                st.markdown(content)

    # 3. Anchored Bottom Composer Bar: [ + ] [ Ask anything about your documents... ] [ ↑ ]
    with st.bottom:
        # Small attachment chips directly above the input box
        if active_docs:
            chips_html = "".join(
                [
                    f"<span class='doc-chip'>📄 {d.get('filename', 'doc')} ({d.get('chunk_count', 0)} chunks)</span>"
                    for d in active_docs
                ]
            )
            st.markdown(
                f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:6px;'>"
                f"{chips_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Single visually connected row: [+] on the left, input in middle, send arrow on right
        col_plus, col_input = st.columns([1, 16], vertical_alignment="center")

        with col_plus:
            with st.popover("＋", help="Attach document (PDF, DOCX, TXT)"):
                st.markdown("<div style='font-size:0.85rem; font-weight:600; margin-bottom:6px;'>Attach Document</div>", unsafe_allow_html=True)
                uploaded_file = st.file_uploader(
                    "Upload file",
                    type=["pdf", "docx", "txt"],
                    label_visibility="collapsed",
                    key="composer_unified_uploader",
                )
                if uploaded_file is not None:
                    sig = f"{uploaded_file.name}_{uploaded_file.size}_{st.session_state.get('current_conversation_id', '')}"
                    processed_sigs = st.session_state.setdefault("processed_upload_signatures", set())

                    if sig not in processed_sigs:
                        active_cid = st.session_state.get("current_conversation_id")
                        if not active_cid:
                            try:
                                new_thread = client.create_conversation(title=f"{uploaded_file.name} Query")
                                active_cid = new_thread["conversation_id"]
                                st.session_state["current_conversation_id"] = active_cid
                            except Exception:
                                active_cid = str(uuid.uuid4())
                                st.session_state["current_conversation_id"] = active_cid

                        with st.spinner("Indexing document..."):
                            try:
                                upload_resp = client.upload_document(
                                    filename=uploaded_file.name,
                                    file_bytes=uploaded_file.getvalue(),
                                    conversation_id=active_cid,
                                    content_type=uploaded_file.type,
                                    timeout=90.0,
                                )
                                processed_sigs.add(sig)

                                try:
                                    conv_data = client.get_conversation(active_cid)
                                    st.session_state["active_documents"] = conv_data.get("documents", [])
                                except Exception:
                                    doc_rec = upload_resp.get("document")
                                    if doc_rec:
                                        st.session_state.setdefault("active_documents", []).append(doc_rec)

                                st.session_state.pop("_cached_backend_health", None)
                                st.rerun()
                            except Exception as err:
                                st.error(f"Upload failed: {err}")

        with col_input:
            prompt = st.chat_input("Ask anything about your documents...")

    if prompt and prompt.strip():
        user_text = prompt.strip()

        # Ensure active conversation ID is set
        if not cid:
            try:
                new_conv = client.create_conversation(title="Document Discussion")
                cid = new_conv["conversation_id"]
                st.session_state["current_conversation_id"] = cid
            except Exception:
                cid = str(uuid.uuid4())
                st.session_state["current_conversation_id"] = cid

        # Append and display user message immediately
        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(user_text)

        # Generate grounded assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_data = client.send_chat_message(
                        conversation_id=cid,
                        message=user_text,
                        auto_create_conversation=True,
                    )

                    answer = response_data.get("assistant_response", "")
                    citations = response_data.get("citations", [])
                    intent_val = response_data.get("intent")
                    sent_res = response_data.get("sentiment_result")
                    sum_res = response_data.get("summary_result")
                    gen_status = response_data.get("status")
                    err_msg = response_data.get("error")

                    # Handle model generation failure gracefully with user-friendly alert
                    if gen_status == "error" or "Unable to generate answer due to language model failure" in answer:
                        st.warning("⚠️ The AI service is currently busy or rate-limited. Please try asking again in a moment.")
                        logger.warning("Language model failure for query '%s': %s", user_text, err_msg)
                        return

                    # Display intent caption if special intent
                    if intent_val and intent_val != "document_question":
                        st.caption(f"`{intent_val.replace('_', ' ').capitalize()}`")

                    # Display clean answer text
                    st.markdown(answer)

                    # Display citations
                    _render_citations(citations)

                    # Display sentiment/summary if present
                    if sent_res:
                        _render_sentiment_card(sent_res)
                    if sum_res:
                        _render_summary_card(sum_res)

                    # Save assistant message to session state
                    asst_msg = {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                        "intent": intent_val,
                        "sentiment_result": sent_res,
                        "summary_result": sum_res,
                    }
                    messages.append(asst_msg)

                except ApiStatusError as err:
                    st.error(f"Error ({err.status_code}): {err.detail or err}")
                except ApiConnectionError as err:
                    st.error(f"Connection failed: {err}")
                except Exception as err:
                    st.error(f"Unexpected error: {err}")
