"""Streamlit Chat View Component for IntelliAssist AI (Option B - Native Streamlit).

Implements:
- Plain white, minimalist ChatGPT-style conversation layout.
- Modest welcome heading: "How can I help you today?" centered horizontally and vertically when empty.
- Chronological, vertically flowing ~800px conversation column.
- Streamlit-native custom composer: [ + ] [ Ask anything about your documents... ] [ ↑ ]
  using st.columns(), st.popover(), st.text_area(), and st.button().
- Small compact attachment chips directly above the composer.
- Grounded assistant responses with compact expandable Source Citations.
- Robust session-state lifecycle (no duplicates, draft clearing, blank rejection).
- Graceful error handling for model generation and connection failures.
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


def _submit_chat_message() -> None:
    """Callback triggered on send button click. Clears draft and schedules query."""
    draft = st.session_state.get("chat_composer_draft", "")
    cleaned = draft.strip()
    if cleaned:
        st.session_state["pending_query"] = cleaned
        st.session_state["chat_composer_draft"] = ""


def render_chat_view(client: IntelliAssistApiClient) -> None:
    """Render the main chat interface, past messages, welcome heading, and custom composer."""
    # Scoped styling for clean, minimalist ChatGPT visual design
    st.markdown(
        """
        <style>
        /* Empty screen modest welcome container */
        .chat-empty-container {
            min-height: 44vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2.5rem 1rem 1rem 1rem;
        }
        .chat-empty-heading {
            font-size: 1.75rem;
            font-weight: 600;
            color: #171717;
            margin: 0;
            letter-spacing: -0.02em;
        }

        /* Custom Composer Row in st.bottom */
        div[data-testid="stBottomBlockContainer"] {
            background-color: #FFFFFF !important;
        }
        div[data-testid="stBottomBlockContainer"] [data-testid="column"] {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        /* Popover [+] button */
        div[data-testid="stBottomBlockContainer"] div[data-testid="stPopover"] {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stPopover"] > button {
            border-radius: 50% !important;
            width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.2rem !important;
            font-weight: 500 !important;
            border: 1px solid #E5E5E5 !important;
            background-color: #FFFFFF !important;
            color: #404040 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stPopover"] > button:hover {
            background-color: #F4F4F4 !important;
            border-color: #D4D4D4 !important;
            color: #171717 !important;
        }

        /* Textarea input in middle column */
        div[data-testid="stBottomBlockContainer"] div[data-testid="stTextArea"] {
            width: 100% !important;
            margin-bottom: 0 !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stTextArea"] > div {
            border: 1px solid #E5E5E5 !important;
            border-radius: 16px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stTextArea"] > div:focus-within {
            border-color: #737373 !important;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.08) !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stTextArea"] textarea {
            border: none !important;
            border-radius: 16px !important;
            background-color: transparent !important;
            color: #171717 !important;
            font-size: 0.95rem !important;
            line-height: 1.45 !important;
            padding: 10px 14px !important;
            resize: none !important;
            min-height: 42px !important;
            max-height: 140px !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stTextArea"] textarea::placeholder {
            color: #8E8E8E !important;
            opacity: 1 !important;
        }

        /* Send [↑] button */
        div[data-testid="stBottomBlockContainer"] div[data-testid="stButton"] {
            width: 100% !important;
            display: flex !important;
            justify-content: center !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stButton"] > button {
            border-radius: 50% !important;
            width: 42px !important;
            height: 42px !important;
            min-height: 42px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.15rem !important;
            font-weight: 600 !important;
            border: 1px solid #171717 !important;
            background-color: #171717 !important;
            color: #FFFFFF !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stButton"] > button:hover {
            background-color: #333333 !important;
            border-color: #333333 !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stBottomBlockContainer"] div[data-testid="stButton"] > button:active {
            background-color: #000000 !important;
            border-color: #000000 !important;
        }

        /* Compact document chip */
        .doc-chip {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: #F4F4F4;
            border: 1px solid #E5E5E5;
            border-radius: 12px;
            padding: 3px 9px;
            font-size: 0.78rem;
            color: #404040;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    messages: List[Dict[str, Any]] = st.session_state.setdefault("messages", [])
    cid = st.session_state.get("current_conversation_id")
    active_docs: List[Dict[str, Any]] = st.session_state.get("active_documents", [])
    st.session_state.setdefault("chat_composer_draft", "")

    # Retrieve and consume any pending query submission
    pending_query = st.session_state.pop("pending_query", None)

    # 1. Empty Screen State (Modest heading, centered horizontally and vertically)
    if not messages and not pending_query:
        st.markdown(
            """
            <div class="chat-empty-container">
                <h2 class="chat-empty-heading">
                    How can I help you today?
                </h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # If a new query was submitted, register it in the conversation thread
    if pending_query:
        if not cid:
            try:
                new_conv = client.create_conversation(title="Document Discussion")
                cid = new_conv["conversation_id"]
                st.session_state["current_conversation_id"] = cid
            except Exception:
                cid = str(uuid.uuid4())
                st.session_state["current_conversation_id"] = cid

        user_msg = {"role": "user", "content": pending_query}
        messages.append(user_msg)

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

                # Sentiment & Summary metrics if present
                sent = metadata.get("sentiment_result") or msg.get("sentiment_result")
                if sent:
                    _render_sentiment_card(sent)

                summary = metadata.get("summary_result") or msg.get("summary_result")
                if summary:
                    _render_summary_card(summary)
            else:
                st.markdown(content)

    # 3. Handle Live Assistant Query Processing
    if pending_query:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response_data = client.send_chat_message(
                        conversation_id=cid,
                        message=pending_query,
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
                        logger.warning("Language model failure for query '%s': %s", pending_query, err_msg)
                    else:
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

    # 4. Anchored Bottom Custom Composer Bar: [ + ] [ Ask anything about your documents... ] [ ↑ ]
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
                f"<div style='display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin-bottom:8px;'>"
                f"{chips_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Custom horizontal composer: [ + ] [ text_area ] [ ↑ ]
        col_plus, col_input, col_send = st.columns([1, 14, 1], vertical_alignment="center")

        with col_plus:
            with st.popover("＋", help="Attach document (PDF, DOCX, TXT)", use_container_width=True):
                st.markdown(
                    "<div style='font-size:0.85rem; font-weight:600; margin-bottom:8px; color:#262626;'>"
                    "Attach Document"
                    "</div>",
                    unsafe_allow_html=True,
                )
                uploader_key = f"composer_uploader_{cid or 'default'}"
                uploaded_file = st.file_uploader(
                    "Upload file",
                    type=["pdf", "docx", "txt"],
                    label_visibility="collapsed",
                    key=uploader_key,
                )
                if uploaded_file is not None:
                    sig = f"{uploaded_file.name}_{uploaded_file.size}_{cid or ''}"
                    processed_sigs = st.session_state.setdefault("processed_upload_signatures", set())

                    if sig not in processed_sigs:
                        active_cid = cid
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
            st.text_area(
                "Ask anything about your documents...",
                placeholder="Ask anything about your documents...",
                key="chat_composer_draft",
                label_visibility="collapsed",
                height=42,
                max_chars=4000,
            )

        with col_send:
            st.button(
                "↑",
                key="chat_send_btn",
                help="Send message",
                on_click=_submit_chat_message,
                type="primary",
                use_container_width=True,
            )
