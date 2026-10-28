"""Streamlit UI Components for IntelliAssist AI."""

try:
    from streamlit_app.components.sidebar import render_sidebar
    from streamlit_app.components.chat_view import render_chat_view
    from streamlit_app.components.doc_summary import render_doc_summary
except ImportError:
    from backend.streamlit_app.components.sidebar import render_sidebar  # type: ignore
    from backend.streamlit_app.components.chat_view import render_chat_view  # type: ignore
    from backend.streamlit_app.components.doc_summary import render_doc_summary  # type: ignore

__all__ = ["render_sidebar", "render_chat_view", "render_doc_summary"]
