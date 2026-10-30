"""IntelliAssist AI - Direct Service Adapter for Streamlit.

Enables the Streamlit frontend to run in a single-process deployment (e.g. on
Streamlit Community Cloud) by calling Python backend services directly in-memory,
completely bypassing HTTP networking while preserving the exact same public
interface and user-safe exception contracts as IntelliAssistApiClient.

Architecture & Isolation:
1. Shared Resources: Embedding models (SentenceTransformers) and Gemini LLM clients
   are loaded once and cached process-wide via @st.cache_resource.
2. Session Isolation: Each user session maintains its own private in-memory
   FAISSVectorStore and SQLite ConversationStore (':memory:'), ensuring documents
   and chat histories are strictly isolated across concurrent user sessions.
3. Clean Lifecycle: Uploaded files are processed via tempfile and immediately
   cleaned up after text extraction and chunk indexing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import streamlit as st

# Ensure backend root is on sys.path for direct module resolution
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.chatbot_service import ChatResponse, ChatbotService, create_chatbot_service
from app.conversation_store import ConversationStore, utcnow_iso
from app.document_processor import SUPPORTED_EXTENSIONS, process_document
from app.embedding_service import DEFAULT_EMBEDDING_DIMENSION, DEFAULT_MODEL_NAME, EmbeddingService
from app.intent_detector import IntentDetector
from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    load_environment_variables,
)
from app.sentiment_analyzer import SentimentAnalyzer
from app.summarizer import TextSummarizer
from app.text_chunker import chunk_document
from app.vector_store import FAISSVectorStore
from streamlit_app.api_client import (
    ApiClientError,
    ApiConnectionError,
    ApiPayloadError,
    ApiStatusError,
)

logger = logging.getLogger("intelliassist.direct_client")

# Ensure environment variables (.env) are loaded
load_environment_variables()


def load_secrets_into_environ() -> None:
    """Synchronize Streamlit Community Cloud secrets into os.environ if present."""
    try:
        if hasattr(st, "secrets"):
            for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL"]:
                if key not in os.environ:
                    try:
                        val = st.secrets.get(key)
                        if val:
                            os.environ[key] = str(val).strip()
                    except Exception:
                        pass
    except Exception:
        pass


load_secrets_into_environ()

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


# ============================================================================
# Shared Process-Wide Singletons (@st.cache_resource)
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_shared_embedding_service(
    model_name: str = DEFAULT_MODEL_NAME,
) -> EmbeddingService:
    """Retrieve or initialize the shared sentence-transformers embedding service.

    Cached once per Python process to conserve memory on free cloud tiers.
    """
    logger.info("Initializing shared EmbeddingService with model '%s'", model_name)
    return EmbeddingService(model_name=model_name, normalize_embeddings=True)


@st.cache_resource(show_spinner=False)
def get_shared_llm_client() -> BaseLLMClient:
    """Retrieve or initialize the shared LLM client singleton.

    Uses GeminiLLMClient when a valid GEMINI_API_KEY is configured
    (via os.environ, .env, or Streamlit Cloud Secrets), and safely falls
    back to MockLLMClient for offline or test execution.
    """
    load_secrets_into_environ()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    placeholder_keys = {
        "your_gemini_api_key_here",
        "your_actual_key_here",
        "your_api_key_here",
        "TODO",
        "CHANGEME",
    }
    if gemini_key and gemini_key.strip() and gemini_key.strip() not in placeholder_keys:
        try:
            model = os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            client = GeminiLLMClient(api_key=gemini_key.strip(), model_name=model)
            logger.info("Initialized shared GeminiLLMClient with model '%s'", model)
            return client
        except Exception as e:
            logger.warning(
                "Failed to initialize live GeminiLLMClient: %s. Using MockLLMClient fallback.", e
            )

    logger.info("Using MockLLMClient fallback (no live Gemini key configured).")
    return MockLLMClient()


# ============================================================================
# Streamlit Context Detection
# ============================================================================

def is_in_streamlit_session() -> bool:
    """Check whether execution is inside an active Streamlit user session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

            return get_script_run_ctx() is not None
        except Exception:
            return False


# ============================================================================
# Direct Service Adapter
# ============================================================================

class DirectServiceAdapter:
    """Direct Python in-process adapter mirroring IntelliAssistApiClient.

    Allows the Streamlit UI to invoke backend services without network hops.
    Each user session operates with private in-memory FAISS and SQLite stores.
    """

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        conversation_store: Optional[ConversationStore] = None,
        chatbot_service: Optional[ChatbotService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_client: Optional[BaseLLMClient] = None,
        db_path: Optional[Union[str, Path]] = ":memory:",
        intent_detector: Optional[IntentDetector] = None,
        summarizer: Optional[TextSummarizer] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        """Initialize DirectServiceAdapter.

        Args:
            vector_store: Optional explicit vector store instance.
            conversation_store: Optional explicit conversation store instance.
            chatbot_service: Optional explicit chatbot service instance.
            embedding_service: Optional explicit embedding service instance.
            llm_client: Optional explicit LLM client instance.
            db_path: SQLite database path (defaults to ':memory:' for transient sessions).
            intent_detector: Optional explicit intent detector.
            summarizer: Optional explicit text summarizer.
            sentiment_analyzer: Optional explicit sentiment analyzer.
        """
        self.base_url = "direct://in-process"
        self._db_path = db_path or ":memory:"

        # 1. Resolve shared process-wide services
        self.embedding_service = embedding_service or get_shared_embedding_service()
        self.llm_client = llm_client or get_shared_llm_client()

        # 2. Resolve session-isolated services
        in_session = is_in_streamlit_session()

        # Vector store
        if vector_store is not None:
            self._vector_store = vector_store
        elif in_session:
            if "_direct_vector_store" not in st.session_state:
                st.session_state["_direct_vector_store"] = FAISSVectorStore(
                    dimension=DEFAULT_EMBEDDING_DIMENSION,
                    normalize_vectors=True,
                )
            self._vector_store = st.session_state["_direct_vector_store"]
        else:
            self._vector_store = FAISSVectorStore(
                dimension=DEFAULT_EMBEDDING_DIMENSION,
                normalize_vectors=True,
            )

        # Conversation store
        if conversation_store is not None:
            self._conversation_store = conversation_store
        elif in_session:
            if "_direct_conversation_store" not in st.session_state:
                st.session_state["_direct_conversation_store"] = ConversationStore(
                    db_path=self._db_path
                )
            self._conversation_store = st.session_state["_direct_conversation_store"]
        else:
            self._conversation_store = ConversationStore(db_path=self._db_path)

        # Chatbot service
        if chatbot_service is not None:
            self._chatbot_service = chatbot_service
        elif in_session:
            if "_direct_chatbot_service" not in st.session_state:
                st.session_state["_direct_chatbot_service"] = create_chatbot_service(
                    vector_store=self._vector_store,
                    embedding_service=self.embedding_service,
                    llm_client=self.llm_client,
                    conversation_store=self._conversation_store,
                    top_k=DEFAULT_TOP_K,
                    max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
                    intent_detector=intent_detector,
                    summarizer=summarizer,
                    sentiment_analyzer=sentiment_analyzer,
                )
            self._chatbot_service = st.session_state["_direct_chatbot_service"]
        else:
            self._chatbot_service = create_chatbot_service(
                vector_store=self._vector_store,
                embedding_service=self.embedding_service,
                llm_client=self.llm_client,
                conversation_store=self._conversation_store,
                top_k=DEFAULT_TOP_K,
                max_context_chars=DEFAULT_MAX_CONTEXT_CHARS,
                intent_detector=intent_detector,
                summarizer=summarizer,
                sentiment_analyzer=sentiment_analyzer,
            )

    @property
    def vector_store(self) -> FAISSVectorStore:
        """Access the session's active FAISS vector store."""
        return self._vector_store

    @property
    def conversation_store(self) -> ConversationStore:
        """Access the session's active conversation store."""
        return self._conversation_store

    @property
    def chatbot_service(self) -> ChatbotService:
        """Access the session's active chatbot service."""
        return self._chatbot_service

    # ========================================================================
    # Public Methods Mirroring IntelliAssistApiClient
    # ========================================================================

    def check_health(self, timeout: Optional[float] = 5.0) -> Dict[str, Any]:
        """Check system health, vector store statistics, and LLM configuration.

        Returns:
            Health status dictionary with 'status', 'service', 'is_live_llm',
            'model_name', 'indexed_chunks', and 'timestamp'.
        """
        is_live = self.chatbot_service._determine_is_live_llm()
        model_name = getattr(self.llm_client, "model_name", "unknown")
        return {
            "status": "healthy",
            "service": "IntelliAssist AI",
            "is_live_llm": is_live,
            "model_name": model_name,
            "indexed_chunks": self.vector_store.total_vectors,
            "timestamp": utcnow_iso(),
        }

    def is_available(self, timeout: float = 3.0) -> bool:
        """Convenience method checking whether services are operational and healthy."""
        try:
            health = self.check_health(timeout=timeout)
            return health.get("status") == "healthy"
        except Exception:
            return False

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        starred: bool = False,
        shared: bool = False,
        timeout: Optional[float] = 10.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation threads from the session's conversation store."""
        return self.conversation_store.list_conversations(
            limit=limit,
            offset=offset,
            only_starred=starred,
            only_shared=shared,
        )

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Create a new conversation thread in the session's conversation store."""
        try:
            conv = self.conversation_store.create_conversation(
                title=title,
                conversation_id=conversation_id,
            )
            conv["documents"] = []
            return conv
        except ValueError as e:
            raise ApiStatusError(
                message=f"Request failed (400): {e}",
                status_code=400,
                detail=str(e),
            ) from e

    def get_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Retrieve full details of a conversation thread including messages and documents."""
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        conv = self.conversation_store.get_conversation_with_messages(conversation_id)
        if conv is None:
            raise ApiStatusError(
                message=f"Request failed (404): Conversation '{conversation_id}' not found.",
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return conv

    def delete_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Delete a conversation thread and its associated records."""
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        if not self.conversation_store.conversation_exists(conversation_id):
            raise ApiStatusError(
                message=f"Request failed (404): Conversation '{conversation_id}' not found.",
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        deleted = self.conversation_store.delete_conversation(conversation_id)
        return {"status": "success", "conversation_id": conversation_id, "deleted": deleted}

    def star_conversation(
        self,
        conversation_id: str,
        is_starred: Optional[bool] = None,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Toggle or set starred status for a conversation thread."""
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        if not self.conversation_store.conversation_exists(conversation_id):
            raise ApiStatusError(
                message=f"Request failed (404): Conversation '{conversation_id}' not found.",
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )

        if is_starred is not None:
            self.conversation_store.set_conversation_starred(conversation_id, is_starred)
            new_status = is_starred
        else:
            new_status = self.conversation_store.toggle_conversation_starred(conversation_id)

        return {"conversation_id": conversation_id, "is_starred": new_status}

    def upload_document(
        self,
        filename: str,
        file_bytes: Union[bytes, bytearray],
        conversation_id: Optional[str] = None,
        content_type: Optional[str] = None,
        timeout: Optional[float] = 60.0,
    ) -> Dict[str, Any]:
        """Process, chunk, embed, and index a document into the session's vector store.

        Uses temporary files for document extraction and securely cleans them up
        immediately after processing.
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string.")
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty.")

        suffix = Path(filename).suffix.lower()

        # 1. Validate file extension
        if suffix not in SUPPORTED_EXTENSIONS:
            supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ApiStatusError(
                message=f"Request failed (400): Unsupported file format '{suffix}'. Supported formats are: {supported_str}",
                status_code=400,
                detail=f"Unsupported file format '{suffix}'. Supported formats are: {supported_str}",
            )

        # 2. Validate file size
        byte_data = bytes(file_bytes)
        if len(byte_data) == 0:
            raise ApiStatusError(
                message="Request failed (400): Uploaded file is empty (0 bytes).",
                status_code=400,
                detail="Uploaded file is empty (0 bytes).",
            )

        if len(byte_data) > MAX_FILE_SIZE_BYTES:
            max_mb = MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise ApiStatusError(
                message=f"Request failed (400): File exceeds maximum allowed size of {max_mb}MB.",
                status_code=400,
                detail=f"File exceeds maximum allowed size of {max_mb}MB.",
            )

        # 3. Write to temporary file, extract text, and clean up
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(byte_data)
            tmp_path = Path(tmp_file.name)

        try:
            try:
                doc_data = process_document(tmp_path)
                doc_data["filename"] = filename
            except ValueError as e:
                raise ApiStatusError(
                    message=f"Request failed (400): {e}",
                    status_code=400,
                    detail=str(e),
                ) from e
            except Exception as e:
                raise ApiStatusError(
                    message=f"Request failed (500): Unexpected error processing document: {e}",
                    status_code=500,
                    detail=f"Unexpected error processing document: {e}",
                ) from e

            # 4. Chunk and embed document into the session vector store
            try:
                chunks = chunk_document(doc_data, chunk_size=600, chunk_overlap=100)
                embedded_chunks = self.embedding_service.embed_chunks(chunks, batch_size=32)
                self.vector_store.add_chunks(embedded_chunks)
            except Exception as e:
                raise ApiStatusError(
                    message=f"Request failed (500): Failed to chunk and embed document: {e}",
                    status_code=500,
                    detail=f"Failed to chunk and embed document: {e}",
                ) from e
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        # 5. Link document to conversation in the session store
        cid = (
            conversation_id.strip()
            if conversation_id and isinstance(conversation_id, str) and conversation_id.strip()
            else None
        )
        if cid:
            if not self.conversation_store.conversation_exists(cid):
                doc_title = f"{Path(filename).stem} Analysis"
                self.conversation_store.create_conversation(title=doc_title, conversation_id=cid)
            else:
                current_conv = self.conversation_store.get_conversation(cid)
                if current_conv and current_conv.get("title") == "New Conversation":
                    self.conversation_store.update_conversation_title(
                        cid, f"{Path(filename).stem} Analysis"
                    )

            doc_record = self.conversation_store.add_document(
                conversation_id=cid,
                filename=filename,
                file_type=doc_data["file_type"],
                total_pages=doc_data["total_pages"],
                total_characters=doc_data["total_characters"],
                chunk_count=len(chunks),
                file_path=f"memory://{filename}",
            )
        else:
            doc_record = {
                "document_id": str(uuid.uuid4()),
                "conversation_id": None,
                "filename": filename,
                "file_type": doc_data["file_type"],
                "total_pages": doc_data["total_pages"],
                "total_characters": doc_data["total_characters"],
                "chunk_count": len(chunks),
                "file_path": f"memory://{filename}",
                "created_at": utcnow_iso(),
            }

        return {
            "status": "success",
            "message": f"Successfully processed '{filename}' ({doc_data['total_pages']} page(s), {len(chunks)} chunks).",
            "document": doc_record,
        }

    def send_chat_message(
        self,
        conversation_id: str,
        message: str,
        auto_create_conversation: bool = True,
        timeout: Optional[float] = 90.0,
    ) -> Dict[str, Any]:
        """Submit a prompt or query to the session's RAG chatbot pipeline.

        Returns:
            Complete ChatResponse dictionary containing assistant_response,
            citations, intent, sentiment_result, summary_result, and sufficiency flags.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")
        if not message or not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        try:
            response: ChatResponse = self.chatbot_service.chat(
                conversation_id=conversation_id.strip(),
                message=message.strip(),
                auto_create_conversation=auto_create_conversation,
            )
            return response.to_dict()
        except ValueError as e:
            raise ApiStatusError(
                message=f"Request failed (400): {e}",
                status_code=400,
                detail=str(e),
            ) from e
        except KeyError as e:
            raise ApiStatusError(
                message=f"Request failed (404): {e}",
                status_code=404,
                detail=str(e),
            ) from e
        except Exception as e:
            logger.exception("Unexpected error during direct chat: %s", e)
            raise ApiStatusError(
                message=f"Request failed (500): An error occurred while answering your question: {e}",
                status_code=500,
                detail=f"An error occurred while answering your question: {e}",
            ) from e

    def remove_document(
        self,
        conversation_id: str,
        document_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Remove a document association from a conversation thread and tidy vectors."""
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")
        if not document_id or not isinstance(document_id, str):
            raise ValueError("document_id must be a non-empty string.")

        # Determine target document filename before deletion
        conv = self.conversation_store.get_conversation_with_messages(conversation_id)
        doc_filename: Optional[str] = None
        if conv:
            for doc in conv.get("documents", []):
                if doc.get("document_id") == document_id:
                    doc_filename = doc.get("filename")
                    break

        success = self.conversation_store.delete_document(conversation_id, document_id)
        if not success:
            raise ApiStatusError(
                message="Request failed (404): Document not found for this conversation.",
                status_code=404,
                detail="Document not found for this conversation.",
            )

        # Synchronize vector store if no other conversation in this session uses the document
        if doc_filename:
            all_docs = self.conversation_store.get_all_documents()
            still_referenced = any(d.get("filename") == doc_filename for d in all_docs)
            if not still_referenced:
                self.vector_store.delete_chunks_by_document(doc_filename)

        return {"status": "success", "document_id": document_id, "conversation_id": conversation_id}

    # ========================================================================
    # Convenience Sharing Passthroughs
    # ========================================================================

    def share_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Enable public sharing for a conversation thread."""
        res = self.conversation_store.share_conversation(conversation_id)
        if res is None:
            raise ApiStatusError(
                message=f"Request failed (404): Conversation '{conversation_id}' not found.",
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        return res

    def unshare_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Revoke public sharing for a conversation thread."""
        if not self.conversation_store.conversation_exists(conversation_id):
            raise ApiStatusError(
                message=f"Request failed (404): Conversation '{conversation_id}' not found.",
                status_code=404,
                detail=f"Conversation '{conversation_id}' not found.",
            )
        self.conversation_store.unshare_conversation(conversation_id)
        return {"status": "success", "conversation_id": conversation_id, "is_shared": False}

    def get_shared_conversation(
        self,
        share_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Retrieve a shared conversation thread by share_id."""
        conv = self.conversation_store.get_conversation_by_share_id(share_id)
        if conv is None:
            raise ApiStatusError(
                message="Request failed (404): Shared conversation not found or access has been revoked.",
                status_code=404,
                detail="Shared conversation not found or access has been revoked.",
            )
        return conv


# ============================================================================
# Session-Scoped Client Provider
# ============================================================================

def get_direct_client() -> DirectServiceAdapter:
    """Retrieve or create the session-scoped DirectServiceAdapter instance."""
    if is_in_streamlit_session():
        client = st.session_state.get("_api_client")
        if client is None or not isinstance(client, DirectServiceAdapter):
            client = DirectServiceAdapter()
            st.session_state["_api_client"] = client
        return client
    return DirectServiceAdapter()

