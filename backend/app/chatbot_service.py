"""IntelliAssist AI - Chatbot Service Module.

This module orchestrates conversational document question-answering by integrating:
1. The persistent ConversationStore (SQLite) for multi-turn history.
2. The existing RAGPipeline for grounded semantic retrieval, citation mapping,
   and refusal on insufficient context.
3. Strict isolation between retrieved document context and conversational history.
4. Defense against hallucinations, citation fabrication, and API secret leakage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.conversation_store import (
    ConversationStore,
    generate_conversation_title,
    get_default_db_path,
    utcnow_iso,
)
from app.embedding_service import EmbeddingService
from app.rag_pipeline import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    validate_question,
)
from app.vector_store import FAISSVectorStore


# ============================================================================
# Constants & Defaults
# ============================================================================

DEFAULT_MAX_HISTORY_MESSAGES = 6


# ============================================================================
# Response Model
# ============================================================================

@dataclass
class ChatResponse:
    """Structured response returned by ChatbotService.chat().

    Attributes:
        conversation_id: Identifier of the active conversation thread.
        user_message: Validated user input message.
        assistant_response: Grounded answer synthesized by RAG or insufficiency response.
        citations: List of structured source citations preserved from RAG.
        retrieved_chunks: Document chunks retrieved from vector store.
        sufficient_context: True if context adequately answered the question.
        status: Operation status ('success', 'insufficient_context', 'error').
        model_name: Name of language model employed.
        is_live_llm: True only when a live cloud LLM (e.g. Gemini) was invoked.
        created_at: ISO 8601 UTC timestamp of the turn.
        error: Error message string if an error occurred, else None.
    """

    conversation_id: str
    user_message: str
    assistant_response: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    sufficient_context: bool = True
    status: str = "success"
    model_name: Optional[str] = None
    is_live_llm: bool = False
    created_at: str = field(default_factory=utcnow_iso)
    error: Optional[str] = None
    unverified_citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility with dictionary consumers."""
        return asdict(self)[item]

    def get(self, item: str, default: Any = None) -> Any:
        """Allow get access for dictionary compatibility."""
        return asdict(self).get(item, default)


# ============================================================================
# Chatbot Service
# ============================================================================

class ChatbotService:
    """Coordinates multi-turn conversations and RAG document intelligence."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        conversation_store: Optional[ConversationStore] = None,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        """Initialize ChatbotService.

        Args:
            rag_pipeline: Initialized RAGPipeline instance.
            conversation_store: Optional ConversationStore instance. Defaults to
                                persistent store at backend/data/chat_history.db.
            max_history_messages: Maximum recent messages to supply as RAG context (default: 6).

        Raises:
            ValueError: If rag_pipeline is None or max_history_messages is not a positive integer.
        """
        if rag_pipeline is None:
            raise ValueError("rag_pipeline cannot be None.")
        if not isinstance(max_history_messages, int) or max_history_messages <= 0:
            raise ValueError(f"max_history_messages must be a positive integer, got: {max_history_messages!r}")

        self.rag_pipeline = rag_pipeline
        self.conversation_store = (
            conversation_store
            if conversation_store is not None
            else ConversationStore()
        )
        self.max_history_messages = max_history_messages

    def _determine_is_live_llm(self) -> bool:
        """Check whether the active LLM client is a real live cloud LLM.

        Never reports True if using a MockLLMClient or offline stub.
        """
        client = getattr(self.rag_pipeline, "llm_client", None)
        if client is None:
            return False
        return isinstance(client, GeminiLLMClient)

    def chat(
        self,
        conversation_id: str,
        message: str,
        auto_create_conversation: bool = False,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> ChatResponse:
        """Process a conversational chat turn with persistent history and RAG retrieval.

        Workflow:
        1. Validates user message (rejects empty/whitespace).
        2. Validates or creates conversation in ConversationStore.
        3. Loads recent history for THIS conversation only (strictly before saving current turn).
        4. Saves user message.
        5. Invokes RAG pipeline with query, separate recent history, and vector store.
        6. On success or insufficient context, saves assistant response and citations.
        7. On provider/RAG failure, avoids persisting failed message and returns error result.

        Args:
            conversation_id: Conversation identifier.
            message: User prompt string.
            auto_create_conversation: If True, creates conversation if not already existing.
            top_k: Optional per-request override for retrieved chunks.
            similarity_threshold: Optional per-request cosine similarity cutoff.

        Returns:
            Structured ChatResponse object.

        Raises:
            ValueError: If message is empty/whitespace or conversation_id is invalid.
            KeyError: If conversation does not exist and auto_create_conversation is False.
        """
        # 1. Validate user message
        cleaned_message = validate_question(message)

        # 2. Validate conversation ID
        if not conversation_id or not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ValueError("conversation_id must be a non-empty string.")

        cid = conversation_id.strip()

        # Check conversation existence
        conv = self.conversation_store.get_conversation(cid)
        if conv is None:
            if auto_create_conversation:
                conv = self.conversation_store.create_conversation(
                    conversation_id=cid,
                    title=generate_conversation_title(cleaned_message),
                )
            else:
                raise KeyError(f"Conversation '{cid}' does not exist.")

        # 3. Load recent history for this conversation BEFORE saving the current turn.
        # This guarantees:
        #   a) The current user message is not duplicated in history passed to the LLM.
        #   b) Prior turns are limited to max_history_messages.
        #   c) History is strictly isolated to this conversation_id.
        prior_history = self.conversation_store.get_recent_messages(
            conversation_id=cid,
            limit=self.max_history_messages,
        )

        # 4. Save user message to persistent conversation history
        user_msg_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="user",
            content=cleaned_message,
        )

        # 5. Call RAG pipeline
        try:
            rag_response: RAGResponse = self.rag_pipeline.answer(
                question=cleaned_message,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                conversation_history=prior_history,
            )
        except Exception as e:
            # Handle unexpected pipeline crash safely without exposing internal secrets
            sanitized_err = str(e)
            return ChatResponse(
                conversation_id=cid,
                user_message=cleaned_message,
                assistant_response="An unexpected error occurred during answer generation.",
                citations=[],
                retrieved_chunks=[],
                sufficient_context=False,
                status="error",
                model_name=getattr(self.rag_pipeline.llm_client, "model_name", None),
                is_live_llm=False,
                created_at=utcnow_iso(),
                error=sanitized_err,
            )

        is_live = self._determine_is_live_llm()

        # 6. Check for pipeline error status (e.g. LLM provider failure or missing API key)
        if rag_response.generation_status == "error":
            # Do NOT persist a failed assistant turn as a valid answer in history
            return ChatResponse(
                conversation_id=cid,
                user_message=cleaned_message,
                assistant_response=rag_response.answer,
                citations=rag_response.citations,
                retrieved_chunks=rag_response.retrieved_chunks,
                sufficient_context=False,
                status="error",
                model_name=rag_response.model_name,
                is_live_llm=False,
                created_at=utcnow_iso(),
                error=rag_response.error or "Pipeline returned error status.",
            )

        # 7. Persist successful or insufficient_context assistant response
        # Insufficient context is a valid, grounded refusal and is preserved in history
        assistant_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="assistant",
            content=rag_response.answer,
            citations=rag_response.citations,
            metadata={
                "model_name": rag_response.model_name,
                "sufficient_context": rag_response.sufficient_context,
                "generation_status": rag_response.generation_status,
                "is_live_llm": is_live,
            },
        )

        return ChatResponse(
            conversation_id=cid,
            user_message=cleaned_message,
            assistant_response=rag_response.answer,
            citations=rag_response.citations,
            retrieved_chunks=rag_response.retrieved_chunks,
            sufficient_context=rag_response.sufficient_context,
            status=rag_response.generation_status,
            model_name=rag_response.model_name,
            is_live_llm=is_live,
            created_at=assistant_record["created_at"],
            error=None,
            unverified_citations=getattr(rag_response, "unverified_citations", []),
        )

    # ========================================================================
    # Convenience Passthrough Operations
    # ========================================================================

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation thread in the underlying store."""
        return self.conversation_store.create_conversation(
            title=title,
            conversation_id=conversation_id,
        )

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations ordered by most recently updated."""
        return self.conversation_store.list_conversations(limit=limit, offset=offset)

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation metadata and message count."""
        return self.conversation_store.get_conversation(conversation_id)

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve full message history for a conversation in chronological order."""
        return self.conversation_store.get_messages(conversation_id, limit=limit)

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its associated messages."""
        return self.conversation_store.delete_conversation(conversation_id)


# ============================================================================
# Convenience Factory
# ============================================================================

def create_chatbot_service(
    vector_store: FAISSVectorStore,
    embedding_service: Optional[EmbeddingService] = None,
    llm_client: Optional[BaseLLMClient] = None,
    conversation_store: Optional[ConversationStore] = None,
    max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: Optional[float] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> ChatbotService:
    """Convenience factory function to build a ChatbotService from vector store.

    Args:
        vector_store: Initialized FAISSVectorStore with indexed document chunks.
        embedding_service: Optional EmbeddingService instance.
        llm_client: Optional BaseLLMClient instance.
        conversation_store: Optional ConversationStore instance.
        max_history_messages: Maximum recent messages for context.
        top_k: Default retrieval chunk count.
        similarity_threshold: Optional minimum cosine similarity cutoff.
        max_context_chars: Context character limit in prompt.

    Returns:
        Configured ChatbotService instance.
    """
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        max_context_chars=max_context_chars,
    )
    return ChatbotService(
        rag_pipeline=pipeline,
        conversation_store=conversation_store,
        max_history_messages=max_history_messages,
    )
