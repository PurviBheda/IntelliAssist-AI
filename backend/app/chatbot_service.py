"""IntelliAssist AI - Chatbot Service Module.

This module orchestrates conversational document intelligence by integrating:
1. Intent Detection for classifying requests (document QA, summarization, sentiment, greetings, out-of-scope).
2. The persistent ConversationStore (SQLite) for multi-turn history.
3. The RAGPipeline for grounded semantic retrieval, citation mapping, and refusal on insufficient context.
4. TextSummarizer for concise and detailed text and document summarization.
5. SentimentAnalyzer for calibrated, non-fabricated sentiment classification.
6. Strict isolation between retrieved document context and conversational history.
7. Defense against hallucinations, citation fabrication, and API secret leakage.
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
from app.intent_detector import IntentDetector, IntentResult, IntentType
from app.rag_pipeline import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    build_citations,
    validate_question,
)
from app.sentiment_analyzer import SentimentAnalyzer, SentimentResult
from app.summarizer import TextSummarizer, SummaryResult, SummaryType
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
        assistant_response: Grounded answer synthesized by RAG or module response.
        citations: List of structured source citations preserved from RAG.
        retrieved_chunks: Document chunks retrieved from vector store.
        sufficient_context: True if context adequately answered the question.
        status: Operation status ('success', 'insufficient_context', 'error').
        model_name: Name of language model employed.
        is_live_llm: True only when a live cloud LLM (e.g. Gemini) was invoked.
        created_at: ISO 8601 UTC timestamp of the turn.
        error: Error message string if an error occurred, else None.
        unverified_citations: Detected ungrounded citations.
        intent: Detected intent category ('document_question', 'summarize', 'sentiment_analysis', etc.).
        intent_confidence: Confidence score of the detected intent.
        sentiment_result: Structured sentiment analysis result if sentiment intent.
        summary_result: Structured summarization result if summarize intent.
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
    intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    sentiment_result: Optional[Dict[str, Any]] = None
    summary_result: Optional[Dict[str, Any]] = None

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
    """Coordinates multi-turn conversations, intent detection, and RAG document intelligence."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        conversation_store: Optional[ConversationStore] = None,
        max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES,
        intent_detector: Optional[IntentDetector] = None,
        summarizer: Optional[TextSummarizer] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        """Initialize ChatbotService.

        Args:
            rag_pipeline: Initialized RAGPipeline instance.
            conversation_store: Optional ConversationStore instance. Defaults to
                                persistent store at backend/data/chat_history.db.
            max_history_messages: Maximum recent messages to supply as RAG context (default: 6).
            intent_detector: Optional IntentDetector instance. If None, initialized using
                             pipeline embedding service.
            summarizer: Optional TextSummarizer instance. If None, initialized using pipeline LLM.
            sentiment_analyzer: Optional SentimentAnalyzer instance.

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

        # Initialize intent detector
        self.intent_detector = (
            intent_detector
            if intent_detector is not None
            else IntentDetector(embedding_service=getattr(rag_pipeline, "embedding_service", None))
        )

        # Initialize summarizer
        self.summarizer = (
            summarizer
            if summarizer is not None
            else TextSummarizer(llm_client=getattr(rag_pipeline, "llm_client", None))
        )

        # Initialize sentiment analyzer
        self.sentiment_analyzer = (
            sentiment_analyzer
            if sentiment_analyzer is not None
            else SentimentAnalyzer()
        )

    def _determine_is_live_llm(self) -> bool:
        """Check whether the active LLM client is a real live cloud LLM.

        Never reports True if using a MockLLMClient or offline stub.
        """
        client = getattr(self.rag_pipeline, "llm_client", None)
        if client is None:
            return False
        return isinstance(client, GeminiLLMClient)

    def _handle_greeting(
        self,
        cid: str,
        cleaned_message: str,
        intent_res: IntentResult,
    ) -> ChatResponse:
        """Generate response for greetings, farewells, gratitude, and bot persona."""
        g_type = intent_res.parameters.get("greeting_type", "hello")

        if g_type == "gratitude":
            response_text = "You're very welcome! Let me know if you have any questions about your documents."
        elif g_type == "goodbye":
            response_text = "Goodbye! Have a great day, and feel free to return whenever you need document assistance."
        elif g_type == "chitchat":
            response_text = (
                "Hello! I am IntelliAssist AI, your document intelligence assistant. "
                "I can answer factual questions strictly grounded in your uploaded documents, "
                "generate concise or detailed summaries, and analyze text sentiment. "
                "How can I help you today?"
            )
        else:
            response_text = (
                "Hello! I am IntelliAssist AI, your document assistant. "
                "How can I help you with your documents today?"
            )

        assistant_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="assistant",
            content=response_text,
            citations=[],
            metadata={
                "intent": IntentType.GREETING.value,
                "intent_confidence": intent_res.confidence,
                "sufficient_context": True,
            },
        )

        return ChatResponse(
            conversation_id=cid,
            user_message=cleaned_message,
            assistant_response=response_text,
            citations=[],
            retrieved_chunks=[],
            sufficient_context=True,
            status="success",
            model_name=getattr(getattr(self.rag_pipeline, "llm_client", None), "model_name", None),
            is_live_llm=False,
            created_at=assistant_record["created_at"],
            intent=IntentType.GREETING.value,
            intent_confidence=intent_res.confidence,
        )

    def _handle_sentiment(
        self,
        cid: str,
        cleaned_message: str,
        intent_res: IntentResult,
        prior_history: List[Dict[str, Any]],
    ) -> ChatResponse:
        """Execute sentiment analysis on user-specified text or recent dialogue turn."""
        target_text = intent_res.parameters.get("text_to_analyze")

        # If not extracted inline, check if targeting conversation history
        if not target_text and prior_history:
            if intent_res.parameters.get("target") == "conversation":
                # Combine recent messages
                target_text = " ".join(m.get("content", "") for m in prior_history[-2:])
            else:
                # Take the most recent user or assistant message
                target_text = prior_history[-1].get("content", "")

        # If still no text to analyze, ask the user to provide the target text
        if not target_text:
            response_text = (
                "Please provide the text you would like me to analyze for sentiment. "
                "For example: 'Analyze the sentiment of: Our experience with the onboarding was smooth and helpful.'"
            )
            assistant_record = self.conversation_store.add_message(
                conversation_id=cid,
                role="assistant",
                content=response_text,
                citations=[],
                metadata={
                    "intent": IntentType.SENTIMENT_ANALYSIS.value,
                    "intent_confidence": intent_res.confidence,
                },
            )
            return ChatResponse(
                conversation_id=cid,
                user_message=cleaned_message,
                assistant_response=response_text,
                citations=[],
                retrieved_chunks=[],
                sufficient_context=True,
                status="success",
                model_name=getattr(getattr(self.rag_pipeline, "llm_client", None), "model_name", None),
                is_live_llm=False,
                created_at=assistant_record["created_at"],
                intent=IntentType.SENTIMENT_ANALYSIS.value,
                intent_confidence=intent_res.confidence,
            )

        sent_res = self.sentiment_analyzer.analyze(target_text, raise_errors=False)

        # Format structured response
        label_cap = sent_res.label.capitalize()
        conf_pct = sent_res.confidence * 100
        compound_sign = "+" if sent_res.compound_score > 0 else ""
        compound_str = f"{compound_sign}{sent_res.compound_score:.2f}"

        response_text = (
            f"Sentiment Analysis Result:\n"
            f"- Classification: {label_cap}\n"
            f"- Confidence Score: {sent_res.confidence:.2f} ({conf_pct:.1f}% certainty)\n"
            f"- Normalized Polarity: {compound_str}\n"
            f"- Valence Breakdown: Positive ({sent_res.positive_score * 100:.1f}%), "
            f"Neutral ({sent_res.neutral_score * 100:.1f}%), "
            f"Negative ({sent_res.negative_score * 100:.1f}%)\n"
            f"- Evaluated Text: \"{target_text}\""
        )

        assistant_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="assistant",
            content=response_text,
            citations=[],
            metadata={
                "intent": IntentType.SENTIMENT_ANALYSIS.value,
                "intent_confidence": intent_res.confidence,
                "sentiment_result": sent_res.to_dict(),
            },
        )

        return ChatResponse(
            conversation_id=cid,
            user_message=cleaned_message,
            assistant_response=response_text,
            citations=[],
            retrieved_chunks=[],
            sufficient_context=True,
            status="success",
            model_name=getattr(getattr(self.rag_pipeline, "llm_client", None), "model_name", None),
            is_live_llm=False,
            created_at=assistant_record["created_at"],
            intent=IntentType.SENTIMENT_ANALYSIS.value,
            intent_confidence=intent_res.confidence,
            sentiment_result=sent_res.to_dict(),
        )

    def _handle_summarize(
        self,
        cid: str,
        cleaned_message: str,
        intent_res: IntentResult,
        prior_history: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> ChatResponse:
        """Execute summarization for inline text, recent conversation, or retrieved documents."""
        summary_type = intent_res.parameters.get("summary_type", SummaryType.CONCISE.value)
        target = intent_res.parameters.get("target", "document")
        inline_text = intent_res.parameters.get("text_to_summarize")

        citations: List[Dict[str, Any]] = []
        retrieved_chunks: List[Dict[str, Any]] = []
        sufficient = True

        # Ensure summarizer has active llm_client if pipeline has one
        if getattr(self.summarizer, "llm_client", None) is None:
            pipeline_client = getattr(self.rag_pipeline, "llm_client", None)
            if pipeline_client is not None:
                self.summarizer.llm_client = pipeline_client

        # Case A: Summarize inline text
        if inline_text:
            sum_res = self.summarizer.summarize(
                inline_text,
                summary_type=summary_type,
                raise_errors=False,
            )
            response_text = sum_res.summary

        # Case B: Summarize conversation history
        elif target == "conversation":
            if not prior_history:
                response_text = "There is no prior conversation history in this thread to summarize yet."
                sum_res = SummaryResult(summary=response_text, summary_type=summary_type)
            else:
                history_lines = [
                    f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
                    for m in prior_history
                ]
                conv_text = "\n".join(history_lines)
                sum_res = self.summarizer.summarize(
                    conv_text,
                    summary_type=summary_type,
                    raise_errors=False,
                )
                response_text = f"Summary of our conversation ({summary_type}):\n{sum_res.summary}"

        # Case C: Summarize retrieved document context
        else:
            # Check vector store
            if self.rag_pipeline.vector_store.is_empty:
                response_text = INSUFFICIENT_CONTEXT_MESSAGE
                sufficient = False
                sum_res = SummaryResult(summary=response_text, summary_type=summary_type, status="insufficient_context")
            else:
                # Retrieve relevant chunks from vector store using the summarization query
                query_vec = self.rag_pipeline.embedding_service.embed_text(cleaned_message)
                effective_k = top_k if top_k is not None else max(4, self.rag_pipeline.top_k)
                raw_chunks = self.rag_pipeline.vector_store.search(query_vec, top_k=effective_k)

                thresh = similarity_threshold if similarity_threshold is not None else self.rag_pipeline.similarity_threshold
                if thresh is not None:
                    retrieved_chunks = [c for c in raw_chunks if c.get("similarity_score", -1.0) >= thresh]
                else:
                    retrieved_chunks = raw_chunks

                if not retrieved_chunks:
                    response_text = INSUFFICIENT_CONTEXT_MESSAGE
                    sufficient = False
                    sum_res = SummaryResult(summary=response_text, summary_type=summary_type, status="insufficient_context")
                else:
                    # Combine retrieved text and summarize
                    combined_doc_text = "\n\n".join(
                        c.get("text", "") for c in retrieved_chunks if c.get("text")
                    )
                    sum_res = self.summarizer.summarize(
                        combined_doc_text,
                        summary_type=summary_type,
                        raise_errors=False,
                    )
                    response_text = sum_res.summary

                    # Build structured source citations matching the retrieved chunks
                    citations = build_citations(retrieved_chunks, response_text)

        status_code = "success" if sufficient else "insufficient_context"
        is_live = self._determine_is_live_llm() if getattr(sum_res, "method", "") == "llm" else False

        assistant_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="assistant",
            content=response_text,
            citations=citations,
            metadata={
                "intent": IntentType.SUMMARIZE.value,
                "intent_confidence": intent_res.confidence,
                "summary_type": summary_type,
                "sufficient_context": sufficient,
                "is_live_llm": is_live,
            },
        )

        return ChatResponse(
            conversation_id=cid,
            user_message=cleaned_message,
            assistant_response=response_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            sufficient_context=sufficient,
            status=status_code,
            model_name=getattr(getattr(self.rag_pipeline, "llm_client", None), "model_name", None),
            is_live_llm=is_live,
            created_at=assistant_record["created_at"],
            intent=IntentType.SUMMARIZE.value,
            intent_confidence=intent_res.confidence,
            summary_result=sum_res.to_dict(),
        )

    def _handle_out_of_scope(
        self,
        cid: str,
        cleaned_message: str,
        intent_res: IntentResult,
    ) -> ChatResponse:
        """Handle inquiries outside document intelligence scope politely."""
        cat = intent_res.parameters.get("category", "general")
        if cat == "coding":
            response_text = (
                "I am IntelliAssist AI, an enterprise assistant specialized in document question-answering, "
                "summarization, and sentiment analysis. Software development and code generation are outside my scope. "
                "Please feel free to ask questions about your documents, policies, or guidelines."
            )
        elif cat == "security_exploit":
            response_text = (
                "I cannot assist with hacking, vulnerability exploitation, or bypassing security controls. "
                "I can, however, provide information regarding your company's documented security policies and compliance standards."
            )
        else:
            response_text = (
                "I am IntelliAssist AI, an enterprise assistant specialized in document question-answering, "
                "summarization, and sentiment analysis. This inquiry appears to be outside my scope. "
                "Please ask a question related to your uploaded documents, policies, or content."
            )

        assistant_record = self.conversation_store.add_message(
            conversation_id=cid,
            role="assistant",
            content=response_text,
            citations=[],
            metadata={
                "intent": IntentType.OUT_OF_SCOPE.value,
                "intent_confidence": intent_res.confidence,
                "sufficient_context": True,
            },
        )

        return ChatResponse(
            conversation_id=cid,
            user_message=cleaned_message,
            assistant_response=response_text,
            citations=[],
            retrieved_chunks=[],
            sufficient_context=True,
            status="success",
            model_name=getattr(getattr(self.rag_pipeline, "llm_client", None), "model_name", None),
            is_live_llm=False,
            created_at=assistant_record["created_at"],
            intent=IntentType.OUT_OF_SCOPE.value,
            intent_confidence=intent_res.confidence,
        )

    def chat(
        self,
        conversation_id: str,
        message: str,
        auto_create_conversation: bool = False,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> ChatResponse:
        """Process a conversational chat turn with intent detection, persistent history, and grounded RAG.

        Workflow:
        1. Validates user message (rejects empty/whitespace).
        2. Validates or creates conversation in ConversationStore.
        3. Loads recent history for THIS conversation only (strictly before saving current turn).
        4. Saves user message.
        5. Performs intent detection to identify the request category.
        6. Routes request:
           - 'greeting': Generates conversational reply.
           - 'sentiment_analysis': Invokes SentimentAnalyzer.
           - 'summarize': Invokes TextSummarizer (for inline text, chat, or retrieved documents).
           - 'out_of_scope': Informs user of document intelligence scope.
           - 'document_question' (or fallback): Executes grounded RAG pipeline.
        7. On provider/RAG failure, avoids persisting failed message and returns error result.
        8. Persists assistant turn in ConversationStore.

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

        # 5. Detect intent
        intent_res = self.intent_detector.detect(cleaned_message)

        # 6. Route according to detected intent
        if intent_res.intent == IntentType.GREETING.value:
            return self._handle_greeting(cid, cleaned_message, intent_res)

        elif intent_res.intent == IntentType.SENTIMENT_ANALYSIS.value:
            return self._handle_sentiment(cid, cleaned_message, intent_res, prior_history)

        elif intent_res.intent == IntentType.SUMMARIZE.value:
            return self._handle_summarize(
                cid, cleaned_message, intent_res, prior_history, top_k, similarity_threshold
            )

        elif intent_res.intent == IntentType.OUT_OF_SCOPE.value:
            return self._handle_out_of_scope(cid, cleaned_message, intent_res)

        # 7. Document Question & Safe Fallback -> Grounded RAG Pipeline
        try:
            rag_response: RAGResponse = self.rag_pipeline.answer(
                question=cleaned_message,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                conversation_history=prior_history,
            )
        except Exception as e:
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
                intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
            )

        is_live = self._determine_is_live_llm()

        # Check for pipeline error status
        if rag_response.generation_status == "error":
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
                intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
            )

        # Persist successful or insufficient_context assistant response
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
                "intent": intent_res.intent,
                "intent_confidence": intent_res.confidence,
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
            intent=intent_res.intent,
            intent_confidence=intent_res.confidence,
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
    intent_detector: Optional[IntentDetector] = None,
    summarizer: Optional[TextSummarizer] = None,
    sentiment_analyzer: Optional[SentimentAnalyzer] = None,
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
        intent_detector: Optional IntentDetector instance.
        summarizer: Optional TextSummarizer instance.
        sentiment_analyzer: Optional SentimentAnalyzer instance.

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
        intent_detector=intent_detector,
        summarizer=summarizer,
        sentiment_analyzer=sentiment_analyzer,
    )
