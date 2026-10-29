"""Unit and integration tests for LangChain adapters and LCEL chains in IntelliAssist AI."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock
import numpy as np
import pytest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

from app.embedding_service import EmbeddingService, LangChainEmbeddingsAdapter
from app.rag_pipeline import (
    DEFAULT_SYSTEM_INSTRUCTION,
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    LangChainLLMAdapter,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    build_langchain_rag_chain,
)
from app.vector_store import FAISSRetrieverAdapter, FAISSVectorStore


# ============================================================================
# Deterministic Test Fixtures
# ============================================================================

def make_test_vector(text: str, dim: int = 384) -> np.ndarray:
    """Generate a deterministic unit-length vector seeded by text hash."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Create a mock EmbeddingService producing deterministic vectors."""
    service = MagicMock(spec=EmbeddingService)
    service.embedding_dimension = 384
    service.embed_text.side_effect = lambda text: make_test_vector(text)
    service.embed_texts.side_effect = lambda texts, batch_size=32: np.vstack(
        [make_test_vector(t) for t in texts]
    )
    return service


@pytest.fixture
def populated_vector_store() -> FAISSVectorStore:
    """Create a FAISS vector store pre-loaded with policy chunks."""
    store = FAISSVectorStore(dimension=384, normalize_vectors=True)
    chunks = [
        {
            "chunk_id": "sec_policy.pdf_page_1_chunk_1",
            "text": "Passwords must be at least 14 characters long and combine letters, numbers, and symbols.",
            "metadata": {
                "filename": "sec_policy.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
            "embedding": make_test_vector("passwords must be 14 characters long"),
        },
        {
            "chunk_id": "sec_policy.pdf_page_1_chunk_2",
            "text": "Multi-factor authentication (MFA) is mandatory for accessing internal company systems.",
            "metadata": {
                "filename": "sec_policy.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 2,
            },
            "embedding": make_test_vector("multi-factor authentication is mandatory"),
        },
        {
            "chunk_id": "handbook.txt_page_1_chunk_1",
            "text": "Employees accrue 20 days of paid time off per year.",
            "metadata": {
                "filename": "handbook.txt",
                "file_type": "txt",
                "page_number": 1,
                "chunk_index": 1,
            },
            "embedding": make_test_vector("employees accrue 20 days pto"),
        },
    ]
    store.add_chunks(chunks)
    return store


# ============================================================================
# Tests: LangChain Embeddings Adapter
# ============================================================================

class TestLangChainEmbeddingsAdapter:
    """Verify that LangChainEmbeddingsAdapter conforms to LangChain Embeddings interface."""

    def test_implements_langchain_embeddings_interface(
        self, mock_embedding_service: MagicMock
    ) -> None:
        """Adapter inherits from LangChain Embeddings."""
        adapter = LangChainEmbeddingsAdapter(mock_embedding_service)
        assert isinstance(adapter, Embeddings)

    def test_as_langchain_embeddings_factory_method(
        self, mock_embedding_service: MagicMock
    ) -> None:
        """EmbeddingService.as_langchain_embeddings() returns valid adapter."""
        # Attach method to mock
        mock_embedding_service.as_langchain_embeddings = lambda: LangChainEmbeddingsAdapter(
            mock_embedding_service
        )
        adapter = mock_embedding_service.as_langchain_embeddings()
        assert isinstance(adapter, LangChainEmbeddingsAdapter)
        assert isinstance(adapter, Embeddings)

    def test_embed_documents(self, mock_embedding_service: MagicMock) -> None:
        """embed_documents returns List[List[float]] matching input count."""
        adapter = LangChainEmbeddingsAdapter(mock_embedding_service)
        texts = ["First passage", "Second passage", "Third passage"]
        result = adapter.embed_documents(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        for item in result:
            assert isinstance(item, list)
            assert len(item) == 384
            assert all(isinstance(v, float) for v in item)

    def test_embed_documents_empty_list(
        self, mock_embedding_service: MagicMock
    ) -> None:
        """embed_documents with empty list returns empty list."""
        adapter = LangChainEmbeddingsAdapter(mock_embedding_service)
        result = adapter.embed_documents([])
        assert result == []

    def test_embed_query(self, mock_embedding_service: MagicMock) -> None:
        """embed_query returns 1D List[float] of dimension 384."""
        adapter = LangChainEmbeddingsAdapter(mock_embedding_service)
        result = adapter.embed_query("How long must passwords be?")

        assert isinstance(result, list)
        assert len(result) == 384
        assert all(isinstance(v, float) for v in result)


# ============================================================================
# Tests: LangChain Retriever Adapter
# ============================================================================

class TestLangChainRetrieverAdapter:
    """Verify that FAISSRetrieverAdapter conforms to LangChain BaseRetriever interface."""

    def test_implements_langchain_retriever_interface(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Adapter inherits from LangChain BaseRetriever."""
        retriever = populated_vector_store.as_retriever(
            embedding_service=mock_embedding_service,
            top_k=2,
        )
        assert isinstance(retriever, BaseRetriever)

    def test_retriever_invoke_returns_langchain_documents(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: MagicMock,
    ) -> None:
        """retriever.invoke() returns Document objects with content and preserved metadata."""
        retriever = populated_vector_store.as_retriever(
            embedding_service=mock_embedding_service,
            top_k=2,
        )
        docs = retriever.invoke("passwords must be 14 characters long")

        assert isinstance(docs, list)
        assert len(docs) == 2
        for doc in docs:
            assert isinstance(doc, Document)
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0
            # Metadata preservation
            assert "filename" in doc.metadata
            assert "page_number" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "similarity_score" in doc.metadata
            assert isinstance(doc.metadata["similarity_score"], float)

    def test_retriever_empty_store_returns_empty_list(
        self, mock_embedding_service: MagicMock
    ) -> None:
        """Retriever on empty store returns empty list without error."""
        empty_store = FAISSVectorStore(dimension=384)
        retriever = empty_store.as_retriever(
            embedding_service=mock_embedding_service,
            top_k=3,
        )
        docs = retriever.invoke("anything")
        assert docs == []

    def test_retriever_similarity_threshold_filtering(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: MagicMock,
    ) -> None:
        """Retriever filters out chunks below similarity threshold."""
        # Set impossible threshold of 0.999
        retriever = populated_vector_store.as_retriever(
            embedding_service=mock_embedding_service,
            top_k=5,
            similarity_threshold=0.999,
        )
        docs = retriever.invoke("random unrelated query")
        # Should filter down heavily or to empty
        for doc in docs:
            assert doc.metadata["similarity_score"] >= 0.999


# ============================================================================
# Tests: LangChain LLM Adapter & LCEL Chain
# ============================================================================

class TestLangChainLLMAdapter:
    """Verify that LangChainLLMAdapter conforms to LangChain LLM interface."""

    def test_implements_langchain_llm_interface(self) -> None:
        """Adapter inherits from LangChain LLM."""
        mock_llm = MockLLMClient()
        adapter = LangChainLLMAdapter(client=mock_llm)
        assert isinstance(adapter, LLM)

    def test_llm_invoke_delegates_to_client(self) -> None:
        """adapter.invoke() calls client.generate_response with instruction and prompt."""
        mock_llm = MockLLMClient(default_response="Custom mock answer.")
        adapter = LangChainLLMAdapter(
            client=mock_llm,
            system_instruction="Strict instructions.",
        )
        output = adapter.invoke("User test prompt")
        assert output == "Custom mock answer."
        assert len(mock_llm.calls) == 1
        assert mock_llm.calls[0]["system_instruction"] == "Strict instructions."
        assert "User test prompt" in mock_llm.calls[0]["prompt"]

    def test_build_langchain_rag_chain_execution(self) -> None:
        """build_langchain_rag_chain creates a working LCEL chain."""
        mock_llm = MockLLMClient(default_response="LCEL answer.")
        chain = build_langchain_rag_chain(
            client=mock_llm,
            system_instruction=DEFAULT_SYSTEM_INSTRUCTION,
        )
        assert isinstance(chain, Runnable)

        result = chain.invoke({"prompt": "Formatted test prompt"})
        assert result == "LCEL answer."


# ============================================================================
# Tests: End-to-End RAGPipeline LangChain Execution Path
# ============================================================================

class TestRAGPipelineLangChainIntegration:
    """Verify that RAGPipeline executes generation through the LangChain LCEL chain."""

    def test_rag_pipeline_answer_uses_langchain_chain(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: MagicMock,
    ) -> None:
        """RAGPipeline.answer() generates answers via LangChain and preserves RAGResponse contract."""
        mock_llm = MockLLMClient(
            default_response="Passwords must be at least 14 characters. (Page No. 1)"
        )
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
            top_k=2,
        )

        response = pipeline.answer("What is the password requirement?")

        # Output must be the existing RAGResponse structure
        assert isinstance(response, RAGResponse)
        assert response.generation_status == "success"
        assert response.sufficient_context is True
        assert "14 characters" in response.answer
        assert len(response.citations) > 0
        assert response.retrieval_count == 2
        assert mock_llm.call_count == 1

    def test_rag_pipeline_as_langchain_chain(
        self,
        populated_vector_store: FAISSVectorStore,
        mock_embedding_service: MagicMock,
    ) -> None:
        """RAGPipeline.as_langchain_chain() produces an end-to-end Runnable."""
        mock_llm = MockLLMClient(
            default_response="Full LCEL RAG response grounded in documents."
        )
        pipeline = RAGPipeline(
            vector_store=populated_vector_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
            top_k=2,
        )

        chain = pipeline.as_langchain_chain()
        assert isinstance(chain, Runnable)

        answer = chain.invoke("What are the password rules?")
        assert isinstance(answer, str)
        assert "Full LCEL RAG response" in answer

    def test_rag_pipeline_langchain_refusal_on_empty_store(
        self, mock_embedding_service: MagicMock
    ) -> None:
        """Empty vector store returns standard safe refusal."""
        empty_store = FAISSVectorStore(dimension=384)
        mock_llm = MockLLMClient()
        pipeline = RAGPipeline(
            vector_store=empty_store,
            embedding_service=mock_embedding_service,
            llm_client=mock_llm,
        )

        response = pipeline.answer("Any question")
        assert response.sufficient_context is False
        assert response.generation_status == "insufficient_context"
        assert response.answer == INSUFFICIENT_CONTEXT_MESSAGE
        assert mock_llm.call_count == 0  # LLM never invoked on empty store
