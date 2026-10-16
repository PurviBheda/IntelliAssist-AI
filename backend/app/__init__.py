"""IntelliAssist AI - Backend Application Package."""

from .document_processor import process_document
from .text_chunker import chunk_document, split_text_into_chunks
from .embedding_service import (
    EmbeddingService,
    embed_chunks,
    embed_text,
    embed_texts,
)
from .vector_store import (
    DEFAULT_EMBEDDING_DIMENSION,
    FAISSVectorStore,
    create_vector_store,
)
from .retrieval_evaluator import (
    EvaluationQuery,
    EvaluationSummary,
    QueryEvaluationResult,
    RetrievalEvaluator,
    compute_hit_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    load_queries_from_json,
)

__all__ = [
    "process_document",
    "chunk_document",
    "split_text_into_chunks",
    "EmbeddingService",
    "embed_chunks",
    "embed_text",
    "embed_texts",
    "DEFAULT_EMBEDDING_DIMENSION",
    "FAISSVectorStore",
    "create_vector_store",
    "RetrievalEvaluator",
    "EvaluationQuery",
    "EvaluationSummary",
    "QueryEvaluationResult",
    "compute_hit_at_k",
    "compute_recall_at_k",
    "compute_reciprocal_rank",
    "load_queries_from_json",
]
