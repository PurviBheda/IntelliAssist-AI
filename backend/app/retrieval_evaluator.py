"""IntelliAssist AI - Retrieval Evaluation Module.

This module provides a rigorous, transparent evaluation harness for measuring
retrieval quality of the document embedding service and FAISS vector store.
It calculates standard IR metrics:
  - Hit Rate@K (proportion of queries with at least one relevant chunk in top K)
  - Recall@K (average proportion of known relevant chunks retrieved in top K)
  - Mean Reciprocal Rank (MRR, average 1/rank of first retrieved relevant chunk)

It integrates directly with the existing document processor, chunker,
embedding service, and FAISS vector store without duplicate retrieval logic.
"""

from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from app.document_processor import SUPPORTED_EXTENSIONS, process_document
from app.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
)
from app.embedding_service import (
    DEFAULT_EMBEDDING_DIMENSION,
    EmbeddingService,
)
from app.vector_store import FAISSVectorStore


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class EvaluationQuery:
    """Represents a single natural-language retrieval evaluation query.

    Attributes:
        query_id: Unique string identifier (e.g., 'SEC-01').
        question: User's natural-language question.
        expected_doc: Expected relevant document filename (e.g., 'security_policy.txt').
        expected_chunk_ids: List of known relevant chunk IDs (e.g., ['security_policy.txt_page_1_chunk_1']).
        expected_passage_text: Signature text or key phrase contained within the relevant passage.
        fact_description: Brief description of the factual information required to answer.
        query_type: Category of query (e.g., 'direct_factual', 'paraphrased', 'synonym').
    """

    query_id: str
    question: str
    expected_doc: str
    expected_chunk_ids: List[str] = field(default_factory=list)
    expected_passage_text: str = ""
    fact_description: str = ""
    query_type: str = "direct_factual"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationQuery:
        """Create an EvaluationQuery from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        if "query_id" not in data or not str(data["query_id"]).strip():
            raise ValueError("Evaluation query missing required 'query_id'")
        if "question" not in data or not str(data["question"]).strip():
            raise ValueError("Evaluation query missing required 'question'")
        if "expected_doc" not in data or not str(data["expected_doc"]).strip():
            raise ValueError("Evaluation query missing required 'expected_doc'")

        chunk_ids = data.get("expected_chunk_ids", [])
        if isinstance(chunk_ids, str):
            chunk_ids = [chunk_ids]
        elif not isinstance(chunk_ids, list):
            chunk_ids = list(chunk_ids)

        return cls(
            query_id=str(data["query_id"]).strip(),
            question=str(data["question"]).strip(),
            expected_doc=str(data["expected_doc"]).strip(),
            expected_chunk_ids=[str(cid).strip() for cid in chunk_ids],
            expected_passage_text=str(data.get("expected_passage_text", "")).strip(),
            fact_description=str(data.get("fact_description", "")).strip(),
            query_type=str(data.get("query_type", "direct_factual")).strip(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to a serializable dictionary."""
        return asdict(self)


@dataclass
class QueryEvaluationResult:
    """Stores detailed evaluation output for a single evaluation query."""

    query_id: str
    question: str
    expected_doc: str
    expected_chunk_ids: List[str]
    fact_description: str
    retrieved_chunks: List[Dict[str, Any]]
    hit_at_k: Dict[int, int]
    recall_at_k: Dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert query evaluation result to a serializable dictionary."""
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Contains aggregate metrics and full evaluation results across all queries."""

    total_queries: int
    total_indexed_chunks: int
    k_values: List[int]
    hit_rate_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    query_results: List[QueryEvaluationResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation summary to a serializable dictionary."""
        return {
            "total_queries": self.total_queries,
            "total_indexed_chunks": self.total_indexed_chunks,
            "k_values": self.k_values,
            "hit_rate_at_k": {str(k): v for k, v in self.hit_rate_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "mrr": self.mrr,
            "query_results": [qr.to_dict() for qr in self.query_results],
        }


# ============================================================================
# Metric Calculation Functions
# ============================================================================


def validate_k(k: Any) -> int:
    """Validate that K is a strictly positive integer.

    Raises:
        ValueError: If k is not an integer or is <= 0.
        TypeError: If k is a boolean or invalid type.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"k must be an integer, got: {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer (> 0), got: {k}")
    return k


def compute_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> int:
    """Calculate Hit@K for a single query.

    Returns 1 if at least one relevant chunk ID appears in the top K retrieved
    chunk IDs, 0 otherwise.

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Set or sequence of ground-truth relevant chunk IDs.
        k: Cutoff rank. Must be a positive integer.

    Returns:
        1 if hit occurs in top k, 0 otherwise.

    Raises:
        ValueError / TypeError: If k is invalid.
    """
    validate_k(k)
    relevant_set = set(relevant_chunk_ids)
    if not relevant_set or not retrieved_chunk_ids:
        return 0

    top_k_ids = retrieved_chunk_ids[:k]
    for cid in top_k_ids:
        if cid in relevant_set:
            return 1
    return 0


def compute_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Union[Set[str], Sequence[str]],
    k: int,
) -> float:
    """Calculate Recall@K for a single query.

    The proportion of known relevant chunks retrieved in top K results:
        Recall@K = |{retrieved[:K]} ∩ {relevant}| / |{relevant}|

    If each query has only one known relevant chunk, Recall@K is identical
    to Hit@K (either 0.0 or 1.0).

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Set or sequence of ground-truth relevant chunk IDs.
        k: Cutoff rank. Must be a positive integer.

    Returns:
        Float value between 0.0 and 1.0. If relevant_chunk_ids is empty, returns 0.0.

    Raises:
        ValueError / TypeError: If k is invalid.
    """
    validate_k(k)
    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    if not retrieved_chunk_ids:
        return 0.0

    top_k_ids = retrieved_chunk_ids[:k]
    hits = sum(1 for cid in set(top_k_ids) if cid in relevant_set)
    return float(hits) / float(len(relevant_set))


def compute_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Union[Set[str], Sequence[str]],
    max_k: Optional[int] = None,
) -> Tuple[float, Optional[int]]:
    """Calculate the Reciprocal Rank (RR) and 1-based rank of the first relevant chunk.

    Reciprocal Rank is 1 / rank of the first relevant chunk retrieved.
    If no relevant chunk is retrieved (or its rank exceeds max_k), RR is 0.0.

    Args:
        retrieved_chunk_ids: Ordered sequence of retrieved chunk IDs.
        relevant_chunk_ids: Set or sequence of ground-truth relevant chunk IDs.
        max_k: Optional maximum cutoff rank to inspect.

    Returns:
        Tuple of (reciprocal_rank: float, first_relevant_rank: Optional[int]).
        If not found, returns (0.0, None).

    Raises:
        ValueError / TypeError: If max_k is provided and invalid.
    """
    if max_k is not None:
        validate_k(max_k)
        candidates = retrieved_chunk_ids[:max_k]
    else:
        candidates = retrieved_chunk_ids

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set or not candidates:
        return 0.0, None

    for rank, cid in enumerate(candidates, start=1):
        if cid in relevant_set:
            return 1.0 / float(rank), rank

    return 0.0, None


def is_chunk_relevant(
    chunk: Dict[str, Any],
    query: EvaluationQuery,
) -> bool:
    """Determine whether a retrieved chunk is relevant to an evaluation query.

    A retrieved chunk is relevant if:
      1. Its source document matches `query.expected_doc`.
      2. AND either:
         a) Its `chunk_id` is in `query.expected_chunk_ids`, OR
         b) Its text contains `query.expected_passage_text` (case-insensitive substring match).

    Args:
        chunk: Result dictionary containing 'chunk_id', 'text', and 'metadata'.
        query: Ground-truth EvaluationQuery object.

    Returns:
        True if the chunk meets the relevance criteria, False otherwise.
    """
    if not isinstance(chunk, dict):
        return False

    metadata = chunk.get("metadata", {})
    doc_name = metadata.get("filename") or ""

    # Document filename check (must match expected document)
    if doc_name != query.expected_doc:
        return False

    chunk_id = chunk.get("chunk_id", "")
    chunk_text = chunk.get("text", "")

    # Match by explicit expected chunk IDs
    if query.expected_chunk_ids and chunk_id in query.expected_chunk_ids:
        return True

    # Match by expected passage text / key phrase
    if query.expected_passage_text:
        passage_needle = query.expected_passage_text.strip().lower()
        if passage_needle and passage_needle in chunk_text.lower():
            return True

    return False


# ============================================================================
# Retrieval Evaluator Harness
# ============================================================================


class RetrievalEvaluator:
    """Orchestrates end-to-end retrieval testing and quality evaluation.

    Encapsulates:
      - Document ingestion via process_document
      - Text chunking via chunk_document
      - Vector embeddings via EmbeddingService
      - Vector storage and search via FAISSVectorStore
      - Metric calculation (Hit Rate@K, Recall@K, MRR)
      - JSON and CSV reporting
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        k_values: Sequence[int] = (1, 3, 5),
    ) -> None:
        """Initialize the retrieval evaluator.

        Args:
            embedding_service: EmbeddingService instance. If None, initialized lazily.
            vector_store: FAISSVectorStore instance. If None, created with dimension=384.
            k_values: Sequence of K cutoff values to evaluate (default: (1, 3, 5)).
        """
        for k in k_values:
            validate_k(k)
        self.k_values: List[int] = sorted(list(set(k_values)))

        self.embedding_service: EmbeddingService = (
            embedding_service if embedding_service is not None else EmbeddingService()
        )
        self.vector_store: FAISSVectorStore = (
            vector_store
            if vector_store is not None
            else FAISSVectorStore(
                dimension=self.embedding_service.embedding_dimension,
                normalize_vectors=True,
            )
        )
        self._indexed_chunks: List[Dict[str, Any]] = []

    @property
    def indexed_chunks(self) -> List[Dict[str, Any]]:
        """Return a copy of all chunks currently indexed in the vector store."""
        return [copy.deepcopy(c) for c in self._indexed_chunks]

    def load_documents_from_directory(
        self,
        directory_path: Union[str, os.PathLike],
    ) -> List[Dict[str, Any]]:
        """Process all supported documents (.pdf, .txt, .docx) from a directory.

        Args:
            directory_path: Path to directory containing document files.

        Returns:
            List of processed document dictionaries from `process_document()`.

        Raises:
            FileNotFoundError: If directory does not exist.
            ValueError: If directory contains no supported readable documents.
        """
        dir_p = Path(directory_path)
        if not dir_p.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {directory_path}")
        if not dir_p.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")

        files = sorted([
            p for p in dir_p.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not files:
            raise ValueError(
                f"No supported document files found in: {directory_path}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        processed_docs: List[Dict[str, Any]] = []
        for file_path in files:
            doc = process_document(file_path)
            processed_docs.append(doc)

        return processed_docs

    def build_evaluation_index(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Chunk, embed, and index processed documents into the vector store.

        Clears existing index state before building the new index.

        Args:
            documents: List of processed document dicts (from process_document).
            chunk_size: Chunk size parameter for chunk_document (default 800).
            chunk_overlap: Overlap parameter for chunk_document (default 150).

        Returns:
            Total number of chunks indexed.
        """
        if documents is None:
            raise ValueError("documents list cannot be None")

        self.vector_store.clear()
        self._indexed_chunks.clear()

        all_chunks: List[Dict[str, Any]] = []
        for doc in documents:
            chunks = chunk_document(
                doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        embedded_chunks = self.embedding_service.embed_chunks(all_chunks)
        self.vector_store.add_chunks(embedded_chunks)
        self._indexed_chunks = copy.deepcopy(embedded_chunks)

        return len(self._indexed_chunks)

    def evaluate_query(
        self,
        query: Union[EvaluationQuery, Dict[str, Any]],
        top_k: int = 5,
    ) -> QueryEvaluationResult:
        """Evaluate retrieval performance for a single evaluation query.

        Embeds the query, retrieves nearest chunks from FAISS, evaluates
        relevance against expected chunks, and computes query-level metrics.

        Args:
            query: EvaluationQuery instance or dictionary with query details.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            QueryEvaluationResult containing retrieved chunks and metrics.
        """
        validate_k(top_k)

        if isinstance(query, dict):
            eq = EvaluationQuery.from_dict(query)
        elif isinstance(query, EvaluationQuery):
            eq = query
        else:
            raise TypeError(f"query must be EvaluationQuery or dict, got: {type(query).__name__}")

        # 1. Embed query vector in same semantic space as document chunks
        query_vector = self.embedding_service.embed_text(eq.question)

        # 2. Retrieve top_k results from FAISS
        raw_results = self.vector_store.search(query_vector, top_k=top_k)

        # 3. Determine relevance for each retrieved chunk
        decorated_results: List[Dict[str, Any]] = []
        retrieved_chunk_ids: List[str] = []
        retrieved_relevant_ids: List[str] = []

        for rank, item in enumerate(raw_results, start=1):
            is_rel = is_chunk_relevant(item, eq)
            decorated_item = {
                "rank": rank,
                "chunk_id": item["chunk_id"],
                "similarity_score": float(item["similarity_score"]),
                "is_relevant": is_rel,
                "metadata": copy.deepcopy(item.get("metadata", {})),
                "text": item.get("text", ""),
            }
            decorated_results.append(decorated_item)
            retrieved_chunk_ids.append(item["chunk_id"])
            if is_rel:
                retrieved_relevant_ids.append(item["chunk_id"])

        # Ground truth relevant IDs
        ground_truth_relevant = set(eq.expected_chunk_ids)

        # If expected_chunk_ids was not explicitly pre-populated, use chunks matching text
        if not ground_truth_relevant:
            # Find all indexed chunks matching doc and text
            for ch in self._indexed_chunks:
                if is_chunk_relevant(ch, eq):
                    ground_truth_relevant.add(ch["chunk_id"])

        # 4. Compute metrics at each K
        hit_at_k: Dict[int, int] = {}
        recall_at_k: Dict[int, float] = {}

        for k in self.k_values:
            # Evaluate only up to min(k, top_k)
            eff_k = min(k, top_k)
            hit = compute_hit_at_k(retrieved_chunk_ids, ground_truth_relevant, eff_k)
            rec = compute_recall_at_k(retrieved_chunk_ids, ground_truth_relevant, eff_k)
            hit_at_k[k] = hit
            recall_at_k[k] = rec

        # 5. Compute reciprocal rank
        max_eval_k = max(self.k_values + [top_k])
        rr, first_rank = compute_reciprocal_rank(
            retrieved_chunk_ids, ground_truth_relevant, max_k=max_eval_k
        )

        return QueryEvaluationResult(
            query_id=eq.query_id,
            question=eq.question,
            expected_doc=eq.expected_doc,
            expected_chunk_ids=sorted(list(ground_truth_relevant)),
            fact_description=eq.fact_description,
            retrieved_chunks=decorated_results,
            hit_at_k=hit_at_k,
            recall_at_k=recall_at_k,
            reciprocal_rank=rr,
            first_relevant_rank=first_rank,
        )

    def evaluate(
        self,
        queries: Sequence[Union[EvaluationQuery, Dict[str, Any]]],
        top_k: int = 5,
    ) -> EvaluationSummary:
        """Run full retrieval evaluation across a sequence of queries.

        Args:
            queries: Sequence of EvaluationQuery objects or dicts.
            top_k: Number of nearest chunks to retrieve per query (default 5).

        Returns:
            EvaluationSummary with aggregate metrics and per-query details.
        """
        validate_k(top_k)

        if queries is None:
            raise ValueError("queries cannot be None")

        query_list = list(queries)
        if len(query_list) == 0:
            return EvaluationSummary(
                total_queries=0,
                total_indexed_chunks=self.vector_store.total_vectors,
                k_values=self.k_values,
                hit_rate_at_k={k: 0.0 for k in self.k_values},
                recall_at_k={k: 0.0 for k in self.k_values},
                mrr=0.0,
                query_results=[],
            )

        query_results: List[QueryEvaluationResult] = []
        for q in query_list:
            res = self.evaluate_query(q, top_k=top_k)
            query_results.append(res)

        total_q = len(query_results)

        # Aggregate Hit Rate@K
        hit_rate_at_k: Dict[int, float] = {}
        for k in self.k_values:
            total_hits = sum(qr.hit_at_k.get(k, 0) for qr in query_results)
            hit_rate_at_k[k] = float(total_hits) / float(total_q)

        # Aggregate Recall@K
        recall_at_k: Dict[int, float] = {}
        for k in self.k_values:
            total_rec = sum(qr.recall_at_k.get(k, 0.0) for qr in query_results)
            recall_at_k[k] = float(total_rec) / float(total_q)

        # Aggregate Mean Reciprocal Rank (MRR)
        total_rr = sum(qr.reciprocal_rank for qr in query_results)
        mrr = float(total_rr) / float(total_q)

        return EvaluationSummary(
            total_queries=total_q,
            total_indexed_chunks=self.vector_store.total_vectors,
            k_values=self.k_values,
            hit_rate_at_k=hit_rate_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            query_results=query_results,
        )

    def export_results_json(
        self,
        summary: EvaluationSummary,
        output_path: Union[str, Path],
    ) -> Path:
        """Export evaluation summary to a structured, reproducible JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = summary.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return path

    def export_results_csv(
        self,
        summary: EvaluationSummary,
        output_path: Union[str, Path],
    ) -> Path:
        """Export query-level evaluation results to a CSV file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "query_id",
            "question",
            "expected_doc",
            "first_relevant_rank",
            "reciprocal_rank",
        ]
        for k in summary.k_values:
            fieldnames.append(f"hit@{k}")
            fieldnames.append(f"recall@{k}")
        fieldnames.extend([
            "top1_chunk_id",
            "top1_similarity_score",
            "top1_source_file",
            "top1_is_relevant",
            "fact_description",
        ])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for qr in summary.query_results:
                top1 = qr.retrieved_chunks[0] if qr.retrieved_chunks else {}
                row: Dict[str, Any] = {
                    "query_id": qr.query_id,
                    "question": qr.question,
                    "expected_doc": qr.expected_doc,
                    "first_relevant_rank": qr.first_relevant_rank if qr.first_relevant_rank else "N/A",
                    "reciprocal_rank": f"{qr.reciprocal_rank:.4f}",
                }
                for k in summary.k_values:
                    row[f"hit@{k}"] = qr.hit_at_k.get(k, 0)
                    row[f"recall@{k}"] = f"{qr.recall_at_k.get(k, 0.0):.4f}"

                row["top1_chunk_id"] = top1.get("chunk_id", "N/A")
                row["top1_similarity_score"] = (
                    f"{top1['similarity_score']:.4f}" if "similarity_score" in top1 else "N/A"
                )
                row["top1_source_file"] = (
                    top1.get("metadata", {}).get("filename", "N/A")
                )
                row["top1_is_relevant"] = top1.get("is_relevant", False)
                row["fact_description"] = qr.fact_description

                writer.writerow(row)

        return path


# ============================================================================
# Convenience Functions
# ============================================================================


def load_queries_from_json(json_path: Union[str, Path]) -> List[EvaluationQuery]:
    """Load evaluation queries from a JSON file.

    Args:
        json_path: Path to JSON file containing a list of query objects.

    Returns:
        List of EvaluationQuery instances.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is malformed or not a list.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list of queries, got: {type(data).__name__}")

    return [EvaluationQuery.from_dict(item) for item in data]
