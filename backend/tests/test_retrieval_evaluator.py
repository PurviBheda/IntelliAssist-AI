"""Automated unit tests for IntelliAssist AI Retrieval Evaluation Module.

Verifies:
  1. Metric calculations using small deterministic examples (Hit@K, Recall@K, MRR).
  2. Correct handling of a query with no relevant results.
  3. Correct handling of empty retrieved results.
  4. Correct ranking and relevance matching.
  5. Correct behavior for multiple relevant passages.
  6. Invalid K and max_k values are handled appropriately.
  7. Aggregate metrics are calculated correctly.
  8. Evaluation results preserve query IDs, expected source, and metadata.
  9. Evaluation code integrates cleanly with existing document processor,
     chunker, embedding service, and FAISS vector store.
  10. JSON and CSV export functionality produces reproducible reports.
  11. Evaluation queries JSON loading and validation.

All unit tests use mock embeddings to guarantee fast, deterministic execution
without downloading models during pytest runs.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.retrieval_evaluator import (
    EvaluationQuery,
    EvaluationSummary,
    QueryEvaluationResult,
    RetrievalEvaluator,
    compute_hit_at_k,
    compute_recall_at_k,
    compute_reciprocal_rank,
    is_chunk_relevant,
    load_queries_from_json,
    validate_k,
)
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore


# ============================================================================
# Helpers & Mock Factories
# ============================================================================


def _create_mock_embedding_service(dim: int = 384) -> EmbeddingService:
    """Create a mock EmbeddingService with deterministic unit-normalized vectors."""
    service = EmbeddingService(embedding_dimension=dim)
    mock_model = MagicMock()

    def mock_encode(
        sentences,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ):
        if isinstance(sentences, str):
            rng = np.random.default_rng(seed=abs(hash(sentences)) % (2**32))
            vec = rng.standard_normal(dim).astype(np.float32)
            if normalize_embeddings:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            return vec
        elif isinstance(sentences, list):
            rows = []
            for s in sentences:
                rng = np.random.default_rng(seed=abs(hash(s)) % (2**32))
                vec = rng.standard_normal(dim).astype(np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                rows.append(vec)
            if not rows:
                return np.empty((0, dim), dtype=np.float32)
            return np.vstack(rows)

    mock_model.encode.side_effect = mock_encode
    service._model = mock_model
    return service


# ============================================================================
# 1. Deterministic Metric Calculations
# ============================================================================


def test_compute_hit_at_k_deterministic() -> None:
    """1. Hit@K correctly identifies hits at various cutoff thresholds."""
    retrieved = ["docA_chunk1", "docA_chunk2", "docB_chunk1", "docC_chunk1"]
    relevant = {"docA_chunk2"}

    # Not in top 1
    assert compute_hit_at_k(retrieved, relevant, k=1) == 0
    # In top 2
    assert compute_hit_at_k(retrieved, relevant, k=2) == 1
    # In top 3
    assert compute_hit_at_k(retrieved, relevant, k=3) == 1
    # In top 4
    assert compute_hit_at_k(retrieved, relevant, k=4) == 1

    # First position hit
    assert compute_hit_at_k(retrieved, {"docA_chunk1"}, k=1) == 1

    # Completely absent chunk
    assert compute_hit_at_k(retrieved, {"non_existent_chunk"}, k=4) == 0


def test_compute_recall_at_k_single_relevant_equals_hit_rate() -> None:
    """2. Recall@K equals Hit@K when there is exactly one known relevant chunk."""
    retrieved = ["chunk_1", "chunk_2", "chunk_3"]
    relevant = {"chunk_2"}

    hit_1 = compute_hit_at_k(retrieved, relevant, k=1)
    recall_1 = compute_recall_at_k(retrieved, relevant, k=1)
    assert hit_1 == 0
    assert recall_1 == 0.0

    hit_2 = compute_hit_at_k(retrieved, relevant, k=2)
    recall_2 = compute_recall_at_k(retrieved, relevant, k=2)
    assert hit_2 == 1
    assert recall_2 == 1.0


def test_compute_recall_at_k_multiple_relevant_passages() -> None:
    """3. Recall@K accurately measures proportion of retrieved relevant chunks."""
    retrieved = ["c1", "c2", "c3", "c4"]
    # 3 relevant chunks in total: c1, c3, and c9 (c9 not retrieved)
    relevant = {"c1", "c3", "c9"}

    # At K=1: only c1 retrieved -> 1/3
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=1)) == 1.0 / 3.0

    # At K=2: c1 retrieved -> 1/3
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=2)) == 1.0 / 3.0

    # At K=3: c1 and c3 retrieved -> 2/3
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=3)) == 2.0 / 3.0

    # At K=4: c1 and c3 retrieved -> 2/3 (c9 was never retrieved)
    assert pytest.approx(compute_recall_at_k(retrieved, relevant, k=4)) == 2.0 / 3.0


def test_compute_reciprocal_rank_calculation() -> None:
    """4. MRR / Reciprocal Rank correctly computes 1/rank."""
    retrieved = ["c_alpha", "c_beta", "c_gamma", "c_delta"]

    # Rank 1: RR = 1/1 = 1.0
    rr, rank = compute_reciprocal_rank(retrieved, {"c_alpha"})
    assert rr == 1.0
    assert rank == 1

    # Rank 2: RR = 1/2 = 0.5
    rr, rank = compute_reciprocal_rank(retrieved, {"c_beta"})
    assert rr == 0.5
    assert rank == 2

    # Rank 3: RR = 1/3
    rr, rank = compute_reciprocal_rank(retrieved, {"c_gamma"})
    assert pytest.approx(rr) == 1.0 / 3.0
    assert rank == 3

    # Rank 4: RR = 1/4 = 0.25
    rr, rank = compute_reciprocal_rank(retrieved, {"c_delta"})
    assert rr == 0.25
    assert rank == 4

    # Not found in retrieved results
    rr, rank = compute_reciprocal_rank(retrieved, {"c_omega"})
    assert rr == 0.0
    assert rank is None

    # Cutoff via max_k: c_delta is at rank 4, but max_k=3
    rr, rank = compute_reciprocal_rank(retrieved, {"c_delta"}, max_k=3)
    assert rr == 0.0
    assert rank is None


# ============================================================================
# 2. Empty Results and No-Relevant Queries Handling
# ============================================================================


def test_handling_of_empty_retrieved_results() -> None:
    """5. Empty retrieved list returns 0 for all metrics safely."""
    empty_retrieved: list[str] = []
    relevant = {"c1", "c2"}

    assert compute_hit_at_k(empty_retrieved, relevant, k=1) == 0
    assert compute_hit_at_k(empty_retrieved, relevant, k=5) == 0
    assert compute_recall_at_k(empty_retrieved, relevant, k=1) == 0.0
    assert compute_recall_at_k(empty_retrieved, relevant, k=5) == 0.0

    rr, rank = compute_reciprocal_rank(empty_retrieved, relevant)
    assert rr == 0.0
    assert rank is None


def test_handling_of_query_with_no_relevant_results() -> None:
    """6. Empty relevant set returns 0 without divide-by-zero errors."""
    retrieved = ["c1", "c2"]
    no_relevant: set[str] = set()

    assert compute_hit_at_k(retrieved, no_relevant, k=1) == 0
    assert compute_recall_at_k(retrieved, no_relevant, k=1) == 0.0

    rr, rank = compute_reciprocal_rank(retrieved, no_relevant)
    assert rr == 0.0
    assert rank is None


# ============================================================================
# 3. Validation & Invalid K Handling
# ============================================================================


def test_invalid_k_values_raise_appropriate_exceptions() -> None:
    """7. Invalid K values (zero, negative, non-integer, bool) are rejected."""
    retrieved = ["c1"]
    relevant = {"c1"}

    # Zero or negative K
    with pytest.raises(ValueError, match="positive integer"):
        validate_k(0)
    with pytest.raises(ValueError, match="positive integer"):
        compute_hit_at_k(retrieved, relevant, k=0)
    with pytest.raises(ValueError, match="positive integer"):
        compute_recall_at_k(retrieved, relevant, k=-2)
    with pytest.raises(ValueError, match="positive integer"):
        compute_reciprocal_rank(retrieved, relevant, max_k=-1)

    # Boolean input (in Python, isinstance(True, int) is True)
    with pytest.raises(TypeError, match="must be an integer"):
        validate_k(True)  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        compute_hit_at_k(retrieved, relevant, k=False)  # type: ignore

    # Non-integer type
    with pytest.raises(TypeError, match="must be an integer"):
        validate_k("5")  # type: ignore
    with pytest.raises(TypeError, match="must be an integer"):
        validate_k(3.14)  # type: ignore


# ============================================================================
# 4. Relevance Matching Logic
# ============================================================================


def test_is_chunk_relevant_matching() -> None:
    """8. is_chunk_relevant verifies document match, chunk_id, and passage text."""
    query = EvaluationQuery(
        query_id="Q1",
        question="What is the password requirement?",
        expected_doc="security_policy.txt",
        expected_chunk_ids=["security_policy.txt_page_1_chunk_1"],
        expected_passage_text="14 characters in length",
    )

    # Correct document and exact chunk ID
    c1 = {
        "chunk_id": "security_policy.txt_page_1_chunk_1",
        "text": "All employee passwords must be a minimum of 14 characters in length.",
        "metadata": {"filename": "security_policy.txt", "page_number": 1},
    }
    assert is_chunk_relevant(c1, query) is True

    # Correct document and matching text phrase even if chunk_id differs
    c2 = {
        "chunk_id": "security_policy.txt_page_1_chunk_99",
        "text": "System credentials require 14 characters in length at minimum.",
        "metadata": {"filename": "security_policy.txt", "page_number": 1},
    }
    assert is_chunk_relevant(c2, query) is True

    # Wrong document: must be rejected even if text or chunk_id match
    c3 = {
        "chunk_id": "security_policy.txt_page_1_chunk_1",
        "text": "14 characters in length",
        "metadata": {"filename": "employee_handbook.txt", "page_number": 1},
    }
    assert is_chunk_relevant(c3, query) is False

    # Correct document but completely irrelevant text and chunk_id
    c4 = {
        "chunk_id": "security_policy.txt_page_1_chunk_4",
        "text": "Access rights follow role-based permissions.",
        "metadata": {"filename": "security_policy.txt", "page_number": 1},
    }
    assert is_chunk_relevant(c4, query) is False

    # Malformed chunk input
    assert is_chunk_relevant("not a dict", query) is False  # type: ignore


# ============================================================================
# 5. Data Model Validation
# ============================================================================


def test_evaluation_query_model_validation() -> None:
    """9. EvaluationQuery enforces required fields and supports serialization."""
    valid_data = {
        "query_id": "SEC-01",
        "question": "What is the password rule?",
        "expected_doc": "security_policy.txt",
        "expected_chunk_ids": ["security_policy.txt_page_1_chunk_1"],
        "expected_passage_text": "14 characters",
        "fact_description": "Password length",
    }
    eq = EvaluationQuery.from_dict(valid_data)
    assert eq.query_id == "SEC-01"
    assert eq.expected_chunk_ids == ["security_policy.txt_page_1_chunk_1"]
    assert eq.to_dict()["query_id"] == "SEC-01"

    # Missing query_id
    with pytest.raises(ValueError, match="query_id"):
        EvaluationQuery.from_dict({"question": "Q?", "expected_doc": "doc.txt"})

    # Missing question
    with pytest.raises(ValueError, match="question"):
        EvaluationQuery.from_dict({"query_id": "1", "expected_doc": "doc.txt"})

    # Missing expected_doc
    with pytest.raises(ValueError, match="expected_doc"):
        EvaluationQuery.from_dict({"query_id": "1", "question": "Q?"})

    # Non-dict input
    with pytest.raises(TypeError, match="Expected dict"):
        EvaluationQuery.from_dict("invalid")  # type: ignore


# ============================================================================
# 6. Aggregate Metric Calculations
# ============================================================================


def test_aggregate_metrics_calculation() -> None:
    """10. EvaluationSummary correctly computes averages across queries."""
    # Query 1: Hit@1=1, Rec@1=1.0, RR=1.0
    qr1 = QueryEvaluationResult(
        query_id="Q1",
        question="Q1?",
        expected_doc="doc1.txt",
        expected_chunk_ids=["c1"],
        fact_description="Fact 1",
        retrieved_chunks=[{"chunk_id": "c1", "is_relevant": True, "rank": 1}],
        hit_at_k={1: 1, 3: 1, 5: 1},
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        reciprocal_rank=1.0,
        first_relevant_rank=1,
    )
    # Query 2: Hit@1=0, Hit@3=1, Rec@1=0.0, Rec@3=1.0, RR=0.5
    qr2 = QueryEvaluationResult(
        query_id="Q2",
        question="Q2?",
        expected_doc="doc2.txt",
        expected_chunk_ids=["c2"],
        fact_description="Fact 2",
        retrieved_chunks=[
            {"chunk_id": "irrelevant", "is_relevant": False, "rank": 1},
            {"chunk_id": "c2", "is_relevant": True, "rank": 2},
        ],
        hit_at_k={1: 0, 3: 1, 5: 1},
        recall_at_k={1: 0.0, 3: 1.0, 5: 1.0},
        reciprocal_rank=0.5,
        first_relevant_rank=2,
    )

    evaluator = RetrievalEvaluator(k_values=(1, 3, 5))
    # Mock evaluate to test aggregation directly
    with patch.object(evaluator, "evaluate_query", side_effect=[qr1, qr2]):
        summary = evaluator.evaluate([{"query_id": "Q1"}, {"query_id": "Q2"}])

    assert summary.total_queries == 2
    # Hit Rate: (1 + 0)/2 = 0.5 at K=1; (1 + 1)/2 = 1.0 at K=3
    assert summary.hit_rate_at_k[1] == 0.5
    assert summary.hit_rate_at_k[3] == 1.0
    assert summary.hit_rate_at_k[5] == 1.0

    # Recall: (1.0 + 0.0)/2 = 0.5 at K=1; (1.0 + 1.0)/2 = 1.0 at K=3
    assert summary.recall_at_k[1] == 0.5
    assert summary.recall_at_k[3] == 1.0

    # MRR: (1.0 + 0.5)/2 = 0.75
    assert summary.mrr == 0.75


def test_evaluator_handles_empty_queries_list() -> None:
    """11. Evaluating an empty queries list returns zeroed summary safely."""
    evaluator = RetrievalEvaluator(k_values=(1, 3, 5))
    summary = evaluator.evaluate([])

    assert summary.total_queries == 0
    assert summary.mrr == 0.0
    assert summary.hit_rate_at_k == {1: 0.0, 3: 0.0, 5: 0.0}
    assert summary.recall_at_k == {1: 0.0, 3: 0.0, 5: 0.0}
    assert summary.query_results == []


# ============================================================================
# 7. End-to-End Evaluator Integration with Mock Embedding and FAISS
# ============================================================================


def test_evaluator_integration_with_mock_embeddings_and_faiss() -> None:
    """12. Evaluator indexes documents, queries FAISS, and preserves metadata."""
    mock_service = _create_mock_embedding_service(dim=384)
    vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

    evaluator = RetrievalEvaluator(
        embedding_service=mock_service,
        vector_store=vector_store,
        k_values=(1, 3),
    )

    doc_sample = {
        "filename": "security_test.txt",
        "file_type": "txt",
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "Passwords must be at least 14 characters with letters, numbers, and symbols. "
                    "Accounts lock after 5 failed attempts for 30 minutes."
                ),
            }
        ],
        "total_pages": 1,
        "total_characters": 130,
    }

    # Build index
    indexed_count = evaluator.build_evaluation_index([doc_sample], chunk_size=300, chunk_overlap=50)
    assert indexed_count == 1
    assert vector_store.total_vectors == 1

    # Run query
    query = EvaluationQuery(
        query_id="SEC-T1",
        question="What is the password requirement?",
        expected_doc="security_test.txt",
        expected_chunk_ids=["security_test.txt_page_1_chunk_1"],
        expected_passage_text="14 characters",
        fact_description="14 char requirement",
    )

    result = evaluator.evaluate_query(query, top_k=1)
    assert result.query_id == "SEC-T1"
    assert result.expected_doc == "security_test.txt"
    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0]["chunk_id"] == "security_test.txt_page_1_chunk_1"
    assert result.retrieved_chunks[0]["is_relevant"] is True
    assert result.hit_at_k[1] == 1
    assert result.recall_at_k[1] == 1.0
    assert result.reciprocal_rank == 1.0


# ============================================================================
# 8. Report Export Tests (JSON & CSV)
# ============================================================================


def test_export_results_json_and_csv(tmp_path: Path) -> None:
    """13. Export methods write valid JSON and CSV formats."""
    qr = QueryEvaluationResult(
        query_id="TEST-01",
        question="How many days off?",
        expected_doc="handbook.txt",
        expected_chunk_ids=["handbook.txt_page_1_chunk_1"],
        fact_description="20 days PTO",
        retrieved_chunks=[
            {
                "chunk_id": "handbook.txt_page_1_chunk_1",
                "similarity_score": 0.88,
                "is_relevant": True,
                "rank": 1,
                "metadata": {"filename": "handbook.txt", "page_number": 1},
                "text": "Employees receive 20 days PTO.",
            }
        ],
        hit_at_k={1: 1, 3: 1, 5: 1},
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        reciprocal_rank=1.0,
        first_relevant_rank=1,
    )
    summary = EvaluationSummary(
        total_queries=1,
        total_indexed_chunks=1,
        k_values=[1, 3, 5],
        hit_rate_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        recall_at_k={1: 1.0, 3: 1.0, 5: 1.0},
        mrr=1.0,
        query_results=[qr],
    )

    evaluator = RetrievalEvaluator()

    # 1. Test JSON Export
    json_file = tmp_path / "report.json"
    evaluator.export_results_json(summary, json_file)
    assert json_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_queries"] == 1
    assert data["mrr"] == 1.0
    assert data["hit_rate_at_k"]["1"] == 1.0
    assert len(data["query_results"]) == 1
    assert data["query_results"][0]["query_id"] == "TEST-01"

    # 2. Test CSV Export
    csv_file = tmp_path / "report.csv"
    evaluator.export_results_csv(summary, csv_file)
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["query_id"] == "TEST-01"
    assert rows[0]["expected_doc"] == "handbook.txt"
    assert rows[0]["top1_chunk_id"] == "handbook.txt_page_1_chunk_1"
    assert rows[0]["hit@1"] == "1"


# ============================================================================
# 9. Query Loading and Directory Ingestion
# ============================================================================


def test_load_queries_from_json(tmp_path: Path) -> None:
    """14. load_queries_from_json parses valid JSON array and handles errors."""
    valid_data = [
        {
            "query_id": "Q-01",
            "question": "What is the leave policy?",
            "expected_doc": "employee_handbook.txt",
            "expected_chunk_ids": ["c1"],
        }
    ]
    file_path = tmp_path / "queries.json"
    file_path.write_text(json.dumps(valid_data), encoding="utf-8")

    queries = load_queries_from_json(file_path)
    assert len(queries) == 1
    assert queries[0].query_id == "Q-01"
    assert queries[0].expected_doc == "employee_handbook.txt"

    # Missing file
    with pytest.raises(FileNotFoundError):
        load_queries_from_json(tmp_path / "non_existent.json")

    # Invalid JSON structure (not a list)
    dict_file = tmp_path / "dict_queries.json"
    dict_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON list"):
        load_queries_from_json(dict_file)


def test_load_documents_from_directory_validation(tmp_path: Path) -> None:
    """15. load_documents_from_directory validates paths and file extensions."""
    evaluator = RetrievalEvaluator()

    # Non-existent directory
    with pytest.raises(FileNotFoundError):
        evaluator.load_documents_from_directory(tmp_path / "missing_dir")

    # Path is a file, not directory
    sample_file = tmp_path / "test.txt"
    sample_file.write_text("Hello world", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        evaluator.load_documents_from_directory(sample_file)

    # Directory with no supported files
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    (empty_dir / "unsupported.csv").write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="No supported document files"):
        evaluator.load_documents_from_directory(empty_dir)
