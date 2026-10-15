"""IntelliAssist AI - FAISS Vector Store Smoke Test.

This script demonstrates and verifies that FAISS vector indexing and similarity retrieval
operate correctly using deterministic synthetic vectors and multi-document chunks.
It requires zero network connectivity and no heavy model downloads.

Usage:
    python smoke_test_vector_store.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.vector_store import FAISSVectorStore


def make_normalized_vector(dim: int, seed: int) -> np.ndarray:
    """Create a deterministic unit-length vector for reproducible testing."""
    rng = np.random.default_rng(seed=seed)
    raw = rng.standard_normal(dim).astype(np.float32)
    return raw / np.linalg.norm(raw)


def run_smoke_test() -> None:
    print("=" * 75)
    print("IntelliAssist AI -- FAISS Vector Store Smoke Test")
    print("=" * 75)

    dimension = 384
    print(f"\n[Step 1] Initializing FAISSVectorStore (dimension={dimension})...")
    store = FAISSVectorStore(dimension=dimension, normalize_vectors=True)
    print(f"  Initialized store: total_vectors={store.total_vectors}, is_empty={store.is_empty}")
    assert store.total_vectors == 0
    assert store.is_empty is True

    print("\n[Step 2] Preparing document chunks across multiple sources...")
    # Multi-document setup: HR Handbook (PDF) & Security Policy (DOCX)
    chunks = [
        {
            "chunk_id": "handbook.pdf_page_1_chunk_1",
            "text": "IntelliAssist AI standard working hours, flex-time, and core meeting schedules.",
            "metadata": {
                "filename": "handbook.pdf",
                "file_type": "pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "handbook.pdf_page_2_chunk_1",
            "text": "Paid time off, sick leave, and parental leave notification procedures.",
            "metadata": {
                "filename": "handbook.pdf",
                "file_type": "pdf",
                "page_number": 2,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "security_policy.docx_page_1_chunk_1",
            "text": "Mandatory multi-factor authentication (MFA) and 14-character minimum password rules.",
            "metadata": {
                "filename": "security_policy.docx",
                "file_type": "docx",
                "page_number": 1,
                "chunk_index": 1,
            },
        },
        {
            "chunk_id": "security_policy.docx_page_1_chunk_2",
            "text": "Data classification guidelines: Public, Internal, Confidential, and Restricted tiers.",
            "metadata": {
                "filename": "security_policy.docx",
                "file_type": "docx",
                "page_number": 1,
                "chunk_index": 2,
            },
        },
    ]

    # Generate distinct deterministic vectors for each chunk
    v0 = make_normalized_vector(dimension, seed=101)  # Handbook p1
    v1 = make_normalized_vector(dimension, seed=202)  # Handbook p2
    v2 = make_normalized_vector(dimension, seed=303)  # Security MFA (Target)
    v3 = make_normalized_vector(dimension, seed=404)  # Security tiers
    embeddings = np.vstack([v0, v1, v2, v3])

    print(f"  Prepared {len(chunks)} chunks with embeddings shape {embeddings.shape}.")

    print("\n[Step 3] Adding chunks and embeddings into FAISS index...")
    added_count = store.add_chunks(chunks, embeddings)
    print(f"  Successfully added {added_count} chunks.")
    print(f"  Store status: total_vectors={store.total_vectors}, is_empty={store.is_empty}")
    assert added_count == 4
    assert store.total_vectors == 4
    assert store.is_empty is False

    print("\n[Step 4] Querying FAISS index with a query vector close to 'Security MFA'...")
    # Query vector is perturbed v2 (90% v2 + 10% noise), normalized
    noise = make_normalized_vector(dimension, seed=999)
    query_raw = 0.90 * v2 + 0.10 * noise
    query_vector = query_raw / np.linalg.norm(query_raw)

    top_k = 4
    results = store.search(query_vector, top_k=top_k)

    print(f"\n[Step 5] Top {len(results)} Retrieval Results (Ranked by Cosine Similarity):")
    print("-" * 75)
    for rank, res in enumerate(results, start=1):
        print(f"Rank #{rank}:")
        print(f"  Chunk ID:         {res['chunk_id']}")
        print(f"  Similarity Score: {res['similarity_score']:.5f}")
        print(f"  Source File:      {res['metadata']['filename']}")
        print(f"  Page Number:      {res['metadata']['page_number']}")
        print(f"  Text Excerpt:     \"{res['text'][:65]}...\"")
        print("-" * 75)

    # Assertions on retrieval quality
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    best_result = results[0]
    assert best_result["chunk_id"] == "security_policy.docx_page_1_chunk_1", (
        f"Expected top match to be 'security_policy.docx_page_1_chunk_1', got '{best_result['chunk_id']}'"
    )
    assert best_result["metadata"]["filename"] == "security_policy.docx"
    assert best_result["metadata"]["page_number"] == 1
    assert best_result["similarity_score"] > 0.85, (
        f"Expected top similarity > 0.85, got {best_result['similarity_score']}"
    )

    # Verify descending similarity scores
    for i in range(len(results) - 1):
        assert results[i]["similarity_score"] >= results[i + 1]["similarity_score"], (
            f"Results not sorted in descending order: {results[i]['similarity_score']} < {results[i + 1]['similarity_score']}"
        )

    print("\n[Step 6] Testing index clearing and state isolation...")
    store.clear()
    print(f"  After clear(): total_vectors={store.total_vectors}, is_empty={store.is_empty}")
    assert store.total_vectors == 0
    assert store.is_empty is True
    empty_search = store.search(query_vector, top_k=3)
    assert empty_search == []
    print(f"  Safe search on empty index returned: {empty_search}")

    print("\n" + "=" * 75)
    print("All FAISS vector store smoke test assertions passed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
