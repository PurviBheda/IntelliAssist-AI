"""IntelliAssist AI - Real Embedding Smoke Test (Manual).

This script tests the actual local SentenceTransformer model ('all-MiniLM-L6-v2')
end-to-end with real vector generation.

Usage:
    python smoke_test_embedding.py

Note:
    First-time execution requires an active internet connection to download
    the model weights (~90MB). Subsequent runs use the local HuggingFace cache.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService, embed_chunks


def run_smoke_test() -> None:
    print("=" * 70)
    print("IntelliAssist AI — Real Embedding Smoke Test")
    print("=" * 70)

    # 1. Prepare sample processed document
    sample_doc = {
        "filename": "sample_handbook.pdf",
        "file_type": "pdf",
        "pages": [
            {
                "page_number": 1,
                "text": (
                    "IntelliAssist AI is an intelligent document assistant designed to "
                    "help users search, extract, and understand structured documentation. "
                    "It combines robust document processing with semantic vector search."
                ),
            },
            {
                "page_number": 2,
                "text": (
                    "Security and privacy are core architectural tenets of the platform. "
                    "All embeddings are computed locally using open-source transformer models, "
                    "ensuring sensitive enterprise data never leaves the host environment."
                ),
            },
        ],
        "total_pages": 2,
        "total_characters": 372,
    }

    print("\n[Step 1] Document text chunking...")
    chunks = chunk_document(sample_doc, chunk_size=200, chunk_overlap=40)
    print(f"Generated {len(chunks)} chunks from 2 pages.")
    for c in chunks:
        print(f"  - {c['chunk_id']} ({len(c['text'])} chars)")

    print("\n[Step 2] Initializing EmbeddingService (lazy loading)...")
    service = EmbeddingService()
    print(f"  Model loaded before inference: {service.is_model_loaded}")

    print("\n[Step 3] Generating real text embeddings with all-MiniLM-L6-v2...")
    print("  (Note: First-time download may take 10-30s depending on network speed)...")
    try:
        embedded_chunks = service.embed_chunks(chunks)
    except RuntimeError as e:
        print(f"\n[ERROR] Model loading failed: {e}")
        print("Please verify your internet connection for the initial model download.")
        sys.exit(1)

    print(f"  Model loaded after inference: {service.is_model_loaded}")
    print(f"  Successfully embedded {len(embedded_chunks)} chunks.")

    print("\n[Step 4] Validating vector outputs:")
    for item in embedded_chunks:
        vec = item["embedding"]
        norm = float(np.linalg.norm(vec))
        print(
            f"  - Chunk ID: {item['chunk_id']}\n"
            f"    Shape: {vec.shape} | Dtype: {vec.dtype} | L2 Norm: {norm:.4f}\n"
            f"    Metadata: {item['metadata']}"
        )
        assert vec.shape == (384,), f"Expected shape (384,), got {vec.shape}"
        assert np.isclose(norm, 1.0, atol=1e-3), f"Expected unit norm ~1.0, got {norm}"

    # Semantic similarity demo (dot product of normalized vectors = cosine similarity)
    if len(embedded_chunks) >= 2:
        v1 = embedded_chunks[0]["embedding"]
        v2 = embedded_chunks[1]["embedding"]
        cos_sim = float(np.dot(v1, v2))
        print(f"\n[Step 5] Cosine similarity between Chunk 1 & Chunk 2: {cos_sim:.4f}")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED: Real text embeddings generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
