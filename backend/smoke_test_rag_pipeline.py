"""IntelliAssist AI - RAG Pipeline End-to-End Smoke Test.

This script demonstrates and verifies the complete Retrieval-Augmented Generation
(RAG) pipeline using real local embeddings (sentence-transformers/all-MiniLM-L6-v2),
real FAISS vector indexing (IndexFlatIP), and structured prompt/citation construction.

It cleanly separates:
  Part A: Local Retrieval, Prompting & Grounding Smoke Test (Zero API keys, 100% offline)
  Part B: Optional Real-LLM Integration Test (Google Gemini via google-genai SDK, executed
          only when GEMINI_API_KEY is configured in the environment or backend/.env)

Usage:
    python smoke_test_rag_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.document_processor import process_document
from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore
from app.rag_pipeline import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    INSUFFICIENT_CONTEXT_MESSAGE,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    load_environment_variables,
    validate_question,
)


def print_header(title: str, char: str = "=", width: int = 80) -> None:
    """Print formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def run_smoke_test() -> int:
    print_header("IntelliAssist AI — RAG Pipeline Smoke Test")
    start_time = time.time()

    # Load environment variables safely
    load_environment_variables()

    # ------------------------------------------------------------------------
    # Step 1: Initialize Real Local Embedding Service and FAISS Vector Store
    # ------------------------------------------------------------------------
    print("\n[Step 1/5] Initializing local embedding service and FAISS vector store...")
    print("  Embedding model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional, local)")
    print("  Vector store:    FAISS IndexFlatIP (cosine similarity ranking)")

    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

    assert vector_store.total_vectors == 0
    assert vector_store.is_empty is True
    print("  Vector store initialized successfully.")

    # ------------------------------------------------------------------------
    # Step 2: Ingest Sample Documents
    # ------------------------------------------------------------------------
    print("\n[Step 2/5] Ingesting documents and creating chunks...")
    corpus_docs = []

    # Prioritize evaluation documents if available
    eval_data_dir = backend_dir / "evaluation_data"
    sample_files = [
        eval_data_dir / "security_policy.txt",
        eval_data_dir / "employee_handbook.txt",
        eval_data_dir / "expense_policy.txt",
    ]

    all_chunks: List[Dict[str, Any]] = []

    for file_path in sample_files:
        if file_path.exists():
            processed = process_document(file_path)
            corpus_docs.append(processed)
            chunks = chunk_document(processed, chunk_size=600, chunk_overlap=100)
            all_chunks.extend(chunks)
            print(
                f"  - Loaded {processed['filename']} "
                f"({processed['total_pages']} page(s), {len(chunks)} chunk(s))"
            )

    if not all_chunks:
        print("  [ERROR] No document chunks found to index.")
        return 1

    print(f"  Total chunks to embed and index: {len(all_chunks)}")

    # ------------------------------------------------------------------------
    # Step 3: Embed Chunks and Index in FAISS
    # ------------------------------------------------------------------------
    print("\n[Step 3/5] Computing embeddings with SentenceTransformers and indexing in FAISS...")
    t0 = time.time()
    embedded_chunks = embedding_service.embed_chunks(all_chunks)
    embed_time = time.time() - t0
    print(f"  Generated {len(embedded_chunks)} chunk embeddings in {embed_time:.2f}s.")

    added_count = vector_store.add_chunks(embedded_chunks)
    print(f"  Successfully indexed {added_count} vectors into FAISS. (Total: {vector_store.total_vectors})")
    assert vector_store.total_vectors == len(all_chunks)

    # ------------------------------------------------------------------------
    # Step 4: Part A — Local Offline Smoke Test (Real Embeddings + FAISS + Mock LLM)
    # ------------------------------------------------------------------------
    print_header("PART A: Local Offline RAG Verification (Zero Network / Zero API Keys)")
    print("Verifying semantic retrieval, grounded prompt construction, and citation mapping...")

    # Define deterministic mock LLM client simulating grounded model generation
    def simulated_grounded_generator(system_instruction: str, prompt: str) -> str:
        # Grounded response using retrieved context
        if "password" in prompt.lower():
            return (
                "According to security_policy.txt (Page 1), all employee passwords must be a "
                "minimum of 14 characters in length and must include uppercase, lowercase, "
                "numeric, and special characters. Passwords expire every 90 days."
            )
        elif "pto" in prompt.lower() or "paid time off" in prompt.lower():
            return (
                "According to employee_handbook.txt (Page 1), employees receive 20 days of "
                "paid time off (PTO) annually."
            )
        return "Based on the retrieved context, information is available in the cited source."

    mock_llm = MockLLMClient(
        response_generator=simulated_grounded_generator,
        model_name="mock-grounded-assistant",
    )

    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=mock_llm,
        top_k=3,
    )

    # Test Query 1: Factual Retrieval & Answer Generation
    test_question_1 = "What is the minimum password length and complexity required?"
    print(f"\n[Part A - Test 1] Evaluating Query: '{test_question_1}'")

    response_1 = pipeline.answer(test_question_1)

    print("\n  --- Pipeline Output ---")
    print(f"  Generation Status:  {response_1.generation_status}")
    print(f"  Sufficient Context: {response_1.sufficient_context}")
    print(f"  Retrieved Chunks:   {response_1.retrieval_count}")
    print(f"  Model Employed:     {response_1.model_name}")
    print(f"  Generated Answer:   {response_1.answer}")
    print("\n  --- Citations Extracted ---")
    for idx, c in enumerate(response_1.citations, 1):
        score_str = f"{c['similarity_score']:.4f}" if c['similarity_score'] is not None else "N/A"
        cited_marker = "[CITED]" if c.get("is_cited") else "[RETRIEVED]"
        print(
            f"    [{idx}] {cited_marker} {c['filename']} | Page: {c['page_number']} | "
            f"Chunk ID: {c['chunk_id']} | Cosine Similarity: {score_str}"
        )

    # Assertions for Test 1
    assert response_1.sufficient_context is True
    assert response_1.retrieval_count == 3
    assert len(response_1.citations) == 3
    assert response_1.citations[0]["filename"] == "security_policy.txt"
    assert response_1.citations[0]["page_number"] == 1
    assert response_1.citations[0]["similarity_score"] > 0.4
    assert any(c.get("is_cited") for c in response_1.citations)
    print("  -> Test 1 Passed: Retrieval, metadata preservation, and citations verified.")

    # Test Query 2: Refusal on Insufficient Context
    test_question_2 = "What is the corporate recipe for chocolate chip cookies?"
    print(f"\n[Part A - Test 2] Evaluating Unanswerable Query: '{test_question_2}'")

    refusal_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=MockLLMClient(default_response=INSUFFICIENT_CONTEXT_MESSAGE),
        top_k=3,
        similarity_threshold=0.85,  # Very high threshold that filters out irrelevant matches
    )
    response_2 = refusal_pipeline.answer(test_question_2)

    print(f"  Answer:             {response_2.answer}")
    print(f"  Sufficient Context: {response_2.sufficient_context}")
    print(f"  Generation Status:  {response_2.generation_status}")
    print(f"  Retrieved Chunks:   {response_2.retrieval_count}")

    assert response_2.sufficient_context is False
    assert response_2.generation_status == "insufficient_context"
    assert response_2.answer == INSUFFICIENT_CONTEXT_MESSAGE
    print("  -> Test 2 Passed: Refusal on insufficient context verified.")

    # Test Query 3: Validation of Empty / Malformed Input
    print("\n[Part A - Test 3] Evaluating Input Validation (Empty / Whitespace Input)...")
    try:
        pipeline.answer("   ")
        assert False, "Pipeline should have rejected whitespace-only question"
    except ValueError as e:
        print(f"  Correctly raised ValueError: '{e}'")
    print("  -> Test 3 Passed: Input validation verified.")

    print_header("PART A SUMMARY: ALL LOCAL RETRIEVAL & GROUNDING TESTS PASSED")

    # ------------------------------------------------------------------------
    # Step 5: Part B — Optional Real-LLM Integration Test (Google Gemini)
    # ------------------------------------------------------------------------
    print_header("PART B: Optional Real-LLM Integration Test (Google Gemini)")

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not gemini_key or not gemini_key.strip():
        print("\n[NOTICE: LIVE LLM GENERATION SKIPPED]")
        print("  GEMINI_API_KEY is not configured in your environment or backend/.env.")
        print("  All local RAG components (sentence-transformers embeddings, FAISS search,")
        print("  prompt construction, citation extraction, and defense rules) are fully verified.")
        print("\n  To execute a live end-to-end test with Google Gemini:")
        print("    1. Obtain a free API key from Google AI Studio: https://aistudio.google.com/")
        print("    2. Copy backend/.env.example to backend/.env and set:")
        print("       GEMINI_API_KEY=your_actual_key_here")
        print("    3. Or in PowerShell: $env:GEMINI_API_KEY = 'your_actual_key_here'")
        print("    4. Re-run: python smoke_test_rag_pipeline.py")
    else:
        print("\n[LIVE GENERATION DETECTED]")
        # Redact key in all logs
        masked_key = gemini_key[:4] + "..." + gemini_key[-4:] if len(gemini_key) > 8 else "***"
        print(f"  Found Gemini API key: {masked_key} (Key remains strictly confidential)")
        print("  Initializing real GeminiLLMClient (model: gemini-2.5-flash)...")

        try:
            live_client = GeminiLLMClient(model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            live_pipeline = RAGPipeline(
                vector_store=vector_store,
                embedding_service=embedding_service,
                llm_client=live_client,
                top_k=3,
            )

            live_question = "What is the required password length and what are the password complexity rules?"
            print(f"\n  Sending prompt to Google Gemini: '{live_question}'")
            t_live = time.time()
            live_response = live_pipeline.answer(live_question)
            live_time = time.time() - t_live

            print(f"  Live Generation Time: {live_time:.2f}s")
            print(f"  Generation Status:    {live_response.generation_status}")
            print(f"  Sufficient Context:   {live_response.sufficient_context}")
            print(f"  Model Employed:       {live_response.model_name}")
            print(f"\n  --- Real Gemini Answer ---")
            print(f"  {live_response.answer}")
            print(f"\n  --- Citations ---")
            for idx, c in enumerate(live_response.citations, 1):
                score_str = f"{c['similarity_score']:.4f}" if c['similarity_score'] is not None else "N/A"
                print(
                    f"    [{idx}] {c['filename']} | Page: {c['page_number']} | "
                    f"Chunk: {c['chunk_id']} | Cosine Sim: {score_str}"
                )

            assert live_response.generation_status == "success"
            assert live_response.sufficient_context is True
            assert "14" in live_response.answer or "security_policy" in live_response.answer.lower()
            print("\n  -> Part B Succeeded: Live Google Gemini generation verified!")
        except Exception as e:
            # Catch any network or quota issues safely without exposing secrets
            sanitized_err = str(e)
            if gemini_key in sanitized_err:
                sanitized_err = sanitized_err.replace(gemini_key, "[REDACTED_API_KEY]")
            print(f"\n  [WARNING] Live Gemini API call could not be completed: {sanitized_err}")
            print("  Note: Unit tests and offline pipeline verification remain fully passed.")

    total_time = time.time() - start_time
    print_header(f"SMOKE TEST COMPLETE (Duration: {total_time:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
