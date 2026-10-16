"""IntelliAssist AI - Executable Retrieval Evaluation Script.

This script executes an end-to-end semantic retrieval evaluation:
  1. Ingests evaluation documents from backend/evaluation_data/
  2. Chunks documents and embeds chunks using sentence-transformers/all-MiniLM-L6-v2
  3. Indexes chunks into an in-memory FAISS IndexFlatIP vector store
  4. Runs natural-language evaluation queries from backend/evaluation_data/evaluation_queries.json
  5. Computes IR metrics: Hit Rate@K, Recall@K, and Mean Reciprocal Rank (MRR)
  6. Displays formatted query rankings and aggregate benchmark results
  7. Exports results to backend/evaluation_results/ as JSON and CSV

Usage:
    python evaluate_retrieval.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.retrieval_evaluator import (
    RetrievalEvaluator,
    load_queries_from_json,
)
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore


def print_banner(title: str, char: str = "=", width: int = 80) -> None:
    """Print a centered header banner."""
    print(char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def main() -> int:
    print_banner("IntelliAssist AI — Semantic Retrieval Quality Evaluation")
    start_time = time.time()

    # Paths setup
    data_dir = backend_dir / "evaluation_data"
    queries_file = data_dir / "evaluation_queries.json"
    results_dir = backend_dir / "evaluation_results"

    if not data_dir.exists():
        print(f"\n[ERROR] Evaluation data directory not found: {data_dir}")
        print("Please ensure sample evaluation documents exist in backend/evaluation_data/.")
        return 1

    if not queries_file.exists():
        print(f"\n[ERROR] Evaluation queries file not found: {queries_file}")
        print("Please ensure evaluation_queries.json exists in backend/evaluation_data/.")
        return 1

    print("\n[Configuration]")
    print(f"  Backend Directory:    {backend_dir}")
    print(f"  Corpus Directory:     {data_dir}")
    print(f"  Queries File:         {queries_file}")
    print(f"  Results Directory:    {results_dir}")
    print(f"  Embedding Model:      sentence-transformers/all-MiniLM-L6-v2 (384-d, local)")
    print(f"  Vector Store Index:   FAISS IndexFlatIP (cosine similarity)")
    print(f"  Evaluation Cutoffs:   K = [1, 3, 5]")

    # 1. Initialize Evaluator & Embedding Service
    print("\n[Step 1/5] Initializing Retrieval Evaluator and Model...")
    print("  Note: If the model weights are not in local cache, an initial download")
    print("  from Hugging Face Hub (~90MB) will occur automatically.")

    try:
        embedding_service = EmbeddingService()
        vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)
        evaluator = RetrievalEvaluator(
            embedding_service=embedding_service,
            vector_store=vector_store,
            k_values=(1, 3, 5),
        )
    except Exception as e:
        print(f"\n[FATAL] Failed to initialize evaluator: {e}")
        return 1

    # 2. Ingest Corpus Documents
    print("\n[Step 2/5] Ingesting evaluation corpus documents...")
    try:
        documents = evaluator.load_documents_from_directory(data_dir)
        print(f"  Loaded {len(documents)} document(s):")
        for doc in documents:
            print(
                f"    - {doc['filename']:<26} | Pages: {doc['total_pages']} | "
                f"Characters: {doc['total_characters']:,}"
            )
    except Exception as e:
        print(f"\n[FATAL] Document loading failed: {e}")
        return 1

    # 3. Chunk, Embed, and Index
    print("\n[Step 3/5] Chunking text and generating local embeddings...")
    try:
        total_chunks = evaluator.build_evaluation_index(
            documents, chunk_size=800, chunk_overlap=150
        )
        print(f"  Successfully created and indexed {total_chunks} chunk vectors in FAISS.")
        print(f"  Vector store total vectors: {vector_store.total_vectors}")
    except RuntimeError as e:
        print(f"\n[FATAL] Model loading or embedding failed: {e}")
        print("Please verify internet connectivity for the initial model download.")
        return 1
    except Exception as e:
        print(f"\n[FATAL] Indexing failed: {e}")
        return 1

    # 4. Load Queries and Evaluate
    print("\n[Step 4/5] Loading queries and evaluating retrieval...")
    try:
        queries = load_queries_from_json(queries_file)
        print(f"  Loaded {len(queries)} evaluation queries.")
    except Exception as e:
        print(f"\n[FATAL] Failed to load queries: {e}")
        return 1

    try:
        summary = evaluator.evaluate(queries, top_k=5)
    except Exception as e:
        print(f"\n[FATAL] Evaluation execution failed: {e}")
        return 1

    # 5. Display Detailed Query Results
    print_banner("Per-Query Retrieval Results (Top 3 Chunks Evaluated)", char="-")

    for idx, qr in enumerate(summary.query_results, start=1):
        status_symbol = "[HIT]" if qr.hit_at_k.get(1, 0) == 1 else (
            "[PARTIAL]" if qr.hit_at_k.get(3, 0) == 1 else "[MISS]"
        )
        first_rank_str = f"Rank #{qr.first_relevant_rank}" if qr.first_relevant_rank else "Not in top 5"

        print(f"\nQuery {idx:02d} ({qr.query_id}) {status_symbol}")
        print(f"  Question:      \"{qr.question}\"")
        print(f"  Expected Doc:  {qr.expected_doc}")
        print(f"  Target Fact:   {qr.fact_description}")
        print(f"  1st Rel Rank:  {first_rank_str} (RR = {qr.reciprocal_rank:.4f})")
        print(f"  Hit@K:         K=1: {qr.hit_at_k.get(1, 0)} | K=3: {qr.hit_at_k.get(3, 0)} | K=5: {qr.hit_at_k.get(5, 0)}")
        print(f"  Recall@K:      K=1: {qr.recall_at_k.get(1, 0.0):.2f} | K=3: {qr.recall_at_k.get(3, 0.0):.2f} | K=5: {qr.recall_at_k.get(5, 0.0):.2f}")
        print("  Top Retrieved Chunks:")

        for item in qr.retrieved_chunks[:3]:
            rel_tag = "[RELEVANT]" if item["is_relevant"] else "[NON-RELEVANT]"
            page_info = f"p.{item['metadata'].get('page_number', '?')}"
            excerpt = item["text"][:90].replace("\n", " ").strip()
            print(
                f"    #{item['rank']} (Score: {item['similarity_score']:.4f}) "
                f"{item['chunk_id']:<38} {page_info:<4} {rel_tag:<14} \"{excerpt}...\""
            )

    # 6. Display Aggregate Metrics
    print_banner("Retrieval Evaluation Benchmark Summary", char="=")
    print(f"\n  Corpus Statistics:")
    print(f"    - Total Documents:      {len(documents)}")
    print(f"    - Total Chunks:         {summary.total_indexed_chunks}")
    print(f"    - Evaluation Queries:   {summary.total_queries}")
    print(f"    - Evaluated Top K:      5")

    print(f"\n  Aggregate Retrieval Metrics:")
    print("  +----------------------+-----------------+----------------------------------------+")
    print("  | Metric               | Measured Value  | Interpretation                         |")
    print("  +----------------------+-----------------+----------------------------------------+")
    for k in summary.k_values:
        hit_val = summary.hit_rate_at_k.get(k, 0.0)
        print(f"  | Hit Rate@{k:<11} | {hit_val:>13.4f}   | {hit_val * 100:>5.1f}% queries retrieved >= 1 chunk  |")
    print("  +----------------------+-----------------+----------------------------------------+")
    for k in summary.k_values:
        rec_val = summary.recall_at_k.get(k, 0.0)
        print(f"  | Recall@{k:<13} | {rec_val:>13.4f}   | {rec_val * 100:>5.1f}% of target facts retrieved    |")
    print("  +----------------------+-----------------+----------------------------------------+")
    print(f"  | Mean Recip. Rank MRR | {summary.mrr:>13.4f}   | Average (1 / First Relevant Rank)      |")
    print("  +----------------------+-----------------+----------------------------------------+")

    print("\n  Metric Equivalence Note:")
    print("    When an evaluation query has exactly one known relevant chunk, Recall@K")
    print("    and Hit Rate@K are mathematically identical for that query (both evaluate")
    print("    to 1.0 when that single chunk is in top K, and 0.0 otherwise).")

    # 7. Export Results
    print("\n[Step 5/5] Exporting reproducible evaluation reports...")
    try:
        json_path = results_dir / "retrieval_evaluation_report.json"
        csv_path = results_dir / "retrieval_evaluation_report.csv"

        evaluator.export_results_json(summary, json_path)
        evaluator.export_results_csv(summary, csv_path)

        print(f"  - JSON report saved: {json_path}")
        print(f"  - CSV report saved:  {csv_path}")
    except Exception as e:
        print(f"  [WARNING] Failed to export results to disk: {e}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation completed successfully in {elapsed:.2f} seconds.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
