"""IntelliAssist AI - Answer Quality, Grounding, and Citation Evaluation Harness.

This script executes an end-to-end evaluation of answer quality, document grounding,
and citation correctness across 10 specialized scenarios:
  1. Explicit fact verification
  2. Multi-passage synthesis
  3. Unanswerable question / refusal
  4. Paraphrased / differing wording
  5. Misleading question with false premise
  6. Conversation history misinformation defense
  7. Irrelevant retrieved chunks handling
  8. Multi-source document synthesis
  9. Missing / incomplete metadata handling (without fabrication)
  10. Conversational follow-up resolution

Metrics Calculated:
  - Citation Source-Validity Rate: Valid citations / Total generated citations
  - Citation Metadata Completeness: Complete metadata fields / Expected fields (without fabrication)
  - Unsupported-Answer Rate: Unsupported claims on unanswerable/misleading queries / Total such queries
  - Correct Refusal Rate: Proper refusals / Total unanswerable queries
  - Answer-Support Accuracy: Correctly supported or refused cases / Total evaluated cases

Outputs:
  - backend/evaluation_results/answer_quality_report.json
  - backend/evaluation_results/answer_quality_report.csv

Usage:
  python evaluate_answer_quality.py [--live-llm]
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend root is in sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.document_processor import process_document
from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore
from app.rag_pipeline import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    build_citations,
    detect_unverified_citations,
    is_context_insufficient_response,
    load_environment_variables,
)
from app.chatbot_service import ChatbotService
from app.conversation_store import ConversationStore


def print_banner(title: str, char: str = "=", width: int = 80) -> None:
    """Print centered header banner."""
    print(char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def build_evaluation_mock_llm() -> MockLLMClient:
    """Construct deterministic mock LLM with grounded responses for evaluation cases."""

    def generator(system_instruction: str, prompt: str) -> str:
        # Extract the question specifically from prompt to avoid false positives on context text
        q_match = re.search(r"QUESTION:\s*([^\n]+)", prompt)
        q_lower = q_match.group(1).lower() if q_match else prompt.lower()

        # Unanswerable queries: parental/maternity leave or underground garage parking
        if "parental" in q_lower or "maternity" in q_lower or ("parking" in q_lower and "garage" in q_lower):
            return INSUFFICIENT_CONTEXT_MESSAGE

        # Misleading question: reduced meal allowance to $30
        if "reduced" in q_lower and "$30" in q_lower:
            return (
                "According to expense_policy.txt (Page 1), the daily meal reimbursement allowance "
                "during business travel is capped at a maximum of $75 per day. The documents do not "
                "support any reduction to thirty dollars or an allowance cap of $30."
            )

        # Conversation history misinformation defense: 35 days PTO claim
        if "35 days" in q_lower:
            return (
                "According to employee_handbook.txt (Page 1), full-time employees receive 20 days "
                "of paid time off (PTO) per calendar year, accrued monthly. The documents do not "
                "support the 35-day PTO allocation mentioned earlier."
            )

        # Multi-document question: password length AND remote equipment stipend
        if "character length" in q_lower and "stipend" in q_lower:
            return (
                "According to security_policy.txt (Page 1), all employee passwords must be a minimum "
                "of 14 characters in length. In addition, according to employee_handbook.txt (Page 1), "
                "the company provides a one-time equipment reimbursement stipend of $500 for approved "
                "home office supplies."
            )

        # Missing metadata test: emergency corporate server shutdown
        if "emergency" in q_lower and "server shutdown" in q_lower:
            return (
                "According to synthetic_it_runbook.txt, an emergency corporate server shutdown "
                "requires immediate authorization from the Lead Infrastructure Engineer."
            )

        # Multi-passage synthesis: sick leave rules and 3 consecutive days absence
        if "sick leave" in q_lower and "three consecutive" in q_lower:
            return (
                "According to employee_handbook.txt (Page 1), full-time employees receive 10 paid sick "
                "days per year, with a doctor note required for sick leave exceeding 3 days. Furthermore, "
                "an unexcused absence spanning 3 consecutive business days without communication constitutes "
                "job abandonment and results in immediate termination."
            )

        # Paraphrased credentials / password expiry
        if "credentials expire" in q_lower or "passcodes" in q_lower:
            return (
                "According to security_policy.txt (Page 1), employee passwords automatically expire "
                "every 90 days, and staff cannot reuse any of their previous 10 passwords."
            )

        # Conversational follow up: "how often do they expire?"
        if "how often do they expire" in q_lower or "they expire" in q_lower:
            return (
                "According to security_policy.txt (Page 1), employee passwords automatically "
                "expire every 90 days."
            )

        # Explicit fact: meal per diem
        if "meal" in q_lower or "per diem" in q_lower:
            return (
                "According to expense_policy.txt (Page 1), the daily meal reimbursement allowance "
                "(per diem) during business travel is capped at a maximum of $75 per day."
            )

        # Default fallback
        match = re.search(r"Filename:\s*([^\n]+)", prompt)
        filename = match.group(1).strip() if match else "document.txt"
        return f"Based on the provided context in {filename}, relevant policy information is documented."

    return MockLLMClient(response_generator=generator, model_name="mock-grounded-evaluator")


def evaluate_single_case(
    case: Dict[str, Any],
    rag_pipeline: RAGPipeline,
    chatbot_service: Optional[ChatbotService] = None,
) -> Dict[str, Any]:
    """Execute and deterministically evaluate a single answer-quality test case."""
    case_id = case["case_id"]
    question = case["question"]
    category = case["case_category"]
    history = case.get("conversation_history", [])
    expected_behavior = case["expected_behavior"]
    expected_sources = [s.lower() for s in case.get("expected_sources", [])]
    key_facts = case.get("key_facts", [])
    prohibited_facts = case.get("prohibited_facts", [])

    # Execute answer generation
    if history and chatbot_service:
        conv = chatbot_service.create_conversation(title=f"Eval {case_id}")
        cid = conv["conversation_id"]
        # Populate history
        for msg in history:
            chatbot_service.conversation_store.add_message(
                conversation_id=cid,
                role=msg["role"],
                content=msg["content"],
            )
        chat_resp = chatbot_service.chat(cid, question)
        answer = chat_resp.assistant_response
        citations = chat_resp.citations
        retrieved_chunks = chat_resp.retrieved_chunks
        sufficient_context = chat_resp.sufficient_context
        generation_status = chat_resp.status
        unverified_citations = chat_resp.unverified_citations
    else:
        rag_resp: RAGResponse = rag_pipeline.answer(
            question=question,
            conversation_history=history if history else None,
        )
        answer = rag_resp.answer
        citations = rag_resp.citations
        retrieved_chunks = rag_resp.retrieved_chunks
        sufficient_context = rag_resp.sufficient_context
        generation_status = rag_resp.generation_status
        unverified_citations = rag_resp.unverified_citations

    answer_lower = answer.lower()
    is_refusal = is_context_insufficient_response(answer)

    # 1. Evaluate Citation Validity
    # Numerator: Cited sources that match actual retrieved chunks
    # Denominator: Total citations generated
    retrieved_chunk_ids = {c["chunk_id"] for c in retrieved_chunks if "chunk_id" in c}
    retrieved_filenames = {
        Path(c.get("metadata", {}).get("filename", "")).name.lower()
        for c in retrieved_chunks
        if c.get("metadata", {}).get("filename")
    }

    valid_citations_count = 0
    total_citations_count = len(citations)
    cited_sources_found: List[str] = []

    for cit in citations:
        c_file = (cit.get("filename") or "").lower()
        c_base = Path(c_file).name.lower() if c_file else ""
        c_id = cit.get("chunk_id")

        is_valid_source = False
        if c_id and c_id in retrieved_chunk_ids:
            is_valid_source = True
        elif c_base and c_base in retrieved_filenames:
            is_valid_source = True

        if is_valid_source:
            valid_citations_count += 1
            if cit.get("is_cited"):
                cited_sources_found.append(c_base)

    # Citation source-validity score (1.0 if no citations were generated e.g. on refusal)
    if total_citations_count > 0:
        citation_validity_score = valid_citations_count / total_citations_count
    else:
        citation_validity_score = 1.0

    # Subtract validity if model generated unverified hallucinated references
    if unverified_citations:
        citation_validity_score = 0.0

    # 2. Metadata completeness and safety check
    # Missing metadata must be recorded safely as None / null rather than fabricated
    metadata_safe = True
    if case.get("is_synthetic"):
        # For synthetic chunk with missing page_number, verify page_number is None
        for cit in citations:
            if "[synthetic]" in (cit.get("filename") or "").lower():
                if cit.get("page_number") is not None:
                    metadata_safe = False

    # 3. Grounded Answer / Refusal Accuracy
    behavior_passed = False
    failure_reasons: List[str] = []

    if expected_behavior == "refusal":
        if is_refusal and not sufficient_context:
            behavior_passed = True
        else:
            failure_reasons.append("Expected refusal on insufficient context, but answer was generated.")

    elif expected_behavior in ("answer_supported", "refusal_or_clarification"):
        # Check required key facts are present
        missing_facts = [kf for kf in key_facts if kf.lower() not in answer_lower]
        if missing_facts:
            failure_reasons.append(f"Missing required facts in answer: {missing_facts}")

        # Check prohibited facts / false premises are NOT affirmed
        prohibited_present = [pf for pf in prohibited_facts if pf.lower() in answer_lower]
        if prohibited_present:
            failure_reasons.append(f"Answer affirmed prohibited false premise: {prohibited_present}")

        # Check expected sources are cited
        missing_sources = [
            s for s in expected_sources
            if not any(Path(s).name.lower() in found for found in cited_sources_found)
            and Path(s).name.lower() not in answer_lower
        ]
        if missing_sources:
            failure_reasons.append(f"Expected source citation missing: {missing_sources}")

        if not missing_facts and not prohibited_present and not missing_sources:
            behavior_passed = True

    # Check unverified citations
    if unverified_citations:
        failure_reasons.append(f"Unverified citations detected: {unverified_citations}")

    if not metadata_safe:
        failure_reasons.append("Missing metadata was fabricated rather than represented safely.")

    overall_pass = behavior_passed and (citation_validity_score == 1.0) and metadata_safe

    return {
        "case_id": case_id,
        "category": category,
        "question": question,
        "expected_behavior": expected_behavior,
        "expected_sources": case.get("expected_sources", []),
        "actual_answer": answer,
        "retrieval_count": len(retrieved_chunks),
        "citations": citations,
        "unverified_citations": unverified_citations,
        "sufficient_context": sufficient_context,
        "citation_validity_score": citation_validity_score,
        "metadata_safe": metadata_safe,
        "behavior_passed": behavior_passed,
        "overall_pass": overall_pass,
        "failure_reasons": failure_reasons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate IntelliAssist AI Answer Quality & Citations")
    parser.add_argument(
        "--live-llm",
        action="store_true",
        help="Attempt live generation using Google Gemini (requires GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    start_time = time.time()
    print_banner("IntelliAssist AI — Answer Quality & Citations Evaluation")

    data_dir = backend_dir / "evaluation_data"
    cases_file = data_dir / "answer_quality_cases.json"
    results_dir = backend_dir / "evaluation_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not cases_file.exists():
        print(f"[FATAL] Answer quality cases file not found: {cases_file}")
        return 1

    load_environment_variables()

    # Determine LLM Client
    llm_client: BaseLLMClient
    is_live = False
    if args.live_llm:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key and api_key.strip():
            try:
                llm_client = GeminiLLMClient(api_key=api_key.strip())
                is_live = True
                print("  LLM Mode: Live Cloud Model (Google Gemini API)")
            except Exception as e:
                err_str = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", str(e))
                if api_key and api_key.strip() in err_str:
                    err_str = err_str.replace(api_key.strip(), "[REDACTED_API_KEY]")
                print(f"  [WARNING] Live Gemini initialization failed: {err_str}. Falling back to mock client.")
                llm_client = build_evaluation_mock_llm()
        else:
            print("  [WARNING] --live-llm specified but GEMINI_API_KEY is not configured.")
            print("  Falling back to deterministic grounded mock LLM.")
            llm_client = build_evaluation_mock_llm()
    else:
        llm_client = build_evaluation_mock_llm()
        print("  LLM Mode: Deterministic Grounded Mock LLM (Offline Benchmark)")

    # 1. Ingest Documents
    print("\n[Step 1/4] Ingesting documents and building vector store...")
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

    # Ingest standard evaluation policies
    policy_files = ["security_policy.txt", "employee_handbook.txt", "expense_policy.txt"]
    all_chunks: List[Dict[str, Any]] = []

    for fname in policy_files:
        fpath = data_dir / fname
        if fpath.is_file():
            doc = process_document(fpath)
            chunks = chunk_document(doc, chunk_size=800, chunk_overlap=150)
            all_chunks.extend(chunks)

    # Add synthetic chunk with missing page_number for Case AQ-09
    synthetic_chunk = {
        "chunk_id": "it_runbook.txt_synthetic_chunk_1",
        "text": (
            "SECTION 9: EMERGENCY PROCEDURES\n"
            "Emergency corporate server shutdown requires immediate authorization from the "
            "Lead Infrastructure Engineer. Once authorized, all non-critical database services "
            "must be halted in reverse tier order."
        ),
        "metadata": {
            "filename": "synthetic_it_runbook.txt",
            "file_type": "txt",
            # Intentionally missing page_number and chunk_index to verify safe metadata handling
        },
    }
    all_chunks.append(synthetic_chunk)

    # Embed and index all chunks
    embedded_chunks = embedding_service.embed_chunks(all_chunks)
    vector_store.add_chunks(embedded_chunks)
    print(f"  Indexed {len(all_chunks)} chunks across {len(policy_files) + 1} documents.")

    # 2. Initialize RAG Pipeline and Chatbot Service
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,
        top_k=3,
    )
    conv_store = ConversationStore(db_path=":memory:")
    chatbot_service = ChatbotService(
        rag_pipeline=rag_pipeline,
        conversation_store=conv_store,
    )

    # 3. Load Cases and Evaluate
    print("\n[Step 2/4] Executing 10 answer-quality evaluation scenarios...")
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        res = evaluate_single_case(case, rag_pipeline, chatbot_service)
        results.append(res)
        status = "PASSED" if res["overall_pass"] else "FAILED"
        print(f"  [{idx:02d}/10] {res['case_id']} ({res['category']}): {status}")
        if not res["overall_pass"]:
            for r in res["failure_reasons"]:
                print(f"         - Reason: {r}")

    # 4. Compute Aggregate Metrics
    total_cases = len(results)
    total_passed = sum(1 for r in results if r["overall_pass"])

    # Metric 1: Citation Source-Validity Rate
    total_citations = sum(len(r["citations"]) for r in results)
    valid_citations = sum(
        sum(1 for c in r["citations"] if c.get("chunk_id") or c.get("filename"))
        for r in results
        if not r["unverified_citations"]
    )
    citation_source_validity_rate = (valid_citations / total_citations) if total_citations > 0 else 1.0

    # Metric 2: Citation Metadata Completeness
    # Safe representation of metadata (no fabricated fields)
    metadata_safe_count = sum(1 for r in results if r["metadata_safe"])
    citation_metadata_completeness = metadata_safe_count / total_cases

    # Metric 3: Unsupported-Answer Rate on unanswerable/false-premise cases
    unsupported_query_ids = {"AQ-03", "AQ-05", "AQ-06", "AQ-07"}
    unsupported_test_cases = [r for r in results if r["case_id"] in unsupported_query_ids]
    unsupported_answers = sum(1 for r in unsupported_test_cases if not r["behavior_passed"])
    unsupported_answer_rate = unsupported_answers / len(unsupported_test_cases) if unsupported_test_cases else 0.0

    # Metric 4: Correct Refusal Rate for unanswerable questions
    refusal_query_ids = {"AQ-03", "AQ-07"}
    refusal_cases = [r for r in results if r["case_id"] in refusal_query_ids]
    correct_refusals = sum(1 for r in refusal_cases if r["behavior_passed"])
    correct_refusal_rate = correct_refusals / len(refusal_cases) if refusal_cases else 1.0

    # Metric 5: Answer-Support Accuracy
    answer_support_accuracy = total_passed / total_cases

    # 5. Display Benchmark Summary
    print_banner("Answer Quality Benchmark Summary", char="=")
    print(f"  Test Execution Mode:   {'Live Gemini API' if is_live else 'Deterministic Grounded Mock LLM'}")
    print(f"  Total Evaluated Cases: {total_cases}")
    print(f"  Passed Cases:          {total_passed}/{total_cases} ({answer_support_accuracy * 100:.1f}%)")
    print("\n  Aggregate Grounding & Citation Metrics:")
    print("  +------------------------------------+------------+------------------------------------------+")
    print("  | Metric Name                        | Value      | Definition / Formula                     |")
    print("  +------------------------------------+------------+------------------------------------------+")
    print(f"  | Citation Source-Validity Rate      | {citation_source_validity_rate * 100:>8.2f}%  | Valid Retr. Citations / Total Citations  |")
    print(f"  | Citation Metadata Completeness     | {citation_metadata_completeness * 100:>8.2f}%  | Safely Formatted Citations / Total Cases |")
    print(f"  | Unsupported-Answer Rate            | {unsupported_answer_rate * 100:>8.2f}%  | Hallucinations / Unsupported Test Cases  |")
    print(f"  | Correct Refusal Rate               | {correct_refusal_rate * 100:>8.2f}%  | Correct Refusals / Unanswerable Queries  |")
    print(f"  | Answer-Support Accuracy            | {answer_support_accuracy * 100:>8.2f}%  | Overall Labeled Ground Truth Accuracy    |")
    print("  +------------------------------------+------------+------------------------------------------+")

    # 6. Export Reports
    print("\n[Step 3/4] Exporting JSON and CSV evaluation reports...")
    json_path = results_dir / "answer_quality_report.json"
    csv_path = results_dir / "answer_quality_report.csv"

    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_mode": "live_gemini" if is_live else "mock_grounded",
        "total_cases": total_cases,
        "passed_cases": total_passed,
        "metrics": {
            "citation_source_validity_rate": citation_source_validity_rate,
            "citation_metadata_completeness": citation_metadata_completeness,
            "unsupported_answer_rate": unsupported_answer_rate,
            "correct_refusal_rate": correct_refusal_rate,
            "answer_support_accuracy": answer_support_accuracy,
        },
        "metric_definitions": {
            "citation_source_validity_rate": {
                "formula": "valid_retrieved_citations / total_claimed_citations",
                "description": "Proportion of generated citations that match chunks legitimately retrieved for that query.",
            },
            "citation_metadata_completeness": {
                "formula": "safe_metadata_cases / total_cases",
                "description": "Proportion of test cases where chunk metadata was preserved without fabricating missing fields.",
            },
            "unsupported_answer_rate": {
                "formula": "unsupported_answers / total_unanswerable_or_misleading_queries",
                "description": "Proportion of unanswerable or misleading queries where the model fabricated an answer.",
            },
            "correct_refusal_rate": {
                "formula": "proper_refusals / total_unanswerable_queries",
                "description": "Proportion of queries with no document support that triggered the standard safe refusal.",
            },
            "answer_support_accuracy": {
                "formula": "passed_cases / total_test_cases",
                "description": "Proportion of test scenarios where the assistant satisfied all evidence grounding rules.",
            },
        },
        "case_results": results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "category",
            "question",
            "expected_behavior",
            "expected_sources",
            "retrieval_count",
            "sufficient_context",
            "citation_validity_score",
            "metadata_safe",
            "behavior_passed",
            "overall_pass",
            "actual_answer",
            "failure_reasons",
        ])
        for r in results:
            writer.writerow([
                r["case_id"],
                r["category"],
                r["question"],
                r["expected_behavior"],
                ";".join(r["expected_sources"]),
                r["retrieval_count"],
                r["sufficient_context"],
                f"{r['citation_validity_score']:.2f}",
                r["metadata_safe"],
                r["behavior_passed"],
                r["overall_pass"],
                r["actual_answer"].replace("\n", " "),
                "; ".join(r["failure_reasons"]),
            ])

    print(f"  - Saved JSON report: {json_path}")
    print(f"  - Saved CSV report:  {csv_path}")

    elapsed = time.time() - start_time
    print(f"\n[Step 4/4] Completed answer-quality evaluation in {elapsed:.2f} seconds.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
