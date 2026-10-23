"""IntelliAssist AI - Live Gemini API Integration Verification.

This script executes live end-to-end verification against the Google Gemini API:
1. Verifies API key resolution from environment or backend/.env.
2. Tests direct LLM client generation (GeminiLLMClient via official google-genai SDK).
3. Tests Grounded RAG Document Question-Answering using real vector retrieval.
4. Tests Citation extraction and source verification.
5. Tests Hallucination defense and safe refusal on unanswerable questions.
6. Tests Multi-turn conversational question-answering with SQLite history.
7. Tests Abstractive Document Summarization via Gemini.

Security & Confidentiality:
- NEVER prints, hardcodes, logs, or exposes API keys.
- Strictly adheres to the rule: Do NOT claim live API functionality is verified
  unless an actual API request succeeds against the live Gemini endpoint.

Usage:
    python backend/verify_live_gemini.py
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.document_processor import process_document
from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore
from app.conversation_store import ConversationStore
from app.chatbot_service import ChatbotService, ChatResponse
from app.summarizer import TextSummarizer, SummaryResult
from app.sentiment_analyzer import SentimentAnalyzer
from app.intent_detector import IntentDetector
from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    INSUFFICIENT_CONTEXT_MESSAGE,
    GeminiLLMClient,
    RAGPipeline,
    RAGResponse,
    load_environment_variables,
)


def print_banner(title: str, char: str = "=", width: int = 80) -> None:
    """Print centered banner."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def sanitize_message(msg: Any, secret: Optional[str] = None) -> str:
    """Ensure no secret tokens appear in output."""
    s = str(msg)
    if secret and secret in s:
        s = s.replace(secret, "[REDACTED_API_KEY]")
    s = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", s)
    s = re.sub(r"(key=)[^&\s]+", r"\1[REDACTED_API_KEY]", s)
    return s


def classify_failure_component(error_str: str, status_str: Optional[str] = None) -> str:
    """Identify the originating pipeline component for a test failure."""
    text = (error_str or "").lower()
    st = (status_str or "").lower()

    if "429" in text or "rate limit" in text or "quota" in text or "resource_exhausted" in text:
        return "Gemini API (Rate Limit / Quota Exceeded)"
    if "503" in text or "500" in text or "overloaded" in text or "unavailable" in text:
        return "Gemini API (Service Overloaded / Temporarily Unavailable)"
    if "timeout" in text or "timed out" in text:
        return "Gemini API (Request Timeout)"
    if "gemini" in text or "interactions" in text or "google.genai" in text or "empty or blocked" in text or "language model" in text:
        return "Gemini API (LLM Generation)"
    if "embedding" in text or "sentence_transformer" in text:
        return "Embedding Service"
    if "faiss" in text or "vector" in text or "retrieval" in text:
        return "Vector Store / Retrieval (FAISS)"
    if "sqlite" in text or "database" in text or "conversation" in text:
        return "Conversation Store (SQLite)"
    if "assertion" in text or "expected" in text:
        return "Assertion / Output Verification"
    if st == "error":
        return "Pipeline Error"
    return "Application Pipeline"


def verify_live_gemini() -> int:
    """Run full live verification suite against Google Gemini API."""
    print_banner("IntelliAssist AI — Live Google Gemini Verification")
    start_total_time = time.time()

    # Step 0: Environment Loading & Key Discovery
    load_environment_variables()
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    placeholder_keys = {
        "your_gemini_api_key_here",
        "your_actual_key_here",
        "your_api_key_here",
        "TODO",
        "CHANGEME",
    }

    if not gemini_key or not gemini_key.strip() or gemini_key.strip() in placeholder_keys:
        print("\n[VERIFICATION STATUS: SKIPPED / NOT CONFIGURED]")
        print("  GEMINI_API_KEY is not configured in your environment or backend/.env.")
        print("  In accordance with validation standards:")
        print("  * Live Gemini integration CANNOT be claimed as verified until an API key is provided")
        print("    and an actual cloud API request succeeds.")
        print("\n  To configure your Gemini API key and run live verification:")
        print("    1. Obtain an API key from Google AI Studio: https://aistudio.google.com/")
        print("    2. Create or edit backend/.env:")
        print("       GEMINI_API_KEY=your_actual_api_key_here")
        print("       GEMINI_MODEL=gemini-2.5-flash")
        print("    3. Or in Windows PowerShell:")
        print("       $env:GEMINI_API_KEY = 'your_actual_api_key_here'")
        print("    4. Re-run this verification command:")
        print("       .\\backend\\.venv\\Scripts\\python.exe backend/verify_live_gemini.py")
        print_banner("RESULT: LIVE VERIFICATION PENDING CONFIGURATION")
        return 1

    print("\n[Step 0/6] Gemini API key detected in environment/configuration.")
    print("  Key status: Configured (Key value is strictly confidential and never displayed).")
    target_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    print(f"  Target model: {target_model}")

    test_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------------
    # Step 1: Direct Model Generation Test
    # ------------------------------------------------------------------------
    print_banner("TEST 1: Direct Gemini Client Connection & Generation", "-")
    try:
        t0 = time.time()
        client = GeminiLLMClient(model_name=target_model)
        direct_response = client.generate_response(
            system_instruction="You are a helpful assistant. Reply concisely.",
            prompt="Respond with exactly the single word: CONFIRMED",
        )
        latency_1 = time.time() - t0
        print(f"  Response received in {latency_1:.2f}s: \"{direct_response.strip()}\"")
        assert len(direct_response.strip()) > 0, "Empty response received from Gemini."
        test_results.append({
            "test": "Direct LLM Generation",
            "passed": True,
            "status": "success",
            "latency": latency_1,
            "detail": f"Model {client.model_name} generated valid output.",
        })
        print("  -> TEST 1 PASSED: Direct connection to Gemini verified.")
    except Exception as e:
        err = sanitize_message(e, gemini_key)
        comp = classify_failure_component(err, "error")
        test_results.append({
            "test": "Direct LLM Generation",
            "passed": False,
            "status": "error",
            "component": comp,
            "error": err,
        })
        print(f"  -> TEST 1 FAILED: [{comp}] {err}")
        print_banner("ABORTING LIVE SUITE DUE TO CLIENT CONNECTION FAILURE")
        return 1

    # ------------------------------------------------------------------------
    # Step 2: Index Corpus Documents into FAISS
    # ------------------------------------------------------------------------
    print_banner("CORPUS INGESTION & LOCAL VECTOR INDEXING", "-")
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

    eval_data_dir = backend_dir / "evaluation_data"
    sample_files = [
        eval_data_dir / "security_policy.txt",
        eval_data_dir / "employee_handbook.txt",
        eval_data_dir / "expense_policy.txt",
    ]

    all_chunks: List[Dict[str, Any]] = []
    for fpath in sample_files:
        if fpath.is_file():
            doc = process_document(fpath)
            chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)
            all_chunks.extend(chunks)
            print(f"  Indexed: {doc['filename']} ({len(chunks)} chunks)")

    assert len(all_chunks) > 0, "No sample chunks available."
    embedded = embedding_service.embed_chunks(all_chunks)
    vector_store.add_chunks(embedded)
    print(f"  Total Indexed Vectors: {vector_store.total_vectors}")

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=client,
        top_k=3,
    )

    # ------------------------------------------------------------------------
    # Step 3: Grounded Document Question-Answering & Citations
    # ------------------------------------------------------------------------
    print_banner("TEST 2: Grounded Document Question-Answering & Citations", "-")
    q2 = "What is the maximum daily meal allowance for business travel according to the expense policy?"
    print(f"  Query: \"{q2}\"")
    try:
        t0 = time.time()
        resp2: RAGResponse = rag_pipeline.answer(q2)
        latency_2 = time.time() - t0
        print(f"  Latency: {latency_2:.2f}s")
        print(f"  Status: {resp2.generation_status} | Sufficient Context: {resp2.sufficient_context}")
        print(f"  Answer: \"{resp2.answer.strip()}\"")
        if resp2.error:
            print(f"  Underlying Error: {resp2.error}")
        print(f"  Citations Count: {len(resp2.citations)}")
        for idx, c in enumerate(resp2.citations, 1):
            print(f"    [{idx}] {c['filename']} (Page {c['page_number']}) | Cosine: {c['similarity_score']:.4f}")

        if resp2.generation_status != "success":
            diag_err = resp2.error or resp2.answer
            comp = classify_failure_component(diag_err, resp2.generation_status)
            raise AssertionError(f"[{comp}] Pipeline returned status '{resp2.generation_status}': {diag_err}")

        assert resp2.sufficient_context is True, "Expected sufficient context for documented fact."
        assert "75" in resp2.answer, "Expected answer to contain the $75 allowance cap."
        assert any("expense_policy" in (c["filename"] or "").lower() for c in resp2.citations), "Expected citation to expense_policy.txt"
        test_results.append({
            "test": "Grounded Document QA",
            "passed": True,
            "status": resp2.generation_status,
            "latency": latency_2,
            "detail": "Accurately answered factual query citing expense_policy.txt with $75 fact.",
        })
        print("  -> TEST 2 PASSED: Grounded RAG with citations verified.")
    except Exception as e:
        err = sanitize_message(e, gemini_key)
        comp = classify_failure_component(err, getattr(resp2, "generation_status", "error") if "resp2" in locals() else "error")
        test_results.append({
            "test": "Grounded Document QA",
            "passed": False,
            "status": getattr(resp2, "generation_status", "error") if "resp2" in locals() else "error",
            "component": comp,
            "error": err,
        })
        print(f"  -> TEST 2 FAILED: [{comp}] {err}")

    # ------------------------------------------------------------------------
    # Step 4: Safe Refusal on Insufficient Context
    # ------------------------------------------------------------------------
    print_banner("TEST 3: Hallucination Defense & Safe Refusal on Unknown Facts", "-")
    q3 = "What is the secret recipe for the executive chef's chocolate lava cake?"
    print(f"  Unanswerable Query: \"{q3}\"")
    try:
        t0 = time.time()
        resp3: RAGResponse = rag_pipeline.answer(q3)
        latency_3 = time.time() - t0
        print(f"  Latency: {latency_3:.2f}s")
        print(f"  Status: {resp3.generation_status} | Sufficient Context: {resp3.sufficient_context}")
        print(f"  Answer: \"{resp3.answer.strip()}\"")
        if resp3.error:
            print(f"  Underlying Error: {resp3.error}")

        if resp3.generation_status == "error":
            diag_err = resp3.error or resp3.answer
            comp = classify_failure_component(diag_err, resp3.generation_status)
            raise AssertionError(f"[{comp}] Language model generation failed during refusal test: {diag_err}")

        assert resp3.sufficient_context is False, "Refusal should set sufficient_context=False"
        assert resp3.generation_status == "insufficient_context", "Expected status 'insufficient_context'"
        assert "not contain enough information" in resp3.answer.lower() or "not mention" in resp3.answer.lower(), "Expected safe refusal message."
        test_results.append({
            "test": "Safe Refusal on Unknown Facts",
            "passed": True,
            "status": resp3.generation_status,
            "latency": latency_3,
            "detail": "Safely refused to invent facts for unanswerable question.",
        })
        print("  -> TEST 3 PASSED: Hallucination defense verified.")
    except Exception as e:
        err = sanitize_message(e, gemini_key)
        comp = classify_failure_component(err, getattr(resp3, "generation_status", "error") if "resp3" in locals() else "error")
        test_results.append({
            "test": "Safe Refusal on Unknown Facts",
            "passed": False,
            "status": getattr(resp3, "generation_status", "error") if "resp3" in locals() else "error",
            "component": comp,
            "error": err,
        })
        print(f"  -> TEST 3 FAILED: [{comp}] {err}")

    # ------------------------------------------------------------------------
    # Step 5: Multi-Turn Conversational Chat with SQLite Store
    # ------------------------------------------------------------------------
    print_banner("TEST 4: Multi-Turn Conversation History with SQLite Store", "-")
    temp_dir = tempfile.mkdtemp(prefix="intelliassist_live_test_")
    db_path = Path(temp_dir) / "live_chat_test.db"

    try:
        conv_store = ConversationStore(db_path=db_path)
        chatbot = ChatbotService(
            rag_pipeline=rag_pipeline,
            conversation_store=conv_store,
            max_history_messages=4,
        )
        conv = chatbot.create_conversation(title="Live Gemini Security Thread")
        cid = conv["conversation_id"]

        # Turn 1
        q_turn1 = "What is the minimum required password length according to the security policy?"
        print(f"  [Turn 1] Query: \"{q_turn1}\"")
        t0 = time.time()
        turn1_resp: ChatResponse = chatbot.chat(cid, q_turn1)
        latency_4a = time.time() - t0
        print(f"  [Turn 1] Response ({latency_4a:.2f}s): {turn1_resp.assistant_response.strip()}")
        if turn1_resp.error:
            print(f"  [Turn 1] Error: {turn1_resp.error}")
        if turn1_resp.status == "error":
            diag_err = turn1_resp.error or turn1_resp.assistant_response
            comp = classify_failure_component(diag_err, turn1_resp.status)
            raise AssertionError(f"[{comp}] Turn 1 failed: {diag_err}")

        assert turn1_resp.status == "success"
        assert turn1_resp.is_live_llm is True
        assert "14" in turn1_resp.assistant_response

        # Turn 2: Follow-up question relying on pronoun resolution
        q_turn2 = "How often must it be changed?"
        print(f"\n  [Turn 2] Follow-up Query: \"{q_turn2}\"")
        t0 = time.time()
        turn2_resp: ChatResponse = chatbot.chat(cid, q_turn2)
        latency_4b = time.time() - t0
        print(f"  [Turn 2] Response ({latency_4b:.2f}s): {turn2_resp.assistant_response.strip()}")
        if turn2_resp.error:
            print(f"  [Turn 2] Error: {turn2_resp.error}")
        if turn2_resp.status == "error":
            diag_err = turn2_resp.error or turn2_resp.assistant_response
            comp = classify_failure_component(diag_err, turn2_resp.status)
            raise AssertionError(f"[{comp}] Turn 2 failed: {diag_err}")

        assert turn2_resp.status == "success"
        assert turn2_resp.is_live_llm is True
        assert "90" in turn2_resp.assistant_response or "ninety" in turn2_resp.assistant_response.lower()

        # Check SQLite history
        history = chatbot.get_conversation_history(cid)
        assert len(history) == 4, f"Expected 4 messages in history, got {len(history)}"

        test_results.append({
            "test": "Multi-Turn Conversational Chat",
            "passed": True,
            "status": "success",
            "latency": latency_4a + latency_4b,
            "detail": "Verified 2-turn conversation resolving pronouns and preserving history in SQLite.",
        })
        print("  -> TEST 4 PASSED: Multi-turn conversational chat verified.")
    except Exception as e:
        err = sanitize_message(e, gemini_key)
        last_st = "error"
        if "turn2_resp" in locals():
            last_st = getattr(turn2_resp, "status", "error")
        elif "turn1_resp" in locals():
            last_st = getattr(turn1_resp, "status", "error")
        comp = classify_failure_component(err, last_st)
        test_results.append({
            "test": "Multi-Turn Conversational Chat",
            "passed": False,
            "status": last_st,
            "component": comp,
            "error": err,
        })
        print(f"  -> TEST 4 FAILED: [{comp}] {err}")
    finally:
        conv_store.close()
        try:
            if db_path.exists():
                db_path.unlink()
            os.rmdir(temp_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------------
    # Step 6: Abstractive Document Summarization via Gemini
    # ------------------------------------------------------------------------
    print_banner("TEST 5: Abstractive Document Summarization via Gemini", "-")
    try:
        summarizer = TextSummarizer(llm_client=client)
        text_to_summarize = (
            "The company's information security policy establishes strict standards to protect "
            "corporate and customer data. All employees must use complex passwords with at least "
            "14 characters, combining uppercase, lowercase, numbers, and symbols. Passwords expire "
            "every 90 days and cannot match the previous 10 passwords. Two-factor authentication (2FA) "
            "is mandatory for all remote access and cloud systems. Laptops must be encrypted with "
            "BitLocker or FileVault, and unapproved USB storage devices are strictly prohibited."
        )
        t0 = time.time()
        sum_result: SummaryResult = summarizer.summarize(
            text_to_summarize,
            summary_type="concise",
            prefer_llm=True,
        )
        latency_5 = time.time() - t0
        print(f"  Summary Latency: {latency_5:.2f}s")
        print(f"  Method Used:     {sum_result.method}")
        print(f"  Summary Output:  \"{sum_result.summary.strip()}\"")
        if sum_result.error:
            print(f"  Diagnostic Note: {sum_result.error}")

        assert sum_result.status == "success"
        if sum_result.method != "llm":
            fallback_reason = sum_result.error or f"Expected method 'llm', got '{sum_result.method}'"
            comp = classify_failure_component(fallback_reason, sum_result.status)
            raise AssertionError(f"[{comp}] Abstractive LLM summarization fell back to extractive: {fallback_reason}")

        assert len(sum_result.summary.strip()) > 0
        test_results.append({
            "test": "Abstractive Summarization via Gemini",
            "passed": True,
            "status": "success",
            "latency": latency_5,
            "detail": f"Generated abstractive summary using {client.model_name}.",
        })
        print("  -> TEST 5 PASSED: Abstractive summarization via Gemini verified.")
    except Exception as e:
        err = sanitize_message(e, gemini_key)
        comp = classify_failure_component(err, getattr(sum_result, "status", "error") if "sum_result" in locals() else "error")
        test_results.append({
            "test": "Abstractive Summarization via Gemini",
            "passed": False,
            "status": getattr(sum_result, "status", "error") if "sum_result" in locals() else "error",
            "component": comp,
            "error": err,
        })
        print(f"  -> TEST 5 FAILED: [{comp}] {err}")

    # ------------------------------------------------------------------------
    # Verification Summary
    # ------------------------------------------------------------------------
    total_duration = time.time() - start_total_time
    print_banner("LIVE GEMINI VERIFICATION SUMMARY")
    passed_count = sum(1 for t in test_results if t["passed"])
    total_count = len(test_results)

    print(f"\n  Total Live Scenarios: {total_count}")
    print(f"  Passed:                {passed_count}/{total_count}")
    print(f"  Total Duration:        {total_duration:.2f}s\n")

    for t in test_results:
        status_label = "[PASSED]" if t["passed"] else "[FAILED]"
        print(f"  {status_label:<9} {t['test']}")
        if t["passed"]:
            print(f"             Status:    {t.get('status', 'success')}")
            print(f"             Latency:   {t.get('latency', 0):.2f}s | {t.get('detail', '')}")
        else:
            print(f"             Status:    {t.get('status', 'error')}")
            print(f"             Component: {t.get('component', 'Unknown Component')}")
            print(f"             Error:     {t.get('error', 'Unknown error')}")

    if passed_count == total_count:
        print("\n  -> ALL LIVE GEMINI INTEGRATION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"\n  -> {total_count - passed_count} LIVE TEST(S) FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(verify_live_gemini())
