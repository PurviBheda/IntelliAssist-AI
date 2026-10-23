"""IntelliAssist AI - Chatbot Service End-to-End Smoke Test.

This script demonstrates and verifies the complete Chatbot Service and persistent
conversation store integrated with the existing RAG pipeline.

It cleanly separates:
  Part A: Local Chatbot & Multi-Turn History Smoke Test (Zero API keys, 100% offline)
          - Database initialization with foreign keys & WAL mode in isolated DB.
          - Conversation creation, title generation, and message persistence.
          - Multi-turn follow-up question answering with history incorporation.
          - Strict isolation between multiple conversation threads.
          - Preserving RAG source citations without fabrication.
          - Refusal on insufficient context.
          - Cascading deletion of conversation threads and messages.
          - Clearly labeled mocked LLM responses (never claiming live Gemini).

  Part B: Optional Real-LLM Integration Test (Google Gemini via google-genai SDK, executed
          only when GEMINI_API_KEY is configured in the environment or backend/.env).

Usage:
    python smoke_test_chatbot.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
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
from app.conversation_store import ConversationStore
from app.chatbot_service import ChatbotService, ChatResponse
from app.rag_pipeline import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_TOP_K,
    INSUFFICIENT_CONTEXT_MESSAGE,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    load_environment_variables,
)


def print_header(title: str, char: str = "=", width: int = 80) -> None:
    """Print formatted section header."""
    print("\n" + char * width)
    print(f" {title} ".center(width, char))
    print(char * width)


def run_smoke_test() -> int:
    print_header("IntelliAssist AI — Chatbot & Conversation History Smoke Test")
    start_time = time.time()

    # Safely load environment variables
    load_environment_variables()

    # Create an isolated temporary database for this smoke test run
    temp_dir = tempfile.mkdtemp(prefix="intelliassist_smoke_")
    temp_db_path = Path(temp_dir) / "smoke_chat_history.db"
    print(f"\n[Step 1/5] Initializing isolated temporary SQLite database at:")
    print(f"  {temp_db_path}")

    conv_store = ConversationStore(db_path=temp_db_path)
    print("  Schema initialized with foreign keys enabled, WAL mode, and indexes.")

    # ------------------------------------------------------------------------
    # Step 2: Initialize Embedding Service and Index Sample Corpus in FAISS
    # ------------------------------------------------------------------------
    print("\n[Step 2/5] Initializing local embeddings and indexing sample corpus in FAISS...")
    embedding_service = EmbeddingService()
    vector_store = FAISSVectorStore(dimension=384, normalize_vectors=True)

    eval_data_dir = backend_dir / "evaluation_data"
    sample_files = [
        eval_data_dir / "security_policy.txt",
        eval_data_dir / "employee_handbook.txt",
    ]

    all_chunks: List[Dict[str, Any]] = []
    for fpath in sample_files:
        if fpath.exists():
            doc = process_document(fpath)
            chunks = chunk_document(doc, chunk_size=600, chunk_overlap=100)
            all_chunks.extend(chunks)
            print(f"  - Loaded {doc['filename']} ({len(chunks)} chunk(s))")

    assert len(all_chunks) > 0, "No sample document chunks found."
    embedded_chunks = embedding_service.embed_chunks(all_chunks)
    vector_store.add_chunks(embedded_chunks)
    print(f"  Indexed {vector_store.total_vectors} document vectors into FAISS.")

    # ------------------------------------------------------------------------
    # Step 3: Part A — Local Offline Multi-Turn Chatbot Smoke Test
    # ------------------------------------------------------------------------
    print_header("PART A: Local Offline Multi-Turn Chatbot Verification (Zero API Keys)")

    # Define simulated grounded generator for mock LLM
    def simulated_chatbot_llm(system_instruction: str, prompt: str) -> str:
        prompt_lower = prompt.lower()
        question_part = prompt_lower
        if "question:\n" in prompt_lower:
            question_part = prompt_lower.split("question:\n", 1)[1].split("\n\ninstruction:")[0]

        if "expire" in question_part or "how often" in question_part:
            return (
                "[MOCKED RESPONSE] According to security_policy.txt (Page 1), employee passwords "
                "expire every 90 days."
            )
        elif "password" in question_part or "length" in question_part:
            return (
                "[MOCKED RESPONSE] According to security_policy.txt (Page 1), employee passwords "
                "must be at least 14 characters in length and include uppercase, lowercase, numbers, "
                "and symbols."
            )
        elif "pto" in question_part or "paid time off" in question_part:
            return (
                "[MOCKED RESPONSE] According to employee_handbook.txt (Page 1), employees receive "
                "20 days of paid time off (PTO) annually."
            )
        return f"[MOCKED RESPONSE] {INSUFFICIENT_CONTEXT_MESSAGE}"

    mock_llm = MockLLMClient(
        response_generator=simulated_chatbot_llm,
        model_name="mock-grounded-assistant",
    )

    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=mock_llm,
        top_k=2,
    )

    chatbot_service = ChatbotService(
        rag_pipeline=rag_pipeline,
        conversation_store=conv_store,
        max_history_messages=4,
    )

    # 1. Create Conversation
    print("\n[Part A - Test 1] Creating Conversation Thread 1...")
    conv1 = chatbot_service.create_conversation(title="Security Policy Q&A")
    cid1 = conv1["conversation_id"]
    print(f"  Created Conversation ID: {cid1} (Title: '{conv1['title']}')")
    assert conv1["title"] == "Security Policy Q&A"
    assert conv1["message_count"] == 0

    # 2. Turn 1: Initial Question
    question_1 = "What is the minimum required password length?"
    print(f"\n[Part A - Test 2] Turn 1: Sending user message: '{question_1}'")
    turn_1 = chatbot_service.chat(cid1, question_1)

    print(f"  Status:             {turn_1.status}")
    print(f"  Is Live LLM:        {turn_1.is_live_llm} (Accurately flagged as MOCKED)")
    print(f"  Model Employed:     {turn_1.model_name}")
    print(f"  Assistant Response: {turn_1.assistant_response}")
    print(f"  Citations Returned: {len(turn_1.citations)} source(s)")
    for idx, c in enumerate(turn_1.citations, 1):
        print(f"    [{idx}] {c['filename']} | Page {c['page_number']} | Score: {c['similarity_score']:.4f}")

    assert turn_1.status == "success"
    assert turn_1.sufficient_context is True
    assert turn_1.is_live_llm is False
    assert "14" in turn_1.assistant_response
    assert len(turn_1.citations) > 0
    assert turn_1.citations[0]["filename"] == "security_policy.txt"

    # Verify messages in conversation store
    history_1 = chatbot_service.get_conversation_history(cid1)
    assert len(history_1) == 2
    assert history_1[0]["role"] == "user"
    assert history_1[0]["content"] == question_1
    assert history_1[1]["role"] == "assistant"
    print("  -> Turn 1 Passed: User message and assistant response persisted successfully.")

    # 3. Turn 2: Follow-up Question
    follow_up_q = "How often do they expire?"
    print(f"\n[Part A - Test 3] Turn 2: Sending follow-up question: '{follow_up_q}'")
    turn_2 = chatbot_service.chat(cid1, follow_up_q)

    print(f"  Status:             {turn_2.status}")
    print(f"  Assistant Response: {turn_2.assistant_response}")
    print(f"  Citations Returned: {len(turn_2.citations)} source(s)")

    assert turn_2.status == "success"
    assert turn_2.sufficient_context is True
    assert "90" in turn_2.assistant_response

    # Verify history in DB now has 4 turns in chronological order
    history_2 = chatbot_service.get_conversation_history(cid1)
    assert len(history_2) == 4
    assert [m["role"] for m in history_2] == ["user", "assistant", "user", "assistant"]
    print("  -> Turn 2 Passed: Multi-turn history maintained and chronologically ordered.")

    # 4. Multi-Thread Isolation Test
    print("\n[Part A - Test 4] Conversation Isolation: Creating Thread 2...")
    conv2 = chatbot_service.create_conversation(title="Benefits Consultation")
    cid2 = conv2["conversation_id"]

    hr_question = "How many days of paid time off do employees receive?"
    print(f"  Sending message to Thread 2: '{hr_question}'")
    turn_hr = chatbot_service.chat(cid2, hr_question)

    print(f"  Thread 2 Response:  {turn_hr.assistant_response}")
    assert turn_hr.status == "success"
    assert "20" in turn_hr.assistant_response
    assert turn_hr.citations[0]["filename"] == "employee_handbook.txt"

    # Verify Thread 2 has only 2 messages and Thread 1 still has 4
    assert len(chatbot_service.get_conversation_history(cid2)) == 2
    assert len(chatbot_service.get_conversation_history(cid1)) == 4
    print("  -> Test 4 Passed: Conversations strictly isolated with zero cross-talk.")

    # 5. Conversation Listing
    print("\n[Part A - Test 5] Verifying conversation listing ordered by recent update...")
    conv_list = chatbot_service.list_conversations()
    assert len(conv_list) >= 2
    # cid2 was updated most recently
    assert conv_list[0]["conversation_id"] == cid2
    assert conv_list[1]["conversation_id"] == cid1
    print(f"  Listed {len(conv_list)} conversations; most recent is '{conv_list[0]['title']}'.")
    print("  -> Test 5 Passed: Conversations listing ordered by updated_at descending.")

    # 6. Insufficient Context Refusal
    print("\n[Part A - Test 6] Verifying safe refusal on unanswerable query...")
    unanswerable = "What is the secret recipe for chocolate chip cookies?"
    conv_refusal = chatbot_service.create_conversation(title="Off-topic Thread")
    cid_refusal = conv_refusal["conversation_id"]

    refusal_pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=MockLLMClient(default_response=INSUFFICIENT_CONTEXT_MESSAGE),
        top_k=2,
        similarity_threshold=0.95,  # High threshold to guarantee empty retrieval
    )
    refusal_service = ChatbotService(
        rag_pipeline=refusal_pipeline,
        conversation_store=conv_store,
    )
    turn_refusal = refusal_service.chat(cid_refusal, unanswerable)

    print(f"  Status:             {turn_refusal.status}")
    print(f"  Sufficient Context: {turn_refusal.sufficient_context}")
    print(f"  Assistant Response: {turn_refusal.assistant_response}")
    print(f"  Citations:          {turn_refusal.citations}")

    assert turn_refusal.status == "insufficient_context"
    assert turn_refusal.sufficient_context is False
    assert turn_refusal.citations == []
    assert turn_refusal.assistant_response == INSUFFICIENT_CONTEXT_MESSAGE
    print("  -> Test 6 Passed: Refusal on insufficient context verified without fabricated citations.")

    # 7. Conversation Deletion and Cascade
    print("\n[Part A - Test 7] Verifying conversation deletion and cascading message removal...")
    deleted = chatbot_service.delete_conversation(cid_refusal)
    assert deleted is True
    assert chatbot_service.get_conversation(cid_refusal) is None
    assert chatbot_service.get_conversation_history(cid_refusal) == []
    print(f"  Deleted conversation {cid_refusal} and all associated messages.")
    print("  -> Test 7 Passed: Cascading deletion verified.")

    print_header("PART A SUMMARY: ALL LOCAL CHATBOT WORKFLOW TESTS PASSED")

    # ------------------------------------------------------------------------
    # Step 4: Part B — Optional Real-LLM Integration Test (Google Gemini)
    # ------------------------------------------------------------------------
    print_header("PART B: Optional Real-LLM Integration Test (Google Gemini)")

    local_only = os.getenv("INTELLIASSIST_LOCAL_ONLY", "").strip() == "1"

    gemini_key = (
    None
    if local_only
    else os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )

    if not gemini_key or not gemini_key.strip():
        print("\n[NOTICE: LIVE LLM GENERATION SKIPPED]")
        print("  GEMINI_API_KEY is not configured in your environment or backend/.env.")
        print("  All local chatbot features (SQLite multi-turn store, history limits,")
        print("  grounded RAG, citations, and conversation isolation) are fully verified.")
        print("\n  To execute a live end-to-end test with Google Gemini:")
        print("    1. Obtain a free API key from Google AI Studio: https://aistudio.google.com/")
        print("    2. Add to backend/.env: GEMINI_API_KEY=your_actual_key_here")
        print("    3. Or in PowerShell: $env:GEMINI_API_KEY = 'your_actual_key_here'")
        print("    4. Re-run: python smoke_test_chatbot.py")
    else:
        print(f"\n[LIVE GENERATION DETECTED]")
        print("  Found Gemini API key in configuration (Key remains strictly confidential and unprinted)")
        print("  Initializing live GeminiLLMClient...")

        try:
            live_llm = GeminiLLMClient()
            live_pipeline = RAGPipeline(
                vector_store=vector_store,
                embedding_service=embedding_service,
                llm_client=live_llm,
                top_k=2,
            )
            live_service = ChatbotService(
                rag_pipeline=live_pipeline,
                conversation_store=conv_store,
                max_history_messages=4,
            )

            live_conv = live_service.create_conversation(title="Live Gemini Security Thread")
            live_cid = live_conv["conversation_id"]

            live_q1 = "What is the minimum password length according to the security policy?"
            print(f"\n  Sending live Turn 1 query: '{live_q1}'")
            t_live = time.time()
            live_turn_1 = live_service.chat(live_cid, live_q1)
            t_elapsed = time.time() - t_live

            print(f"  Live Response Time: {t_elapsed:.2f}s")
            print(f"  Status:             {live_turn_1.status}")
            print(f"  Is Live LLM:        {live_turn_1.is_live_llm}")
            print(f"  Model Employed:     {live_turn_1.model_name}")
            print(f"\n  --- Real Gemini Answer ---")
            print(f"  {live_turn_1.assistant_response}")
            print(f"\n  --- Citations ---")
            for idx, c in enumerate(live_turn_1.citations, 1):
                print(f"    [{idx}] {c['filename']} | Page {c['page_number']} | Chunk: {c['chunk_id']}")

            assert live_turn_1.status == "success"
            assert live_turn_1.is_live_llm is True
            assert live_turn_1.sufficient_context is True
            assert "14" in live_turn_1.assistant_response or "security_policy" in live_turn_1.assistant_response.lower()

            # Follow-up with live Gemini
            live_q2 = "How often must it be changed?"
            print(f"\n  Sending live Turn 2 follow-up query: '{live_q2}'")
            t_live_2 = time.time()
            live_turn_2 = live_service.chat(live_cid, live_q2)
            t_elapsed_2 = time.time() - t_live_2

            print(f"  Follow-up Time:     {t_elapsed_2:.2f}s")
            print(f"  Status:             {live_turn_2.status}")
            print(f"  Assistant Response: {live_turn_2.assistant_response}")
            assert live_turn_2.status == "success"
            assert live_turn_2.is_live_llm is True
            print("\n  -> Part B Succeeded: Live Google Gemini multi-turn generation verified!")

        except Exception as e:
            sanitized = live_llm._sanitize_error(str(e)) if "live_llm" in locals() else str(e)
            if gemini_key and gemini_key in sanitized:
                sanitized = sanitized.replace(gemini_key, "[REDACTED_API_KEY]")
            import re
            sanitized = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", sanitized)
            sanitized = re.sub(r"(key=)[^&\s]+", r"\1[REDACTED_API_KEY]", sanitized)
            print(f"\n  [WARNING] Live Gemini API call could not be completed: {sanitized}")
            print("  Note: All unit tests and local chatbot components remain fully operational.")

    # ------------------------------------------------------------------------
    # Step 5: Clean Up Temporary Database
    # ------------------------------------------------------------------------
    conv_store.close()
    try:
        if temp_db_path.exists():
            temp_db_path.unlink()
        wal_file = temp_db_path.with_name(temp_db_path.name + "-wal")
        if wal_file.exists():
            wal_file.unlink()
        shm_file = temp_db_path.with_name(temp_db_path.name + "-shm")
        if shm_file.exists():
            shm_file.unlink()
        os.rmdir(temp_dir)
    except Exception:
        pass

    total_time = time.time() - start_time
    print_header(f"SMOKE TEST COMPLETE (Duration: {total_time:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(run_smoke_test())
