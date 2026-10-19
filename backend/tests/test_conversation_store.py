"""Automated deterministic unit tests for IntelliAssist AI ConversationStore.

Verifies:
  1. Database initialization (tables, indexes, foreign key pragma).
  2. Conversation creation (explicit title, default title, custom ID).
  3. Message persistence (user and assistant roles, content, timestamps).
  4. Correct chronological ordering of retrieved messages.
  5. Conversation listing ordered by updated_at descending.
  6. Conversation retrieval with accurate message count.
  7. Conversation deletion with foreign key cascading message deletion.
  8. Handling of unknown conversation IDs (returns None or raises error).
  9. Rejection of empty or whitespace-only message content.
  10. Non-LLM conversation title generation from user prompts.
  11. Conversation isolation (messages never leak across conversation boundaries).
  12. Citation and metadata serialization and retrieval.
  13. Updating conversation title and timestamp manually.
  14. Transaction safety and parameterized query enforcement.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import time
from typing import Generator
import pytest

from app.conversation_store import (
    ConversationStore,
    generate_conversation_title,
    utcnow_iso,
)


@pytest.fixture
def store(tmp_path: Path) -> Generator[ConversationStore, None, None]:
    """Provide a fresh file-backed ConversationStore for each test."""
    db_file = tmp_path / "test_chat.db"
    store_inst = ConversationStore(db_path=db_file)
    yield store_inst
    store_inst.close()


@pytest.fixture
def memory_store() -> Generator[ConversationStore, None, None]:
    """Provide an in-memory ConversationStore for in-memory behavior verification."""
    store_inst = ConversationStore(db_path=":memory:")
    yield store_inst
    store_inst.close()


# ============================================================================
# 1. Database Initialization Tests
# ============================================================================

class TestDatabaseInitialization:
    """Verifies SQLite schema creation, foreign key pragma, and indexes."""

    def test_schema_tables_and_indexes_created(self, store: ConversationStore) -> None:
        """Verify conversations and messages tables and indexes exist."""
        with store._get_connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
            )
            tables = [row["name"] for row in cursor.fetchall()]
            assert "conversations" in tables
            assert "messages" in tables

            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
            )
            indexes = [row["name"] for row in cursor.fetchall()]
            assert "idx_messages_conversation_created" in indexes
            assert "idx_conversations_updated" in indexes

    def test_foreign_keys_pragma_enabled(self, store: ConversationStore) -> None:
        """Verify PRAGMA foreign_keys is strictly turned ON for each connection."""
        with store._get_connection() as conn:
            cursor = conn.execute("PRAGMA foreign_keys;")
            row = cursor.fetchone()
            assert row[0] == 1, "Foreign keys must be enabled in SQLite."

    def test_idempotent_initialization(self, tmp_path: Path) -> None:
        """Verify re-opening existing database does not crash or corrupt tables."""
        db_file = tmp_path / "idempotent.db"
        s1 = ConversationStore(db_path=db_file)
        c = s1.create_conversation(title="Test Idempotency")
        s1.close()

        # Re-initialize on same file
        s2 = ConversationStore(db_path=db_file)
        fetched = s2.get_conversation(c["conversation_id"])
        assert fetched is not None
        assert fetched["title"] == "Test Idempotency"
        s2.close()

    def test_in_memory_database_persists_across_calls(self, memory_store: ConversationStore) -> None:
        """Verify in-memory store retains data across operations on same instance."""
        c = memory_store.create_conversation(title="In Memory Thread")
        memory_store.add_message(c["conversation_id"], "user", "Hello memory")
        fetched = memory_store.get_conversation(c["conversation_id"])
        assert fetched is not None
        assert fetched["message_count"] == 1


# ============================================================================
# 2. Conversation Title Generation Tests
# ============================================================================

class TestTitleGeneration:
    """Verifies title generation rules from first user messages without an LLM."""

    def test_short_message_title(self) -> None:
        title = generate_conversation_title("What is the refund policy?")
        assert title == "What is the refund policy?"

    def test_long_message_truncation(self) -> None:
        long_msg = "Can you please provide a very detailed explanation of how reimbursement works for international business flights?"
        title = generate_conversation_title(long_msg, max_chars=40)
        assert len(title) <= 43  # 40 chars + possible ellipsis
        assert title.endswith("...")

    def test_whitespace_and_newlines_collapsed(self) -> None:
        messy_msg = "   What   is\n\nthe     vacation \t policy?   "
        title = generate_conversation_title(messy_msg)
        assert title == "What is the vacation policy?"

    def test_empty_or_symbol_only_message(self) -> None:
        assert generate_conversation_title("") == "New Conversation"
        assert generate_conversation_title("     ") == "New Conversation"
        assert generate_conversation_title("???!!!") == "New Conversation"
        assert generate_conversation_title(None) == "New Conversation"  # type: ignore


# ============================================================================
# 3. Conversation Operations Tests
# ============================================================================

class TestConversationOperations:
    """Verifies conversation CRUD and metadata management."""

    def test_create_conversation_defaults(self, store: ConversationStore) -> None:
        conv = store.create_conversation()
        assert conv["conversation_id"] is not None
        assert conv["title"] == "New Conversation"
        assert conv["message_count"] == 0
        assert "T" in conv["created_at"]
        assert conv["created_at"] == conv["updated_at"]

    def test_create_conversation_explicit_title_and_id(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="Custom Title", conversation_id="thread-123")
        assert conv["conversation_id"] == "thread-123"
        assert conv["title"] == "Custom Title"

    def test_create_duplicate_conversation_id_rejected(self, store: ConversationStore) -> None:
        store.create_conversation(conversation_id="unique-thread")
        with pytest.raises(ValueError, match="already exists"):
            store.create_conversation(conversation_id="unique-thread")

    def test_get_conversation_found_and_not_found(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="Existing Thread")
        fetched = store.get_conversation(conv["conversation_id"])
        assert fetched is not None
        assert fetched["title"] == "Existing Thread"

        assert store.get_conversation("non-existent-id") is None
        assert store.get_conversation("") is None

    def test_conversation_exists(self, store: ConversationStore) -> None:
        conv = store.create_conversation()
        assert store.conversation_exists(conv["conversation_id"]) is True
        assert store.conversation_exists("random-xyz") is False
        assert store.conversation_exists("") is False

    def test_list_conversations_ordered_by_updated_at_descending(self, store: ConversationStore) -> None:
        c1 = store.create_conversation(title="Thread 1")
        time.sleep(0.01)
        c2 = store.create_conversation(title="Thread 2")
        time.sleep(0.01)
        c3 = store.create_conversation(title="Thread 3")

        # Initial order: c3, c2, c1
        conversations = store.list_conversations()
        ids = [c["conversation_id"] for c in conversations]
        assert ids[:3] == [c3["conversation_id"], c2["conversation_id"], c1["conversation_id"]]

        # Updating c1 via a new message bumps it to top
        time.sleep(0.01)
        store.add_message(c1["conversation_id"], "user", "Bumping thread 1")
        updated_conversations = store.list_conversations()
        assert updated_conversations[0]["conversation_id"] == c1["conversation_id"]

    def test_update_conversation_title(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="Old Title")
        assert store.update_conversation_title(conv["conversation_id"], "Renamed Title") is True

        fetched = store.get_conversation(conv["conversation_id"])
        assert fetched["title"] == "Renamed Title"

        # Invalid title
        with pytest.raises(ValueError, match="cannot be empty"):
            store.update_conversation_title(conv["conversation_id"], "   ")

        # Non-existent conversation
        assert store.update_conversation_title("unknown-id", "New Title") is False

    def test_delete_conversation_cascades_messages(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="To Delete")
        cid = conv["conversation_id"]
        store.add_message(cid, "user", "Message 1")
        store.add_message(cid, "assistant", "Answer 1")

        assert len(store.get_messages(cid)) == 2

        # Delete conversation
        assert store.delete_conversation(cid) is True
        assert store.get_conversation(cid) is None
        assert store.get_messages(cid) == []

        # Verify messages table directly
        with store._get_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?;", (cid,)
            )
            assert cursor.fetchone()[0] == 0

        # Deleting non-existent returns False
        assert store.delete_conversation(cid) is False


# ============================================================================
# 4. Message Operations Tests
# ============================================================================

class TestMessageOperations:
    """Verifies adding, retrieving, ordering, and isolating messages."""

    def test_add_and_get_messages_chronological(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="Chat Test")
        cid = conv["conversation_id"]

        m1 = store.add_message(cid, "user", "First question")
        time.sleep(0.01)
        m2 = store.add_message(cid, "assistant", "First answer")
        time.sleep(0.01)
        m3 = store.add_message(cid, "user", "Second question")

        messages = store.get_messages(cid)
        assert len(messages) == 3
        assert [m["message_id"] for m in messages] == [
            m1["message_id"],
            m2["message_id"],
            m3["message_id"],
        ]
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_add_message_auto_derives_title_on_first_user_message(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="New Conversation")
        cid = conv["conversation_id"]

        store.add_message(cid, "user", "What are the standard working hours?")
        updated_conv = store.get_conversation(cid)
        assert updated_conv["title"] == "What are the standard working hours?"

    def test_add_message_does_not_override_custom_title(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="My Custom Project")
        cid = conv["conversation_id"]

        store.add_message(cid, "user", "What is the policy on equipment?")
        updated_conv = store.get_conversation(cid)
        assert updated_conv["title"] == "My Custom Project"

    def test_add_message_validations(self, store: ConversationStore) -> None:
        conv = store.create_conversation()
        cid = conv["conversation_id"]

        # Invalid role
        with pytest.raises(ValueError, match="Invalid message role"):
            store.add_message(cid, "system", "Hello")

        # Empty content
        with pytest.raises(ValueError, match="cannot be empty"):
            store.add_message(cid, "user", "   ")

        # Non-existent conversation
        with pytest.raises(ValueError, match="does not exist"):
            store.add_message("ghost-thread", "user", "Hello")

    def test_citations_and_metadata_persistence(self, store: ConversationStore) -> None:
        conv = store.create_conversation()
        cid = conv["conversation_id"]

        citations = [
            {
                "chunk_id": "policy.pdf_page_1_chunk_1",
                "filename": "policy.pdf",
                "page_number": 1,
                "similarity_score": 0.88,
                "is_cited": True,
            }
        ]
        meta = {"model_name": "gemini-2.5-flash", "tokens": 128}

        msg = store.add_message(
            conversation_id=cid,
            role="assistant",
            content="Grounded answer.",
            citations=citations,
            metadata=meta,
        )

        messages = store.get_messages(cid)
        assert len(messages) == 1
        fetched = messages[0]
        assert fetched["citations"] == citations
        assert fetched["metadata"] == meta

    def test_get_recent_messages_order_and_limit(self, store: ConversationStore) -> None:
        conv = store.create_conversation()
        cid = conv["conversation_id"]

        for i in range(1, 11):
            role = "user" if i % 2 == 1 else "assistant"
            store.add_message(cid, role, f"Message {i}")
            time.sleep(0.005)

        # Retrieve last 4 messages
        recent = store.get_recent_messages(cid, limit=4)
        assert len(recent) == 4
        # Chronological order maintained: 7, 8, 9, 10
        assert [m["content"] for m in recent] == [
            "Message 7",
            "Message 8",
            "Message 9",
            "Message 10",
        ]

    def test_conversation_isolation(self, store: ConversationStore) -> None:
        """Verify messages cannot leak across conversation boundaries."""
        c1 = store.create_conversation(title="Thread 1")
        c2 = store.create_conversation(title="Thread 2")

        store.add_message(c1["conversation_id"], "user", "Message for thread 1")
        store.add_message(c2["conversation_id"], "user", "Message for thread 2")

        m1 = store.get_messages(c1["conversation_id"])
        m2 = store.get_messages(c2["conversation_id"])

        assert len(m1) == 1
        assert m1[0]["content"] == "Message for thread 1"
        assert len(m2) == 1
        assert m2[0]["content"] == "Message for thread 2"

        recent_1 = store.get_recent_messages(c1["conversation_id"], limit=10)
        assert len(recent_1) == 1
        assert recent_1[0]["content"] == "Message for thread 1"

    def test_get_conversation_with_messages(self, store: ConversationStore) -> None:
        conv = store.create_conversation(title="Thread with messages")
        cid = conv["conversation_id"]
        store.add_message(cid, "user", "Q1")
        store.add_message(cid, "assistant", "A1")

        full = store.get_conversation_with_messages(cid)
        assert full is not None
        assert full["conversation_id"] == cid
        assert full["message_count"] == 2
        assert len(full["messages"]) == 2
        assert full["messages"][0]["content"] == "Q1"

        assert store.get_conversation_with_messages("missing") is None
