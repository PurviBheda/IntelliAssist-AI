"""IntelliAssist AI - Persistent Conversation Store Module.

This module provides a production-grade, thread-safe, SQLite-backed conversation
and message storage layer for the IntelliAssist AI document assistant.

Key Features:
- Parameterized SQL queries preventing SQL injection vulnerabilities.
- SQLite foreign-key enforcement (PRAGMA foreign_keys = ON).
- Write-Ahead Logging (WAL) and busy timeouts for reliable concurrency.
- Atomic transactions for multi-row writes.
- Strict isolation across conversation boundaries.
- Consistent ISO 8601 UTC timestamps.
- Automatic cascading deletion of messages when a conversation is deleted.
- Non-LLM safe conversation title derivation from first user prompt.
- Full support for in-memory (":memory:") or persistent file databases.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Union
import uuid


# ============================================================================
# Constants & Defaults
# ============================================================================

DEFAULT_DB_FILENAME = "chat_history.db"
DEFAULT_TITLE_MAX_CHARS = 45


def get_default_db_path() -> Path:
    """Return default path to chat history database in backend/data/."""
    backend_dir = Path(__file__).resolve().parent.parent
    data_dir = backend_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / DEFAULT_DB_FILENAME


def utcnow_iso() -> str:
    """Generate current UTC timestamp in standardized ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def generate_conversation_title(message: str, max_chars: int = DEFAULT_TITLE_MAX_CHARS) -> str:
    """Derive a concise, safe conversation title from the first user prompt without calling an LLM.

    Args:
        message: First user prompt text.
        max_chars: Maximum character limit for the title (default: 45).

    Returns:
        Cleaned, sanitized title string (defaults to 'New Conversation' if empty).
    """
    if not message or not isinstance(message, str):
        return "New Conversation"

    # Collapse consecutive whitespace and newlines
    cleaned = " ".join(message.split()).strip()
    if not cleaned:
        return "New Conversation"

    # Strip leading non-alphanumeric punctuation
    cleaned = re.sub(r"^[^a-zA-Z0-9]+", "", cleaned).strip()
    if not cleaned:
        return "New Conversation"

    if len(cleaned) <= max_chars:
        return cleaned

    # Truncate at word boundary when possible
    truncated = cleaned[:max_chars].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space].rstrip()

    return f"{truncated}..."


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ConversationRecord:
    """Structured representation of a conversation thread."""

    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


@dataclass
class MessageRecord:
    """Structured representation of a chat message."""

    message_id: str
    conversation_id: str
    role: str
    content: str
    created_at: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


# ============================================================================
# Main Conversation Store Class
# ============================================================================

class ConversationStore:
    """SQLite-backed persistent repository for conversations and message history."""

    SCHEMA_STATEMENTS = [
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            citations TEXT,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
            ON messages(conversation_id, created_at ASC);
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_conversations_updated
            ON conversations(updated_at DESC);
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation_documents (
            document_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            total_pages INTEGER NOT NULL,
            total_characters INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_documents_conversation
            ON conversation_documents(conversation_id, created_at ASC);
        """,
    ]

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the conversation store and ensure schema existence.

        Args:
            db_path: Path to SQLite database file or ':memory:'. Defaults to
                     backend/data/chat_history.db.
        """
        if db_path is None:
            self._db_path: Union[str, Path] = get_default_db_path()
        else:
            self._db_path = db_path

        self._is_memory = str(self._db_path) == ":memory:"
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self._is_memory:
            # Persistent connection for in-memory database to preserve data across calls
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._configure_connection(self._memory_conn)
        else:
            # Ensure parent directory exists for file-based DBs
            path_obj = Path(self._db_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_schema()

    @property
    def db_path(self) -> Union[str, Path]:
        """Return the database path."""
        return self._db_path

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply required SQLite PRAGMAs for safety and performance."""
        conn.row_factory = sqlite3.Row
        # Crucial: SQLite turns foreign keys OFF by default; must enable per connection
        conn.execute("PRAGMA foreign_keys = ON;")
        if not self._is_memory:
            try:
                conn.execute("PRAGMA journal_mode = WAL;")
            except Exception:
                pass
            conn.execute("PRAGMA busy_timeout = 5000;")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Yield an active SQLite connection with appropriate pragmas and cleanup."""
        if self._is_memory:
            assert self._memory_conn is not None
            yield self._memory_conn
        else:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._configure_connection(conn)
            try:
                yield conn
            finally:
                conn.close()

    def _initialize_schema(self) -> None:
        """Create database tables and indexes if they do not already exist."""
        with self._get_connection() as conn:
            with conn:
                for statement in self.SCHEMA_STATEMENTS:
                    conn.execute(statement)

    def close(self) -> None:
        """Explicitly close persistent database connection if open."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:
                pass
            self._memory_conn = None

    def __enter__(self) -> ConversationStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # ========================================================================
    # Conversation Operations
    # ========================================================================

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation thread.

        Args:
            title: Human-readable title or None (defaults to 'New Conversation').
            conversation_id: Unique string identifier or None (auto-generates UUID).

        Returns:
            Dictionary representing the created conversation record.

        Raises:
            ValueError: If conversation_id is invalid or already exists.
        """
        cid = conversation_id.strip() if isinstance(conversation_id, str) else None
        if not cid:
            cid = str(uuid.uuid4())

        now = utcnow_iso()
        resolved_title = title.strip() if isinstance(title, str) and title.strip() else "New Conversation"

        with self._get_connection() as conn:
            # Check for existing conversation
            cursor = conn.execute(
                "SELECT conversation_id FROM conversations WHERE conversation_id = ?;",
                (cid,),
            )
            if cursor.fetchone() is not None:
                raise ValueError(f"Conversation with ID '{cid}' already exists.")

            with conn:
                conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?);
                    """,
                    (cid, resolved_title, now, now),
                )

        return {
            "conversation_id": cid,
            "title": resolved_title,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check whether a conversation with the given ID exists.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if conversation exists, False otherwise.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            return False

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ? LIMIT 1;",
                (conversation_id,),
            )
            return cursor.fetchone() is not None

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve conversation record and message count by ID.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Conversation dictionary or None if not found.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            return None

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                WHERE c.conversation_id = ?
                GROUP BY c.conversation_id;
                """,
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "conversation_id": row["conversation_id"],
                "title": row["title"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": int(row["message_count"]),
                "documents": self.get_documents(conversation_id),
            }

    def list_conversations(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List conversations ordered by most recently updated first.

        Args:
            limit: Maximum number of conversations to retrieve (default: 50).
            offset: Pagination offset (default: 0).

        Returns:
            List of conversation dictionaries.
        """
        if not isinstance(limit, int) or limit <= 0:
            limit = 50
        if not isinstance(offset, int) or offset < 0:
            offset = 0

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON c.conversation_id = m.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()

            return [
                {
                    "conversation_id": row["conversation_id"],
                    "title": row["title"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "message_count": int(row["message_count"]),
                    "documents": self.get_documents(row["conversation_id"]),
                }
                for row in rows
            ]

    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """Update the title of an existing conversation.

        Args:
            conversation_id: Unique conversation identifier.
            title: New title string.

        Returns:
            True if updated, False if conversation not found.

        Raises:
            ValueError: If title is empty or invalid.
        """
        if not title or not isinstance(title, str) or not title.strip():
            raise ValueError("Conversation title cannot be empty.")

        with self._get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE conversations SET title = ? WHERE conversation_id = ?;",
                    (title.strip(), conversation_id),
                )
                return cursor.rowcount > 0

    def update_conversation_timestamp(
        self,
        conversation_id: str,
        timestamp: Optional[str] = None,
    ) -> bool:
        """Update the updated_at timestamp of a conversation.

        Args:
            conversation_id: Conversation identifier.
            timestamp: ISO 8601 UTC timestamp string or None (uses current time).

        Returns:
            True if updated, False if conversation not found.
        """
        ts = timestamp or utcnow_iso()
        with self._get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;",
                    (ts, conversation_id),
                )
                return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its associated messages.

        With SQLite foreign key cascading enabled, messages referencing this
        conversation are automatically deleted in the same transaction.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            True if conversation existed and was deleted, False otherwise.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            return False

        with self._get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?;",
                    (conversation_id,),
                )
                return cursor.rowcount > 0

    # ========================================================================
    # Message Operations
    # ========================================================================

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
        auto_update_title: bool = True,
    ) -> Dict[str, Any]:
        """Add a message to a conversation and update conversation timestamp.

        Args:
            conversation_id: Target conversation identifier.
            role: Message sender role ('user' or 'assistant').
            content: Non-empty message text.
            citations: Structured citations list or None.
            metadata: Additional structured metadata dictionary or None.
            message_id: Optional unique identifier or None (auto-generates UUID).
            auto_update_title: If True, derives a clean title on the first user message
                              if current title is 'New Conversation'.

        Returns:
            Structured message dictionary.

        Raises:
            ValueError: If conversation does not exist, role is invalid, or content is empty.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid message role '{role}'. Must be 'user' or 'assistant'.")

        if not isinstance(content, str) or not content.strip():
            raise ValueError("Message content cannot be empty or whitespace-only.")

        cleaned_content = content.strip()
        mid = message_id.strip() if isinstance(message_id, str) and message_id.strip() else str(uuid.uuid4())
        now = utcnow_iso()

        citations_json = json.dumps(citations) if citations is not None else None
        metadata_json = json.dumps(metadata) if metadata is not None else None

        with self._get_connection() as conn:
            # Verify conversation exists
            cursor = conn.execute(
                "SELECT title FROM conversations WHERE conversation_id = ?;",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise ValueError(f"Conversation '{conversation_id}' does not exist.")

            current_title = row["title"]

            with conn:
                # 1. Insert message
                conn.execute(
                    """
                    INSERT INTO messages (
                        message_id, conversation_id, role, content, created_at, citations, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (mid, conversation_id, role, cleaned_content, now, citations_json, metadata_json),
                )

                # 2. Update conversation updated_at
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;",
                    (now, conversation_id),
                )

                # 3. Optional auto-title on first user message
                if auto_update_title and role == "user" and current_title == "New Conversation":
                    derived_title = generate_conversation_title(cleaned_content)
                    conn.execute(
                        "UPDATE conversations SET title = ? WHERE conversation_id = ?;",
                        (derived_title, conversation_id),
                    )

        return {
            "message_id": mid,
            "conversation_id": conversation_id,
            "role": role,
            "content": cleaned_content,
            "created_at": now,
            "citations": citations if citations is not None else [],
            "metadata": metadata if metadata is not None else {},
        }

    def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve messages for a given conversation in ascending chronological order.

        Args:
            conversation_id: Conversation identifier.
            limit: Optional maximum number of messages to return.

        Returns:
            List of message dictionaries ordered chronologically.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            return []

        query = """
            SELECT message_id, conversation_id, role, content, created_at, citations, metadata
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC
        """
        params: List[Any] = [conversation_id]
        if limit is not None and isinstance(limit, int) and limit > 0:
            query += " LIMIT ?"
            params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            messages: List[Dict[str, Any]] = []
            for row in rows:
                citations_val = json.loads(row["citations"]) if row["citations"] else []
                metadata_val = json.loads(row["metadata"]) if row["metadata"] else {}
                messages.append({
                    "message_id": row["message_id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "citations": citations_val,
                    "metadata": metadata_val,
                })
            return messages

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most recent messages for RAG context, sorted chronologically.

        Fetches the latest `limit` messages from the conversation and returns them
        in ascending chronological order so conversation context flows naturally.

        Args:
            conversation_id: Target conversation identifier.
            limit: Maximum number of recent messages to return (default: 10).

        Returns:
            List of recent message dictionaries in chronological order.
        """
        if not conversation_id or not isinstance(conversation_id, str):
            return []

        if not isinstance(limit, int) or limit <= 0:
            limit = 10

        query = """
            SELECT message_id, conversation_id, role, content, created_at, citations, metadata
            FROM (
                SELECT message_id, conversation_id, role, content, created_at, citations, metadata
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
            ORDER BY created_at ASC;
        """

        with self._get_connection() as conn:
            cursor = conn.execute(query, (conversation_id, limit))
            rows = cursor.fetchall()

            messages: List[Dict[str, Any]] = []
            for row in rows:
                citations_val = json.loads(row["citations"]) if row["citations"] else []
                metadata_val = json.loads(row["metadata"]) if row["metadata"] else {}
                messages.append({
                    "message_id": row["message_id"],
                    "conversation_id": row["conversation_id"],
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"],
                    "citations": citations_val,
                    "metadata": metadata_val,
                })
            return messages

    def get_conversation_with_messages(
        self,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve conversation details along with all its messages in chronological order.

        Args:
            conversation_id: Conversation identifier.

        Returns:
            Dictionary with conversation metadata and 'messages' list, or None.
        """
        conv = self.get_conversation(conversation_id)
        if conv is None:
            return None

        conv["messages"] = self.get_messages(conversation_id)
        return conv

    # ========================================================================
    # Document Operations
    # ========================================================================

    def add_document(
        self,
        conversation_id: str,
        filename: str,
        file_type: str,
        total_pages: int = 1,
        total_characters: int = 0,
        chunk_count: int = 0,
        file_path: str = "",
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Associate an uploaded document record with a conversation."""
        doc_id = (
            document_id.strip()
            if isinstance(document_id, str) and document_id.strip()
            else str(uuid.uuid4())
        )
        now = utcnow_iso()

        with self._get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO conversation_documents 
                    (document_id, conversation_id, filename, file_type, total_pages, total_characters, chunk_count, file_path, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        doc_id,
                        conversation_id,
                        filename,
                        file_type,
                        total_pages,
                        total_characters,
                        chunk_count,
                        file_path,
                        now,
                    ),
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?;",
                    (now, conversation_id),
                )

        return {
            "document_id": doc_id,
            "conversation_id": conversation_id,
            "filename": filename,
            "file_type": file_type,
            "total_pages": total_pages,
            "total_characters": total_characters,
            "chunk_count": chunk_count,
            "file_path": file_path,
            "created_at": now,
        }

    def get_documents(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all documents associated with a conversation."""
        if not conversation_id or not isinstance(conversation_id, str):
            return []

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT document_id, conversation_id, filename, file_type, total_pages, total_characters, chunk_count, file_path, created_at
                FROM conversation_documents
                WHERE conversation_id = ?
                ORDER BY created_at ASC;
                """,
                (conversation_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, conversation_id: str, document_id: str) -> bool:
        """Remove an associated document record from a conversation."""
        if not conversation_id or not document_id:
            return False

        with self._get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversation_documents WHERE conversation_id = ? AND document_id = ?;",
                    (conversation_id, document_id),
                )
                return cursor.rowcount > 0

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents across all conversations."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT document_id, conversation_id, filename, file_type, total_pages, total_characters, chunk_count, file_path, created_at
                FROM conversation_documents
                ORDER BY created_at ASC;
                """
            )
            return [dict(row) for row in cursor.fetchall()]

