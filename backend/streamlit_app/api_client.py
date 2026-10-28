"""IntelliAssist AI - Streamlit FastAPI Client.

Independent HTTP client for communicating with the IntelliAssist AI FastAPI backend server.
Reuses existing API contracts without direct module imports or secondary state stores.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

logger = logging.getLogger("intelliassist.streamlit_client")

DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_CONNECT = 5.0
DEFAULT_TIMEOUT_READ = 60.0
DEFAULT_TIMEOUT = (DEFAULT_TIMEOUT_CONNECT, DEFAULT_TIMEOUT_READ)


# ============================================================================
# Custom User-Safe Exceptions
# ============================================================================

class ApiClientError(Exception):
    """Base exception for all IntelliAssist API client failures."""


class ApiConnectionError(ApiClientError):
    """Raised when the backend server is unreachable or times out."""


class ApiStatusError(ApiClientError):
    """Raised when the backend server responds with an HTTP error code (4xx, 5xx)."""

    def __init__(self, message: str, status_code: int, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


class ApiPayloadError(ApiClientError):
    """Raised when the backend returns malformed or unexpected data."""


# ============================================================================
# API Client Implementation
# ============================================================================

class IntelliAssistApiClient:
    """Client for interacting with the IntelliAssist AI FastAPI backend endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
        timeout: Union[float, Tuple[float, float]] = DEFAULT_TIMEOUT,
    ) -> None:
        """Initialize the API client.

        Args:
            base_url: Optional root URL of the backend. Defaults to INTELLIASSIST_API_URL or http://127.0.0.1:8000.
            session: Optional custom requests.Session for testing or connection pooling.
            timeout: Default timeout in seconds (connect, read).
        """
        raw_url = base_url or os.getenv("INTELLIASSIST_API_URL", DEFAULT_API_URL)
        self.base_url = raw_url.strip().rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def _url(self, path: str) -> str:
        """Construct full URL from path ensuring single slash joining."""
        clean_path = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{clean_path}"

    def _handle_response(self, response: requests.Response) -> Any:
        """Validate response status and return parsed JSON payload."""
        if not response.ok:
            error_detail = None
            try:
                err_data = response.json()
                if isinstance(err_data, dict):
                    error_detail = err_data.get("detail")
            except Exception:
                error_detail = response.text.strip() or response.reason

            user_msg = error_detail or f"HTTP {response.status_code} error from server"
            raise ApiStatusError(
                message=f"Request failed ({response.status_code}): {user_msg}",
                status_code=response.status_code,
                detail=error_detail,
            )

        try:
            return response.json()
        except Exception as e:
            raise ApiPayloadError(f"Failed to decode server JSON response: {e}") from e

    def _safe_request(self, method: str, path: str, **kwargs) -> Any:
        """Execute HTTP request with safe error wrapping and timeouts."""
        url = self._url(path)
        if "timeout" not in kwargs:
            kwargs["timeout"] = self.timeout

        try:
            res = self.session.request(method=method, url=url, **kwargs)
            return self._handle_response(res)
        except requests.exceptions.Timeout as e:
            raise ApiConnectionError(
                f"Connection to IntelliAssist backend at {self.base_url} timed out. "
                "Please verify that the backend server is running."
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ApiConnectionError(
                f"Could not connect to IntelliAssist backend at {self.base_url}. "
                "Please ensure python server.py is running on port 8000."
            ) from e
        except requests.exceptions.RequestException as e:
            raise ApiClientError(f"HTTP request error: {e}") from e

    # ------------------------------------------------------------------------
    # Endpoint Methods
    # ------------------------------------------------------------------------

    def check_health(self, timeout: Optional[float] = 5.0) -> Dict[str, Any]:
        """Check backend server health and readiness.

        Calls: GET /api/health
        Returns: Health status dictionary including 'status', 'is_live_llm', 'indexed_chunks'.
        """
        data = self._safe_request("GET", "/api/health", timeout=timeout or self.timeout)
        if not isinstance(data, dict) or "status" not in data:
            raise ApiPayloadError("Unexpected health check response schema.")
        return data

    def is_available(self, timeout: float = 3.0) -> bool:
        """Convenience method checking whether backend server is reachable and healthy."""
        try:
            data = self.check_health(timeout=timeout)
            return data.get("status") == "healthy"
        except Exception:
            return False

    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        starred: bool = False,
        shared: bool = False,
        timeout: Optional[float] = 10.0,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversation threads.

        Calls: GET /api/conversations
        """
        params = {
            "limit": limit,
            "offset": offset,
            "starred": str(starred).lower(),
            "shared": str(shared).lower(),
        }
        data = self._safe_request("GET", "/api/conversations", params=params, timeout=timeout or self.timeout)
        if not isinstance(data, list):
            raise ApiPayloadError(f"Expected list of conversations, got: {type(data).__name__}")
        return data

    def create_conversation(
        self,
        title: Optional[str] = None,
        conversation_id: Optional[str] = None,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Create a new conversation thread.

        Calls: POST /api/conversations
        """
        payload: Dict[str, Any] = {}
        if title:
            payload["title"] = title
        if conversation_id:
            payload["conversation_id"] = conversation_id

        data = self._safe_request("POST", "/api/conversations", json=payload, timeout=timeout or self.timeout)
        if not isinstance(data, dict) or "conversation_id" not in data:
            raise ApiPayloadError("Failed to receive created conversation record.")
        return data

    def get_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Retrieve full details of a conversation thread, including messages and documents.

        Calls: GET /api/conversations/{conversation_id}
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        data = self._safe_request(
            "GET",
            f"/api/conversations/{conversation_id}",
            timeout=timeout or self.timeout,
        )
        if not isinstance(data, dict) or "conversation_id" not in data:
            raise ApiPayloadError("Invalid conversation detail payload.")
        return data

    def delete_conversation(
        self,
        conversation_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Delete a conversation thread and its messages.

        Calls: DELETE /api/conversations/{conversation_id}
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        return self._safe_request(
            "DELETE",
            f"/api/conversations/{conversation_id}",
            timeout=timeout or self.timeout,
        )

    def star_conversation(
        self,
        conversation_id: str,
        is_starred: Optional[bool] = None,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Toggle or set starred status for a conversation thread.

        Calls: POST /api/conversations/{conversation_id}/star
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")

        payload = {"is_starred": is_starred} if is_starred is not None else {}
        return self._safe_request(
            "POST",
            f"/api/conversations/{conversation_id}/star",
            json=payload,
            timeout=timeout or self.timeout,
        )

    def upload_document(
        self,
        filename: str,
        file_bytes: Union[bytes, bytearray],
        conversation_id: Optional[str] = None,
        content_type: Optional[str] = None,
        timeout: Optional[float] = 60.0,
    ) -> Dict[str, Any]:
        """Upload and index a document (PDF, DOCX, TXT) into the vector store.

        Calls: POST /api/upload (multipart/form-data)
        """
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string.")
        if not file_bytes:
            raise ValueError("file_bytes cannot be empty.")

        mime = content_type or "application/octet-stream"
        files = {
            "file": (filename, file_bytes, mime),
        }
        data = {}
        if conversation_id:
            data["conversation_id"] = conversation_id

        result = self._safe_request(
            "POST",
            "/api/upload",
            files=files,
            data=data,
            timeout=timeout or self.timeout,
        )
        if not isinstance(result, dict) or "document" not in result:
            raise ApiPayloadError("Server response did not include uploaded document record.")
        return result

    def send_chat_message(
        self,
        conversation_id: str,
        message: str,
        auto_create_conversation: bool = True,
        timeout: Optional[float] = 90.0,
    ) -> Dict[str, Any]:
        """Send a prompt or document query to the AI assistant.

        Calls: POST /api/chat
        Returns: Complete ChatResponse dictionary containing:
            - 'assistant_response': Synthesized grounded answer
            - 'citations': Structured citations list (filename, page_number, similarity_score)
            - 'intent': Detected intent category
            - 'sentiment_result': Sentiment metrics if requested
            - 'summary_result': Structured summary if requested
            - 'sufficient_context': Evidence sufficiency flag
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")
        if not message or not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string.")

        payload = {
            "conversation_id": conversation_id.strip(),
            "message": message.strip(),
            "auto_create_conversation": auto_create_conversation,
        }
        result = self._safe_request(
            "POST",
            "/api/chat",
            json=payload,
            timeout=timeout or self.timeout,
        )
        if not isinstance(result, dict) or "assistant_response" not in result:
            raise ApiPayloadError("Server response did not include assistant response.")
        return result

    def remove_document(
        self,
        conversation_id: str,
        document_id: str,
        timeout: Optional[float] = 10.0,
    ) -> Dict[str, Any]:
        """Remove a document association from a conversation thread.

        Calls: DELETE /api/conversations/{conversation_id}/documents/{document_id}
        """
        if not conversation_id or not isinstance(conversation_id, str):
            raise ValueError("conversation_id must be a non-empty string.")
        if not document_id or not isinstance(document_id, str):
            raise ValueError("document_id must be a non-empty string.")

        return self._safe_request(
            "DELETE",
            f"/api/conversations/{conversation_id}/documents/{document_id}",
            timeout=timeout or self.timeout,
        )
