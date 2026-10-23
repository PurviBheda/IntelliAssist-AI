"""IntelliAssist AI - Unit Tests for GeminiLLMClient and Gemini Integration.

These tests run 100% offline with mocked Google GenAI client dependencies.
Zero live API tokens, network calls, or paid resources are required.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from app.rag_pipeline import (
    DEFAULT_GEMINI_MODEL,
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    RAGPipeline,
    RAGResponse,
    build_retrieval_query,
    is_followup_query,
)
from app.summarizer import TextSummarizer
from app.vector_store import FAISSVectorStore
from app.embedding_service import EmbeddingService


class TestGeminiLLMClientConfiguration:
    """Test GeminiLLMClient initialization and configuration loading."""

    def test_missing_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client raises clear ValueError when no API key is provided."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="Gemini API key is missing"):
            GeminiLLMClient(api_key=None)

    def test_placeholder_api_key_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client raises ValueError when key is set to a template placeholder."""
        monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")

        with pytest.raises(ValueError, match="Gemini API key is missing or set to a placeholder"):
            GeminiLLMClient()

    def test_reads_api_key_from_gemini_api_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client resolves key from GEMINI_API_KEY environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-12345")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with patch("google.genai.Client") as mock_genai:
            client = GeminiLLMClient()
            assert client._api_key == "test-secret-key-12345"
            mock_genai.assert_called_once()
            call_kwargs = mock_genai.call_args.kwargs
            assert call_kwargs["api_key"] == "test-secret-key-12345"

    def test_reads_api_key_from_google_api_key_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client falls back to GOOGLE_API_KEY when GEMINI_API_KEY is unset."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key-999")

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            assert client._api_key == "test-google-key-999"

    def test_explicit_api_key_overrides_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicitly passed api_key takes precedence over environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "env-key")

        with patch("google.genai.Client"):
            client = GeminiLLMClient(api_key="explicit-key")
            assert client._api_key == "explicit-key"

    def test_default_model_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default model name is DEFAULT_GEMINI_MODEL when GEMINI_MODEL is not set."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.delenv("GEMINI_MODEL", raising=False)

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            assert client.model_name == DEFAULT_GEMINI_MODEL

    def test_model_name_from_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model name is loaded from GEMINI_MODEL when not passed to constructor."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            assert client.model_name == "gemini-2.0-flash"

    def test_explicit_model_name_overrides_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor model_name argument overrides GEMINI_MODEL environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")

        with patch("google.genai.Client"):
            client = GeminiLLMClient(model_name="custom-gemini-pro")
            assert client.model_name == "custom-gemini-pro"


class TestGeminiLLMClientTimeout:
    """Test timeout configuration in GeminiLLMClient."""

    def test_default_timeout_is_30_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default timeout is 30,000 milliseconds."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.delenv("GEMINI_TIMEOUT_MS", raising=False)
        monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)

        with patch("google.genai.Client") as mock_genai:
            client = GeminiLLMClient()
            assert client.timeout_ms == 30000
            http_opts = mock_genai.call_args.kwargs["http_options"]
            assert http_opts.timeout == 30000

    def test_timeout_from_env_seconds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Timeout in seconds from GEMINI_TIMEOUT_SECONDS is converted to ms."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "45")

        with patch("google.genai.Client") as mock_genai:
            client = GeminiLLMClient()
            assert client.timeout_ms == 45000
            http_opts = mock_genai.call_args.kwargs["http_options"]
            assert http_opts.timeout == 45000

    def test_explicit_timeout_arg_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constructor timeout_ms argument overrides environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.setenv("GEMINI_TIMEOUT_SECONDS", "45")

        with patch("google.genai.Client") as mock_genai:
            client = GeminiLLMClient(timeout_ms=15000)
            assert client.timeout_ms == 15000
            http_opts = mock_genai.call_args.kwargs["http_options"]
            assert http_opts.timeout == 15000


class TestGeminiLLMClientSanitization:
    """Test that API keys and secrets are never leaked into error messages."""

    def test_sanitize_error_redacts_instance_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redacts the configured API key from error text."""
        monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key-xyz")

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            err_msg = "Error connecting with key super-secret-key-xyz at api.endpoint"
            sanitized = client._sanitize_error(err_msg)
            assert "super-secret-key-xyz" not in sanitized
            assert "[REDACTED_API_KEY]" in sanitized

    def test_sanitize_error_redacts_google_aiza_pattern(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redacts Google API key pattern even if different from instance key."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            err_msg = "Request failed for AIzaSyD1234567890abcdefghijklmnopqrstuv"
            sanitized = client._sanitize_error(err_msg)
            assert "AIzaSyD" not in sanitized
            assert "[REDACTED_API_KEY]" in sanitized

    def test_sanitize_error_redacts_url_query_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Redacts key parameter in URLs."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client"):
            client = GeminiLLMClient()
            err_msg = "HTTP 400 at https://generativelanguage.googleapis.com/v1beta/models?key=someRandomKey999&alt=json"
            sanitized = client._sanitize_error(err_msg)
            assert "someRandomKey999" not in sanitized
            assert "key=[REDACTED_API_KEY]" in sanitized


class TestGeminiLLMClientGeneration:
    """Test response extraction, fallback, and error handling in generate_response."""

    def test_generate_response_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Successfully extracts response text from Gemini response."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.output_text = "According to security_policy.txt, passwords must be 14 characters."
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()
            result = client.generate_response("You are an assistant", "What is password length?")
            assert "14 characters" in result

   
    def test_generate_response_fallback_to_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError when the Interactions API returns no output text."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_resp = MagicMock()
            mock_resp.output_text = ""
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()

            with pytest.raises(
                RuntimeError,
                match="Gemini API returned an empty or blocked response",
            ):
                client.generate_response("sys", "prompt")

      

    def test_generate_response_raises_on_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises RuntimeError when model returns empty or blocked response."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.output_text = " " 
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()
            with pytest.raises(RuntimeError, match="Gemini API returned an empty or blocked response"):
                client.generate_response("sys", "prompt")

    def test_generate_response_sanitizes_network_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Network or SDK exceptions are caught and sanitized."""
        monkeypatch.setenv("GEMINI_API_KEY", "secret-key-xyz")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.interactions.create.side_effect = ConnectionError(
                "Connection failed with key secret-key-xyz: 503 Unavailable" 
            )

            client = GeminiLLMClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.generate_response("sys", "prompt")

            err_str = str(exc_info.value)
            assert "secret-key-xyz" not in err_str
            assert "[REDACTED_API_KEY]" in err_str


class TestRAGPipelineEmptyLLMResponseHandling:
    """Test RAGPipeline behavior when LLM client returns empty or whitespace output."""

    def test_gemini_empty_response_results_in_pipeline_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When Gemini client returns empty/blocked text, it raises and RAGPipeline reports error status."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_resp = MagicMock() 
            mock_resp.output_text = "" 
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()

            store = FAISSVectorStore(dimension=384, normalize_vectors=True)
            store.add_chunks([{
                "chunk_id": "test_1",
                "text": "Some relevant policy text here.",
                "embedding": [0.1] * 384,
                "metadata": {"filename": "policy.txt", "page_number": 1},
            }])

            mock_emb = MagicMock(spec=EmbeddingService)
            mock_emb.embed_text.return_value = [0.1] * 384

            pipeline = RAGPipeline(
                vector_store=store,
                embedding_service=mock_emb,
                llm_client=client,
            )

            response = pipeline.answer("What is the policy?")
            assert response.generation_status == "error"
            assert response.sufficient_context is False
            assert "Unable to generate answer due to language model failure" in response.answer
            assert "empty or blocked response" in (response.error or "")

    def test_mock_empty_llm_treated_as_insufficient_context(self) -> None:
        """When a mock LLM returns empty text directly, it is safely treated as insufficient context."""
        empty_llm = MagicMock(spec=BaseLLMClient)
        empty_llm.model_name = "mock-empty"
        empty_llm.generate_response.return_value = ""

        store = FAISSVectorStore(dimension=384, normalize_vectors=True)
        store.add_chunks([{
            "chunk_id": "test_1",
            "text": "Some relevant policy text here.",
            "embedding": [0.1] * 384,
            "metadata": {"filename": "policy.txt", "page_number": 1},
        }])

        mock_emb = MagicMock(spec=EmbeddingService)
        mock_emb.embed_text.return_value = [0.1] * 384

        pipeline = RAGPipeline(
            vector_store=store,
            embedding_service=mock_emb,
            llm_client=empty_llm,
        )

        response = pipeline.answer("What is the policy?")
        assert response.generation_status == "insufficient_context"
        assert response.sufficient_context is False


class TestGeminiAPIErrorAndResilience:
    """Test detailed API error handling: timeouts, HTTP 429, HTTP 503, retries, and fallback extraction."""

    def test_generate_response_handles_api_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client properly surfaces and sanitizes API timeout errors."""
        monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.interactions.create.side_effect = TimeoutError(
                "Request timed out after 30000ms connecting with key secret-test-key"
            )

            client = GeminiLLMClient(timeout_ms=30000)
            with pytest.raises(RuntimeError) as exc_info:
                client.generate_response("system", "prompt")

            err_str = str(exc_info.value)
            assert "secret-test-key" not in err_str
            assert "[REDACTED_API_KEY]" in err_str
            assert "timed out" in err_str.lower()

    def test_generate_response_handles_http_429_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client surfaces HTTP 429 rate limit errors with status code tag and without secrets."""
        monkeypatch.setenv("GEMINI_API_KEY", "secret-key-429")

        class MockHttpError(Exception):
            status_code = 429

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.interactions.create.side_effect = MockHttpError(
                "RESOURCE_EXHAUSTED: Rate limit exceeded for key secret-key-429"
            )

            client = GeminiLLMClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.generate_response("system", "prompt")

            err_str = str(exc_info.value)
            assert "[HTTP 429]" in err_str
            assert "RESOURCE_EXHAUSTED" in err_str
            assert "secret-key-429" not in err_str

    def test_generate_response_handles_http_503_service_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client surfaces HTTP 503 unavailable / overloaded errors with status code tag."""
        monkeypatch.setenv("GEMINI_API_KEY", "secret-key-503")

        class MockServerError(Exception):
            status_code = 503

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.interactions.create.side_effect = MockServerError(
                "The model API is currently overloaded and may experience intermittent errors."
            )

            client = GeminiLLMClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.generate_response("system", "prompt")

            err_str = str(exc_info.value)
            assert "[HTTP 503]" in err_str
            assert "overloaded" in err_str.lower()

    def test_generate_response_fallback_extracts_from_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When output_text attribute is None, extracts text from response steps."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_resp = MagicMock()
            mock_resp.output_text = None
            mock_resp.errors = []
            mock_resp.status = "completed"
            mock_resp.steps = [
                {
                    "type": "thought",
                    "content": [{"type": "text", "text": "Thinking about policy..."}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Passwords must be at least 14 characters."}],
                },
            ]
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()
            result = client.generate_response("sys", "prompt")
            assert "14 characters" in result

    def test_generate_response_raises_on_interaction_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client raises RuntimeError when interaction response contains error objects."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")

        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            mock_resp = MagicMock()
            mock_resp.errors = [
                {"code": "SAFETY_BLOCKED", "message": "Content blocked by safety policy"},
            ]
            mock_client.interactions.create.return_value = mock_resp

            client = GeminiLLMClient()
            with pytest.raises(RuntimeError) as exc_info:
                client.generate_response("sys", "prompt")

            assert "SAFETY_BLOCKED" in str(exc_info.value)

    def test_gemini_client_max_retries_configured_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Client respects GEMINI_MAX_RETRIES environment variable."""
        monkeypatch.setenv("GEMINI_API_KEY", "valid-key")
        monkeypatch.setenv("GEMINI_MAX_RETRIES", "2")

        with patch("google.genai.Client") as mock_genai:
            client = GeminiLLMClient()
            assert client.max_retries == 2
            retry_opts = mock_genai.call_args.kwargs["http_options"].retry_options
            assert retry_opts.attempts == 2


class TestConversationAwareRetrievalAndFollowUps:
    """Test conversation-aware retrieval logic and follow-up resolution."""

    def test_standalone_question_does_not_incorporate_old_history(self) -> None:
        """Standalone questions do not append old conversation topics."""
        history = [
            {"role": "user", "content": "What is the software license cost?"},
            {"role": "assistant", "content": "According to doc_alpha.pdf, license fee is $1,200."},
        ]
        q = "What is the monthly internet stipend for remote staff?"
        query = build_retrieval_query(q, history)
        assert query == q
        assert "license" not in query
        assert "doc_alpha" not in query

    def test_followup_question_resolves_pronoun_context(self) -> None:
        """Follow-up questions with pronouns ('it', 'they') incorporate prior user query context."""
        history = [
            {"role": "user", "content": "What is the minimum required password length according to the security policy?"},
            {"role": "assistant", "content": "Passwords must be at least 14 characters long."},
        ]
        q = "How often must it be changed?"
        assert is_followup_query(q) is True

        query = build_retrieval_query(q, history)
        assert "How often must it be changed?" in query
        assert "Context: What is the minimum required password length" in query
        # Must NOT include assistant message content
        assert "Passwords must be at least 14 characters long" not in query


class TestSummarizerErrorPreservation:
    """Test summarization fallback error diagnostics."""

    def test_summarization_fallback_preserves_diagnostic_error(self) -> None:
        """When LLM summarization fails, extractive fallback preserves the underlying error note."""
        failing_llm = MockLLMClient()

        def fail_gen(s, p):
            raise RuntimeError("Gemini API rate limit: [HTTP 429] RESOURCE_EXHAUSTED")

        failing_llm.response_generator = fail_gen
        summarizer = TextSummarizer(llm_client=failing_llm)

        text = (
            "All employees must maintain high security standards across corporate devices. "
            "Passwords must be at least 14 characters long and include numbers and special characters. "
            "Passwords must expire every 90 days to prevent unauthorized credential reuse. "
            "Multi-factor authentication (MFA) is strictly required for all administrative access."
        )
        result = summarizer.summarize(text, summary_type="concise", prefer_llm=True)

        assert result.status == "success"
        assert result.method == "extractive"
        assert len(result.summary) > 0
        assert result.error is not None
        assert "LLM generation failed" in result.error
        assert "RESOURCE_EXHAUSTED" in result.error


