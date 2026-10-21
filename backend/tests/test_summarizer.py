"""Automated unit tests for IntelliAssist AI TextSummarizer module.

Verifies:
  1. Concise summarization reduces text and captures core essence.
  2. Detailed summarization generates structured, multi-point overview.
  3. Concise vs detailed length comparison (concise <= detailed).
  4. Single sentence text handling (returns text safely without distortion).
  5. Empty and whitespace-only input validation (raises ValueError).
  6. None input validation (raises ValueError).
  7. Non-string input validation (raises TypeError).
  8. Safe error mode without raising exceptions (raise_errors=False).
  9. Invalid summary_type rejection.
  10. Abstractive LLM-based summarization with BaseLLMClient.
  11. Transparent fallback to extractive summarizer on LLM generation failure.
  12. Custom max_sentences parameter enforcement.
  13. SummaryResult dictionary access and schema serialization.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.rag_pipeline import MockLLMClient
from app.summarizer import (
    DEFAULT_SUMMARY_TYPE,
    ExtractiveSummarizer,
    SummaryResult,
    SummaryType,
    TextSummarizer,
    split_into_sentences,
    summarize_text,
    validate_summarization_input,
)


# ============================================================================
# Sample Corpus Fixtures
# ============================================================================

SAMPLE_SECURITY_TEXT = (
    "All employees must maintain high security standards across corporate devices. "
    "Passwords must be at least 14 characters long and include numbers and special characters. "
    "Passwords must expire every 90 days to prevent unauthorized credential reuse. "
    "Multi-factor authentication (MFA) is strictly required for all administrative access. "
    "Employees must never write down credentials or share accounts under any circumstances. "
    "Any suspected security breach must be reported to the IT Security Team within one hour. "
    "Regular compliance audits are conducted quarterly across all engineering departments."
)

SAMPLE_SHORT_TEXT = "IntelliAssist AI provides document intelligence and conversational retrieval."


# ============================================================================
# 1. Extractive & Core Summarization Tests
# ============================================================================

class TestExtractiveSummarizer:
    """Verifies deterministic local extractive summarization."""

    def test_concise_summarization_reduces_text(self) -> None:
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="concise")

        assert isinstance(result, SummaryResult)
        assert result.status == "success"
        assert result.summary_type == "concise"
        assert result.method == "extractive"
        assert result.original_length == len(SAMPLE_SECURITY_TEXT)
        assert result.summary_length < result.original_length
        assert result.reduction_ratio > 0.0
        assert result.summary_sentence_count <= 2
        assert len(result.summary) > 0

    def test_detailed_summarization_preserves_structure(self) -> None:
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="detailed")

        assert result.status == "success"
        assert result.summary_type == "detailed"
        assert result.summary_sentence_count >= 2
        assert result.summary_sentence_count > 1
        # Detailed mode formats bullet points when selecting 3 or more sentences
        if result.summary_sentence_count >= 3:
            assert "-" in result.summary

    def test_concise_vs_detailed_length_comparison(self) -> None:
        summarizer = ExtractiveSummarizer()
        concise = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="concise")
        detailed = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="detailed")

        assert concise.summary_length <= detailed.summary_length
        assert concise.summary_sentence_count <= detailed.summary_sentence_count

    def test_single_sentence_text_preserved(self) -> None:
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(SAMPLE_SHORT_TEXT, summary_type="concise")

        assert result.status == "success"
        assert result.summary == SAMPLE_SHORT_TEXT
        assert result.original_sentence_count == 1
        assert result.summary_sentence_count == 1
        assert result.reduction_ratio == 0.0

    def test_custom_max_sentences_override(self) -> None:
        summarizer = ExtractiveSummarizer()
        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, max_sentences=1)
        assert result.summary_sentence_count == 1


# ============================================================================
# 2. Input Validation & Error Handling Tests
# ============================================================================

class TestSummarizerValidation:
    """Verifies safe input validation and exception contracts."""

    def test_empty_and_whitespace_input_raises_value_error(self) -> None:
        summarizer = TextSummarizer()

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            summarizer.summarize("")

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            summarizer.summarize("     \n\t   ")

    def test_none_input_raises_value_error(self) -> None:
        summarizer = TextSummarizer()
        with pytest.raises(ValueError, match="cannot be None"):
            summarizer.summarize(None)

    def test_non_string_input_raises_type_error(self) -> None:
        summarizer = TextSummarizer()

        with pytest.raises(TypeError, match="must be a string"):
            summarizer.summarize(12345)

        with pytest.raises(TypeError, match="must be a string"):
            summarizer.summarize(["Text in a list"])

    def test_invalid_summary_type_rejected(self) -> None:
        summarizer = TextSummarizer()

        with pytest.raises(ValueError, match="Invalid summary_type"):
            summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="unsupported_mode")

    def test_safe_error_mode_without_raising(self) -> None:
        summarizer = TextSummarizer()

        # Empty string with raise_errors=False
        res_empty = summarizer.summarize("", raise_errors=False)
        assert res_empty.status == "error"
        assert res_empty.error is not None
        assert res_empty.summary == ""

        # None with raise_errors=False
        res_none = summarizer.summarize(None, raise_errors=False)
        assert res_none.status == "error"

        # Invalid type with raise_errors=False
        res_type = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="invalid_style", raise_errors=False)
        assert res_type.status == "error"


# ============================================================================
# 3. LLM Integration & Fallback Tests
# ============================================================================

class TestTextSummarizerWithLLM:
    """Verifies LLM-based abstractive generation and graceful fallback."""

    def test_llm_based_summarization_invokes_client(self) -> None:
        mock_llm = MockLLMClient(
            default_response="Passwords must be 14 characters and expire every 90 days."
        )
        summarizer = TextSummarizer(llm_client=mock_llm)

        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="concise")

        assert result.status == "success"
        assert result.method == "llm"
        assert "14 characters" in result.summary
        assert mock_llm.call_count == 1
        assert "CONCISE" in mock_llm.last_call["prompt"]

    def test_detailed_llm_summarization_prompt_construction(self) -> None:
        mock_llm = MockLLMClient(
            default_response="- Minimum 14 characters\n- 90 day expiration\n- Mandatory MFA"
        )
        summarizer = TextSummarizer(llm_client=mock_llm)

        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="detailed")

        assert result.status == "success"
        assert result.method == "llm"
        assert mock_llm.call_count == 1
        assert "DETAILED" in mock_llm.last_call["prompt"]

    def test_llm_failure_falls_back_to_extractive(self) -> None:
        failing_llm = MockLLMClient()

        def fail_gen(s, p):
            raise RuntimeError("Cloud LLM rate limited")

        failing_llm.response_generator = fail_gen
        summarizer = TextSummarizer(llm_client=failing_llm)

        # Should NOT raise; transparently falls back to local extractive engine
        result = summarizer.summarize(SAMPLE_SECURITY_TEXT, summary_type="concise")

        assert result.status == "success"
        assert result.method == "extractive"
        assert len(result.summary) > 0


# ============================================================================
# 4. Functional Interface & Serialization Tests
# ============================================================================

class TestSummarizerHelpers:
    """Verifies functional helper summarize_text() and SummaryResult data model."""

    def test_functional_summarize_text_helper(self) -> None:
        result = summarize_text(SAMPLE_SECURITY_TEXT, summary_type="concise")
        assert isinstance(result, SummaryResult)
        assert result.status == "success"

    def test_summary_result_dictionary_access(self) -> None:
        result = summarize_text(SAMPLE_SECURITY_TEXT)
        d = result.to_dict()

        assert isinstance(d, dict)
        assert "summary" in d
        assert "reduction_ratio" in d

        # Subscript access
        assert result["summary"] == d["summary"]
        assert result.get("non_existent_key", "default_val") == "default_val"

    def test_split_into_sentences_handles_newlines_and_bullets(self) -> None:
        multiline = (
            "First sentence here.\n\n"
            "Second sentence follows.\n"
            "* Third bullet sentence.\n"
            "4. Fourth numbered sentence."
        )
        sentences = split_into_sentences(multiline)
        assert len(sentences) == 4
        assert sentences[0] == "First sentence here."
        assert sentences[1] == "Second sentence follows."
        assert sentences[2] == "Third bullet sentence."
        assert sentences[3] == "Fourth numbered sentence."
