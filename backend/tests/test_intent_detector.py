"""Automated unit tests for IntelliAssist AI IntentDetector module.

Verifies:
  1. Recognition of each supported intent:
     - document_question
     - summarize
     - sentiment_analysis
     - greeting
     - out_of_scope
  2. Diverse phrasings of the same intent:
     - Summarization ('TL;DR', 'recap', 'brief summary', 'detailed overview').
     - Sentiment ('tone of', 'positive or negative', 'sentiment analysis of', 'vibe check').
     - Greetings ('hi', 'good morning', 'thanks', 'bye', 'who are you').
     - Out of scope ('write python code', 'recipe for cake', 'weather forecast', 'hack wifi').
  3. Parameter extraction:
     - summary_type ('concise' vs 'detailed').
     - summarization targets ('inline_text', 'conversation', 'document').
     - inline text extraction for summarization and sentiment.
     - greeting sub-types ('hello', 'gratitude', 'goodbye', 'chitchat').
  4. Ambiguous queries safely resolving to document_question fallback.
  5. Empty and whitespace-only input validation (raises ValueError).
  6. None input validation (raises ValueError).
  7. Non-string input validation (raises TypeError).
  8. Safe error mode without raising exceptions (raise_errors=False).
  9. Intent detection with optional EmbeddingService.
  10. IntentResult dictionary access and schema serialization.
"""

from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from app.embedding_service import EmbeddingService
from app.intent_detector import (
    IntentDetector,
    IntentResult,
    IntentType,
    detect_intent,
    validate_intent_input,
)


# ============================================================================
# Deterministic Embedding Helper Fixture
# ============================================================================

def make_deterministic_vec(text: str, dim: int = 384) -> np.ndarray:
    """Generate deterministic unit vector seeded by text hash."""
    import hashlib
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    service = EmbeddingService(embedding_dimension=384)
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda sentences, **kw: (
        make_deterministic_vec(sentences)
        if isinstance(sentences, str)
        else np.vstack([make_deterministic_vec(s) for s in sentences])
    )
    service._model = mock_model
    return service


# ============================================================================
# 1. Supported Intents Detection Tests
# ============================================================================

class TestSupportedIntentsDetection:
    """Verifies that all 5 primary intents are recognized accurately."""

    def test_document_question_intent(self) -> None:
        detector = IntentDetector()
        queries = [
            "What is the minimum required password length in the security policy?",
            "How many days of paid time off do full-time employees receive?",
            "What are the requirements for multi-factor authentication (MFA)?",
            "According to the handbook, what is the travel meal expense allowance?",
        ]
        for q in queries:
            res = detector.detect(q)
            assert res.intent == IntentType.DOCUMENT_QUESTION.value, f"Failed for query: {q}"
            assert res.confidence >= 0.70

    def test_summarize_intent(self) -> None:
        detector = IntentDetector()
        queries = [
            "Summarize the security policy document for me.",
            "Can you give me a brief summary of the vacation policy?",
            "TL;DR of the employee handbook.",
            "Provide a detailed summary of this text.",
        ]
        for q in queries:
            res = detector.detect(q)
            assert res.intent == IntentType.SUMMARIZE.value, f"Failed for query: {q}"
            assert res.confidence >= 0.85

    def test_sentiment_analysis_intent(self) -> None:
        detector = IntentDetector()
        queries = [
            "Analyze the sentiment of this feedback: 'The application is wonderful!'",
            "Is this customer review positive or negative?",
            "What is the tone of this message?",
            "Detect sentiment of: 'The service is terrible and slow.'",
        ]
        for q in queries:
            res = detector.detect(q)
            assert res.intent == IntentType.SENTIMENT_ANALYSIS.value, f"Failed for query: {q}"
            assert res.confidence >= 0.85

    def test_greeting_intent(self) -> None:
        detector = IntentDetector()
        queries = [
            "Hello!",
            "Hi there",
            "Good morning",
            "Thank you very much for your help!",
            "Goodbye, have a great day!",
            "Who are you?",
        ]
        for q in queries:
            res = detector.detect(q)
            assert res.intent == IntentType.GREETING.value, f"Failed for query: {q}"
            assert res.confidence >= 0.85

    def test_out_of_scope_intent(self) -> None:
        detector = IntentDetector()
        queries = [
            "Write a python script to scrape data from a web page.",
            "Create a javascript function to sort an array of objects.",
            "How do I hack into my neighbor's wifi network?",
            "What is the weather forecast in Tokyo tomorrow?",
            "Who won the 1994 World Cup final?",
        ]
        for q in queries:
            res = detector.detect(q)
            assert res.intent == IntentType.OUT_OF_SCOPE.value, f"Failed for query: {q}"
            assert res.confidence >= 0.80


# ============================================================================
# 2. Phrasing Variations Tests
# ============================================================================

class TestPhrasingVariations:
    """Verifies that alternative phrasings and idioms trigger expected intents."""

    def test_summarization_phrasings(self) -> None:
        detector = IntentDetector()
        variations = [
            ("tldr please", "concise"),
            ("Give me a quick recap of the policy", "concise"),
            ("Can you provide an in-depth, detailed summary?", "detailed"),
            ("Give me a comprehensive bullet-point summary", "detailed"),
        ]
        for text, expected_style in variations:
            res = detector.detect(text)
            assert res.intent == IntentType.SUMMARIZE.value
            assert res.parameters.get("summary_type") == expected_style

    def test_sentiment_phrasings(self) -> None:
        detector = IntentDetector()
        variations = [
            "vibe check on this message",
            "check the tone of this feedback",
            "Classify the sentiment of: 'Outstanding service!'",
            "What is the mood of this text?",
        ]
        for text in variations:
            res = detector.detect(text)
            assert res.intent == IntentType.SENTIMENT_ANALYSIS.value

    def test_greeting_subtypes(self) -> None:
        detector = IntentDetector()
        assert detector.detect("Hey there!").parameters.get("greeting_type") == "hello"
        assert detector.detect("Thanks a lot!").parameters.get("greeting_type") == "gratitude"
        assert detector.detect("See you later, bye!").parameters.get("greeting_type") == "goodbye"
        assert detector.detect("What can you do?").parameters.get("greeting_type") == "chitchat"


# ============================================================================
# 3. Parameter Extraction Tests
# ============================================================================

class TestParameterExtraction:
    """Verifies parameter extraction from user messages."""

    def test_inline_text_extracted_for_sentiment(self) -> None:
        detector = IntentDetector()
        text = "Analyze the sentiment of: 'The user interface is incredibly fast and responsive.'"
        res = detector.detect(text)

        assert res.intent == IntentType.SENTIMENT_ANALYSIS.value
        extracted = res.parameters.get("text_to_analyze")
        assert extracted is not None
        assert "user interface is incredibly fast" in extracted

    def test_inline_text_extracted_for_summarize(self) -> None:
        detector = IntentDetector()
        text = 'Summarize: "Company revenue grew 25% year-over-year. Operating costs decreased by 10%."'
        res = detector.detect(text)

        assert res.intent == IntentType.SUMMARIZE.value
        assert res.parameters.get("target") == "inline_text"
        assert "revenue grew 25%" in res.parameters.get("text_to_summarize", "")

    def test_conversation_target_extracted_for_summarize(self) -> None:
        detector = IntentDetector()
        text = "Can you summarize our conversation so far?"
        res = detector.detect(text)

        assert res.intent == IntentType.SUMMARIZE.value
        assert res.parameters.get("target") == "conversation"


# ============================================================================
# 4. Ambiguity Resolution & Safe Fallback Tests
# ============================================================================

class TestAmbiguityAndSafeFallback:
    """Verifies that ambiguous queries fall back safely to document_question."""

    def test_vague_query_falls_back_to_document_question(self) -> None:
        detector = IntentDetector(ambiguity_threshold=0.60)
        # Ambiguous vague phrase without clear intent triggers safe fallback
        res = detector.detect("Tell me about this")

        # Must fall back to document_question so RAG pipeline can evaluate document context
        assert res.intent == IntentType.DOCUMENT_QUESTION.value
        assert res.is_ambiguous is True or res.fallback_used is True

    def test_greeting_with_embedded_document_question(self) -> None:
        detector = IntentDetector()
        # Query starts with greeting but contains a document inquiry
        query = "Hello, what is the required password expiration period?"
        res = detector.detect(query)

        # Must NOT be classified as a pure greeting; must route to document_question
        assert res.intent == IntentType.DOCUMENT_QUESTION.value


# ============================================================================
# 5. Input Validation & Error Handling Tests
# ============================================================================

class TestIntentValidation:
    """Verifies input validation and exception contracts."""

    def test_empty_and_whitespace_input_raises_value_error(self) -> None:
        detector = IntentDetector()

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            detector.detect("")

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            detector.detect("   \n\t  ")

    def test_none_input_raises_value_error(self) -> None:
        detector = IntentDetector()
        with pytest.raises(ValueError, match="cannot be None"):
            detector.detect(None)

    def test_non_string_input_raises_type_error(self) -> None:
        detector = IntentDetector()

        with pytest.raises(TypeError, match="must be a string"):
            detector.detect(12345)

        with pytest.raises(TypeError, match="must be a string"):
            detector.detect({"query": "text"})

    def test_safe_error_mode_without_raising(self) -> None:
        detector = IntentDetector()

        res_empty = detector.detect("", raise_errors=False)
        assert res_empty.status == "error"
        assert res_empty.intent == IntentType.DOCUMENT_QUESTION.value
        assert res_empty.fallback_used is True

        res_none = detector.detect(None, raise_errors=False)
        assert res_none.status == "error"


# ============================================================================
# 6. Hybrid Detection with Embedding Service
# ============================================================================

class TestHybridIntentDetection:
    """Verifies intent detection with precomputed semantic exemplar embeddings."""

    def test_detection_with_mock_embedding_service(
        self, mock_embedding_service: EmbeddingService
    ) -> None:
        detector = IntentDetector(embedding_service=mock_embedding_service)

        res_doc = detector.detect("What is the company password policy?")
        assert res_doc.intent == IntentType.DOCUMENT_QUESTION.value

        res_greet = detector.detect("Good morning!")
        assert res_greet.intent == IntentType.GREETING.value

        res_sum = detector.detect("Give me a concise summary of the document")
        assert res_sum.intent == IntentType.SUMMARIZE.value

    def test_functional_detect_intent_helper(self) -> None:
        res = detect_intent("Hello!")
        assert isinstance(res, IntentResult)
        assert res.intent == IntentType.GREETING.value

    def test_intent_result_dictionary_access(self) -> None:
        res = detect_intent("What is the password policy?")
        d = res.to_dict()

        assert isinstance(d, dict)
        assert d["intent"] == IntentType.DOCUMENT_QUESTION.value
        assert "confidence" in d
        assert "parameters" in d

        # Subscript access
        assert res["intent"] == d["intent"]
        assert res.get("intent") == d["intent"]
        assert res.get("missing_key", "default") == "default"
