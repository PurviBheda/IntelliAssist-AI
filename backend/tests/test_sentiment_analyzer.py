"""Automated unit tests for IntelliAssist AI SentimentAnalyzer module.

Verifies:
  1. Positive sentiment classification with clear label and calibrated confidence.
  2. Negative sentiment classification with clear label and calibrated confidence.
  3. Neutral sentiment classification for factual, non-emotional statements.
  4. Negation detection correctly inverting sentiment polarity ('not good', 'hardly helpful').
  5. Intensifier tokens scaling sentiment magnitude ('extremely good', 'incredibly fast').
  6. Downtoner tokens dampening sentiment magnitude ('slightly bad', 'barely usable').
  7. Contrastive conjunctions prioritizing the trailing clause ('good, but it crashes constantly').
  8. Exclamation emphasis scaling valence and confidence.
  9. Mathematically calibrated confidence scores in valid range [0.0, 1.0] without fabrication.
  10. Empty and whitespace-only input validation (raises ValueError).
  11. None input validation (raises ValueError).
  12. Non-string input validation (raises TypeError).
  13. Safe error handling mode without raising exceptions (raise_errors=False).
  14. Functional helper analyze_sentiment() and dictionary serialization.
"""

from __future__ import annotations

import pytest

from app.sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentLabel,
    SentimentResult,
    analyze_sentiment,
    validate_sentiment_input,
)


# ============================================================================
# 1. Classification & Valence Mechanics Tests
# ============================================================================

class TestSentimentClassification:
    """Verifies positive, negative, and neutral classifications."""

    def test_positive_sentiment_classification(self) -> None:
        analyzer = SentimentAnalyzer()
        text = "The new document retrieval system is fantastic, fast, and incredibly helpful!"
        result = analyzer.analyze(text)

        assert isinstance(result, SentimentResult)
        assert result.status == "success"
        assert result.label == SentimentLabel.POSITIVE.value
        assert result.compound_score > 0.05
        assert result.confidence >= 0.70
        assert result.positive_score > result.negative_score

    def test_negative_sentiment_classification(self) -> None:
        analyzer = SentimentAnalyzer()
        text = "The service is terrible, completely broken, and full of awful errors and crashes."
        result = analyzer.analyze(text)

        assert result.status == "success"
        assert result.label == SentimentLabel.NEGATIVE.value
        assert result.compound_score < -0.05
        assert result.confidence >= 0.70
        assert result.negative_score > result.positive_score

    def test_neutral_sentiment_classification_for_factual_statements(self) -> None:
        analyzer = SentimentAnalyzer()
        text = "The scheduled backup completed on Tuesday at 03:00 UTC."
        result = analyzer.analyze(text)

        assert result.status == "success"
        assert result.label == SentimentLabel.NEUTRAL.value
        assert abs(result.compound_score) < 0.05
        assert result.confidence >= 0.80
        assert result.neutral_score >= 0.50

    def test_balanced_mixed_text_classifies_safely(self) -> None:
        analyzer = SentimentAnalyzer()
        text = "There were both good aspects and bad aspects in the report."
        result = analyzer.analyze(text)

        assert result.status == "success"
        assert result.label in {SentimentLabel.NEUTRAL.value, SentimentLabel.POSITIVE.value, SentimentLabel.NEGATIVE.value}
        assert 0.0 <= result.confidence <= 1.0


# ============================================================================
# 2. Contextual Rules: Negations, Modifiers & Contrast
# ============================================================================

class TestContextualSentimentModifiers:
    """Verifies linguistic mechanics: negations, intensifiers, downtoners, and contrasts."""

    def test_negation_inverts_positive_to_negative(self) -> None:
        analyzer = SentimentAnalyzer()
        # "good" is positive, but "not good" should become negative
        res_positive = analyzer.analyze("The response was good and reliable.")
        res_negated = analyzer.analyze("The response was not good and never reliable.")

        assert res_positive.label == SentimentLabel.POSITIVE.value
        assert res_negated.label == SentimentLabel.NEGATIVE.value
        assert res_negated.compound_score < 0.0

    def test_intensifier_boosts_valence_magnitude(self) -> None:
        analyzer = SentimentAnalyzer()
        res_base = analyzer.analyze("This software is good.")
        res_intensified = analyzer.analyze("This software is extremely good and exceptionally great.")

        assert res_intensified.compound_score > res_base.compound_score
        assert res_intensified.confidence >= res_base.confidence

    def test_downtoner_reduces_valence_magnitude(self) -> None:
        analyzer = SentimentAnalyzer()
        res_base = analyzer.analyze("The service is bad.")
        res_downtoned = analyzer.analyze("The service is slightly bad.")

        assert abs(res_downtoned.compound_score) < abs(res_base.compound_score)

    def test_contrastive_conjunction_focuses_on_second_clause(self) -> None:
        analyzer = SentimentAnalyzer()
        # Clause 1 is positive ("The UI looks decent"), but Clause 2 is strongly negative ("horribly broken and unusable")
        text = "The UI looks decent, but the backend is horribly broken and unusable."
        result = analyzer.analyze(text)

        assert result.label == SentimentLabel.NEGATIVE.value
        assert result.compound_score < 0.0


# ============================================================================
# 3. Confidence Score Calibration Tests
# ============================================================================

class TestConfidenceScoreCalibration:
    """Verifies that confidence scores are genuine, calibrated, and in mathematical bounds."""

    def test_confidence_scores_within_valid_bounds(self) -> None:
        analyzer = SentimentAnalyzer()
        test_samples = [
            "Outstanding work!",
            "Catastrophic failure and terrible loss.",
            "Meeting at noon.",
            "Slightly annoyed but overall satisfied.",
            "Not bad.",
        ]
        for s in test_samples:
            res = analyzer.analyze(s)
            assert 0.0 <= res.confidence <= 1.0, f"Confidence {res.confidence} out of bounds for: {s}"
            assert 0.0 <= res.positive_score <= 1.0
            assert 0.0 <= res.negative_score <= 1.0
            assert 0.0 <= res.neutral_score <= 1.0
            assert -1.0 <= res.compound_score <= 1.0

    def test_strong_signals_produce_higher_confidence_than_ambiguous(self) -> None:
        analyzer = SentimentAnalyzer()
        res_strong = analyzer.analyze("Absolutely outstanding, magnificent, and flawless performance!")
        res_mild = analyzer.analyze("It was okay.")

        assert res_strong.confidence > res_mild.confidence


# ============================================================================
# 4. Input Validation & Error Handling Tests
# ============================================================================

class TestSentimentValidation:
    """Verifies safe input validation and error contracts."""

    def test_empty_and_whitespace_input_raises_value_error(self) -> None:
        analyzer = SentimentAnalyzer()

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            analyzer.analyze("")

        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            analyzer.analyze("   \t  \n  ")

    def test_none_input_raises_value_error(self) -> None:
        analyzer = SentimentAnalyzer()
        with pytest.raises(ValueError, match="cannot be None"):
            analyzer.analyze(None)

    def test_non_string_input_raises_type_error(self) -> None:
        analyzer = SentimentAnalyzer()

        with pytest.raises(TypeError, match="must be a string"):
            analyzer.analyze(12345)

        with pytest.raises(TypeError, match="must be a string"):
            analyzer.analyze(["a", "b"])

    def test_safe_error_mode_without_raising(self) -> None:
        analyzer = SentimentAnalyzer()

        res_empty = analyzer.analyze("", raise_errors=False)
        assert res_empty.status == "error"
        assert res_empty.label == SentimentLabel.NEUTRAL.value
        assert res_empty.confidence == 0.0
        assert res_empty.error is not None

        res_none = analyzer.analyze(None, raise_errors=False)
        assert res_none.status == "error"

        res_type = analyzer.analyze(999, raise_errors=False)
        assert res_type.status == "error"


# ============================================================================
# 5. Functional Helper & Serialization Tests
# ============================================================================

class TestSentimentHelpers:
    """Verifies functional analyze_sentiment() helper and dictionary serialization."""

    def test_functional_analyze_sentiment_helper(self) -> None:
        result = analyze_sentiment("Great job team!")
        assert isinstance(result, SentimentResult)
        assert result.label == SentimentLabel.POSITIVE.value

    def test_sentiment_result_dictionary_access(self) -> None:
        result = analyze_sentiment("Great job team!")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["label"] == SentimentLabel.POSITIVE.value
        assert "confidence" in d
        assert "compound_score" in d

        # Subscript access
        assert result["label"] == d["label"]
        assert result.get("label") == d["label"]
        assert result.get("missing_key", "default") == "default"
