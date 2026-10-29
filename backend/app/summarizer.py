"""IntelliAssist AI - Text Summarization Module.

This module provides reusable, production-grade text summarization capabilities:
1. Support for 'concise' (executive/high-level) and 'detailed' (comprehensive/structured) summaries.
2. Extractive summarization using frequency-weighted sentence scoring and position-aware ranking
   (100% local, offline, deterministic, and requires no external API keys or heavy models).
3. Abstractive summarization via BaseLLMClient (Google Gemini or MockLLMClient) when available,
   with automatic, transparent fallback to the extractive engine on LLM errors.
4. Robust input validation and graceful error handling for empty, whitespace-only, or invalid inputs.
5. Rich metadata reporting including compression ratio, original length, summary length, and method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.rag_pipeline import BaseLLMClient


# ============================================================================
# Enums, Constants & Defaults
# ============================================================================

class SummaryType(str, Enum):
    """Supported summarization styles."""

    CONCISE = "concise"
    DETAILED = "detailed"


DEFAULT_SUMMARY_TYPE = SummaryType.CONCISE.value

# Standard English stopwords for frequency-based extractive scoring
STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}


# ============================================================================
# Response Model
# ============================================================================

@dataclass
class SummaryResult:
    """Structured result returned by summarization operations.

    Attributes:
        summary: The generated summary text.
        summary_type: The requested summary style ('concise' or 'detailed').
        original_length: Character length of the original input text.
        summary_length: Character length of the summary.
        original_sentence_count: Number of sentences in the original text.
        summary_sentence_count: Number of sentences in the summary.
        reduction_ratio: Percentage reduction in character length (0.0 to 1.0).
        method: Generation mechanism used ('extractive' or 'llm').
        status: Operation status ('success' or 'error').
        error: Error message string if summarization failed, else None.
    """

    summary: str
    summary_type: str
    original_length: int = 0
    summary_length: int = 0
    original_sentence_count: int = 0
    summary_sentence_count: int = 0
    reduction_ratio: float = 0.0
    method: str = "extractive"
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)

    def __getitem__(self, item: str) -> Any:
        """Allow subscript access for backward compatibility."""
        return asdict(self)[item]

    def get(self, item: str, default: Any = None) -> Any:
        """Allow get access for dictionary compatibility."""
        return asdict(self).get(item, default)


# ============================================================================
# Validation Helper
# ============================================================================

def validate_summarization_input(text: Any, summary_type: str = DEFAULT_SUMMARY_TYPE) -> Tuple[str, str]:
    """Validate input text and summary_type for summarization.

    Args:
        text: Input text string to be summarized.
        summary_type: Requested summary style ('concise' or 'detailed').

    Returns:
        Tuple of (cleaned_text, normalized_summary_type).

    Raises:
        ValueError: If text is None, empty, or whitespace-only, or summary_type is invalid.
        TypeError: If text is not a string.
    """
    if text is None:
        raise ValueError("Text to summarize cannot be None.")
    if not isinstance(text, str):
        raise TypeError(f"Text to summarize must be a string, got: {type(text).__name__}")

    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text to summarize cannot be empty or whitespace-only.")

    if not isinstance(summary_type, str):
        raise ValueError(f"summary_type must be a string, got: {type(summary_type).__name__}")

    normalized_type = summary_type.strip().lower()
    valid_types = {SummaryType.CONCISE.value, SummaryType.DETAILED.value}
    if normalized_type not in valid_types:
        raise ValueError(
            f"Invalid summary_type: {summary_type!r}. Must be one of: {sorted(valid_types)}"
        )

    return cleaned, normalized_type


# ============================================================================
# Sentence Splitting and Tokenization Helpers
# ============================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences while respecting common punctuation and abbreviations.

    Handles periods, question marks, exclamation marks, and newline breaks.
    Preserves complete, coherent sentences.
    """
    if not text or not text.strip():
        return []

    # Normalize paragraph breaks
    normalized = re.sub(r"\r\n|\r", "\n", text)

    # Split on double newlines (paragraphs) and sentence-ending punctuation
    # Pattern looks for . ! ? followed by whitespace/newline or end of text
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n{2,}|\n(?=[A-Z0-9\-\*\#])", normalized)

    sentences: List[str] = []
    for s in raw_sentences:
        cleaned = s.strip()
        # Clean bullet markers or numbering from the start of standalone lines
        cleaned = re.sub(r"^[\*\-\•\d\.\)]+\s*", "", cleaned).strip()
        if cleaned:
            sentences.append(cleaned)

    # Fallback if regex split didn't find multiple sentences but text has newlines
    if len(sentences) <= 1 and "\n" in normalized:
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        if len(lines) > len(sentences):
            sentences = lines

    return sentences


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"\b[a-zA-Z0-9_'\-]+\b", text.lower())


# ============================================================================
# Extractive Summarizer Implementation
# ============================================================================

class ExtractiveSummarizer:
    """Local, offline extractive summarizer using genuine TF-IDF sentence salience.

    Features:
    - Fits scikit-learn TfidfVectorizer over document sentences with sublinear TF scaling.
    - Computes sentence-to-document centroid cosine similarity in sparse TF-IDF space.
    - Combines centroid semantic alignment with length-normalized term salience.
    - Applies positional weighting (first sentence and introductory sentences receive a boost).
    - Preserves chronological sequence in the generated summary.
    - Handles short documents, repeated terms, uncommon terms, and all-stopword inputs safely.
    - Completely deterministic and fast.
    """

    def summarize(
        self,
        text: str,
        summary_type: str = DEFAULT_SUMMARY_TYPE,
        max_sentences: Optional[int] = None,
    ) -> SummaryResult:
        """Generate an extractive summary of text using TF-IDF sentence salience.

        Args:
            text: Input text string.
            summary_type: 'concise' or 'detailed'.
            max_sentences: Optional override for sentence count.

        Returns:
            Structured SummaryResult.
        """
        cleaned_text, norm_type = validate_summarization_input(text, summary_type)
        sentences = split_into_sentences(cleaned_text)

        orig_len = len(cleaned_text)
        total_sentences = len(sentences)

        # Edge case: If text has 1 or 2 sentences, it is already concise
        if total_sentences <= 1:
            return SummaryResult(
                summary=cleaned_text,
                summary_type=norm_type,
                original_length=orig_len,
                summary_length=orig_len,
                original_sentence_count=1,
                summary_sentence_count=1,
                reduction_ratio=0.0,
                method="extractive",
                status="success",
            )

        # Genuine TF-IDF vectorization across sentences
        try:
            vectorizer = TfidfVectorizer(
                token_pattern=r"(?u)\b\w+\b",
                lowercase=True,
                stop_words="english",
                sublinear_tf=True,
                norm="l2",
            )
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Fallback if text contains only stopwords or punctuation
            try:
                vectorizer = TfidfVectorizer(
                    token_pattern=r"(?u)\b\w+\b",
                    lowercase=True,
                    stop_words=None,
                    sublinear_tf=True,
                    norm="l2",
                )
                tfidf_matrix = vectorizer.fit_transform(sentences)
            except Exception:
                tfidf_matrix = None

        if tfidf_matrix is not None and tfidf_matrix.shape[1] > 0:
            # 1. Document centroid in TF-IDF space
            centroid = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            centroid_norm = float(np.linalg.norm(centroid))
            if centroid_norm > 0:
                centroid = centroid / centroid_norm
            centroid_sims = (tfidf_matrix * centroid).ravel()

            # 2. Length-normalized TF-IDF term salience
            sum_tfidf = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
            sentence_word_lens = [
                math.sqrt(max(1, len(tokenize_words(s)))) for s in sentences
            ]
            term_salience = sum_tfidf / sentence_word_lens
            max_sal = float(np.max(term_salience)) if len(term_salience) > 0 and np.max(term_salience) > 0 else 1.0

            # 3. Combined TF-IDF score
            base_scores = 0.6 * centroid_sims + 0.4 * (term_salience / max_sal)
        else:
            # Safe fallback if vectorization yields 0 features
            base_scores = np.ones(total_sentences, dtype=float)

        # Sentence scoring with positional weighting
        scored_sentences: List[Tuple[int, str, float]] = []
        for idx, sentence in enumerate(sentences):
            sent_words = tokenize_words(sentence)
            if not sent_words:
                continue

            score = float(base_scores[idx])

            # Positional weighting: First sentence gets 1.35x boost, second gets 1.15x
            if idx == 0:
                score *= 1.35
            elif idx == 1:
                score *= 1.15
            elif idx == total_sentences - 1 and total_sentences > 3:
                # Concluding sentence slight boost
                score *= 1.10

            scored_sentences.append((idx, sentence, score))

        if not scored_sentences:
            return SummaryResult(
                summary=cleaned_text,
                summary_type=norm_type,
                original_length=orig_len,
                summary_length=orig_len,
                original_sentence_count=total_sentences,
                summary_sentence_count=total_sentences,
                reduction_ratio=0.0,
                method="extractive",
                status="success",
            )

        # Determine target sentence count based on summary_type
        if max_sentences is not None and max_sentences > 0:
            target_count = min(max_sentences, total_sentences)
        elif norm_type == SummaryType.CONCISE.value:
            # Concise: 1 to 2 sentences for short texts, up to 3 for long texts
            if total_sentences <= 3:
                target_count = 1
            elif total_sentences <= 6:
                target_count = 2
            else:
                target_count = min(3, max(2, int(total_sentences * 0.25)))
        else:
            # Detailed: 40-50% of sentences, at least 3 (if available) up to 8
            target_count = max(2, min(total_sentences, max(3, int(total_sentences * 0.45))))

        # Rank sentences by score
        ranked = sorted(scored_sentences, key=lambda x: x[2], reverse=True)
        top_selected = ranked[:target_count]

        # Restore chronological order
        chronological_selected = sorted(top_selected, key=lambda x: x[0])
        selected_texts = [item[1] for item in chronological_selected]

        # Format output
        if norm_type == SummaryType.DETAILED.value and len(selected_texts) >= 3:
            # Provide structured bullet-point presentation for detailed summaries
            summary_body = "\n".join(f"- {s}" for s in selected_texts)
        else:
            summary_body = " ".join(selected_texts)

        summary_len = len(summary_body)
        reduction = max(0.0, round(1.0 - (summary_len / orig_len), 4)) if orig_len > 0 else 0.0

        return SummaryResult(
            summary=summary_body,
            summary_type=norm_type,
            original_length=orig_len,
            summary_length=summary_len,
            original_sentence_count=total_sentences,
            summary_sentence_count=len(selected_texts),
            reduction_ratio=reduction,
            method="extractive",
            status="success",
        )


# ============================================================================
# Reusable Text Summarizer (Hybrid Extractive + LLM)
# ============================================================================

class TextSummarizer:
    """Production-grade reusable summarizer supporting extractive and LLM-based summarization."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        default_summary_type: str = DEFAULT_SUMMARY_TYPE,
    ) -> None:
        """Initialize the TextSummarizer.

        Args:
            llm_client: Optional language model client (e.g. GeminiLLMClient or MockLLMClient).
                        If None, uses the fast, deterministic local ExtractiveSummarizer.
            default_summary_type: Default summary style ('concise' or 'detailed').
        """
        self.llm_client = llm_client
        self.default_summary_type = default_summary_type
        self.extractive_engine = ExtractiveSummarizer()

    def _build_llm_prompts(self, text: str, summary_type: str) -> Tuple[str, str]:
        """Construct system instructions and user prompt for LLM summarization."""
        if summary_type == SummaryType.CONCISE.value:
            system_instruction = (
                "You are an expert summarization assistant. Your task is to provide a concise, high-level "
                "summary of the provided text. Focus exclusively on the most critical takeaways in 1 to 3 "
                "well-crafted sentences. Do not introduce outside knowledge or fabricate facts."
            )
            prompt = (
                f"Please provide a CONCISE summary of the following text:\n\n"
                f"TEXT:\n{text}\n\n"
                f"INSTRUCTION:\nSummarize in 1-3 sentences capturing the core essence."
            )
        else:
            system_instruction = (
                "You are an expert summarization assistant. Your task is to provide a detailed, comprehensive, "
                "and well-structured summary of the provided text. Cover all key sections, significant facts, "
                "and major arguments, organized clearly with concise bullet points or structured paragraphs. "
                "Do not introduce outside knowledge or fabricate facts."
            )
            prompt = (
                f"Please provide a DETAILED, comprehensive summary of the following text:\n\n"
                f"TEXT:\n{text}\n\n"
                f"INSTRUCTION:\nProvide a structured, multi-point overview covering all key facts and details."
            )
        return system_instruction, prompt

    def summarize(
        self,
        text: Any,
        summary_type: Optional[str] = None,
        max_sentences: Optional[int] = None,
        prefer_llm: bool = True,
        raise_errors: bool = True,
    ) -> SummaryResult:
        """Summarize text according to the requested style.

        Args:
            text: Input text to summarize.
            summary_type: 'concise' or 'detailed' (defaults to instance default).
            max_sentences: Optional sentence count cap for extractive engine.
            prefer_llm: If True and llm_client is configured, attempts LLM summarization
                        before falling back to extractive.
            raise_errors: If True, raises ValueError/TypeError on invalid input; if False,
                          returns a safe error SummaryResult.

        Returns:
            Structured SummaryResult.

        Raises:
            ValueError: If text is empty/whitespace or summary_type is invalid and raise_errors is True.
            TypeError: If text is not a string and raise_errors is True.
        """
        chosen_type = summary_type if summary_type is not None else self.default_summary_type

        # 1. Input validation
        try:
            cleaned_text, norm_type = validate_summarization_input(text, chosen_type)
        except Exception as e:
            if raise_errors:
                raise
            return SummaryResult(
                summary="",
                summary_type=str(chosen_type),
                original_length=len(str(text)) if text is not None else 0,
                summary_length=0,
                status="error",
                error=str(e),
                method="none",
            )

        orig_len = len(cleaned_text)
        orig_sentences = split_into_sentences(cleaned_text)

        # 2. If text is extremely short (e.g. <= 1 sentence), return directly
        if len(orig_sentences) <= 1:
            return SummaryResult(
                summary=cleaned_text,
                summary_type=norm_type,
                original_length=orig_len,
                summary_length=orig_len,
                original_sentence_count=1,
                summary_sentence_count=1,
                reduction_ratio=0.0,
                method="extractive",
                status="success",
            )

        # 3. Attempt LLM generation if available and preferred
        llm_fallback_error: Optional[str] = None
        if prefer_llm and self.llm_client is not None:
            try:
                sys_inst, prompt = self._build_llm_prompts(cleaned_text, norm_type)
                generated_summary = self.llm_client.generate_response(
                    system_instruction=sys_inst,
                    prompt=prompt,
                )
                generated_clean = generated_summary.strip()
                if generated_clean:
                    sum_len = len(generated_clean)
                    reduction = max(0.0, round(1.0 - (sum_len / orig_len), 4)) if orig_len > 0 else 0.0
                    sum_sentences = split_into_sentences(generated_clean)

                    return SummaryResult(
                        summary=generated_clean,
                        summary_type=norm_type,
                        original_length=orig_len,
                        summary_length=sum_len,
                        original_sentence_count=len(orig_sentences),
                        summary_sentence_count=len(sum_sentences),
                        reduction_ratio=reduction,
                        method="llm",
                        status="success",
                    )
            except Exception as e:
                # Fall back gracefully to extractive engine on LLM failure,
                # recording diagnostic note so callers know why fallback occurred
                llm_fallback_error = f"LLM generation failed ({e}); fell back to extractive engine"

        # 4. Fallback or direct extractive summarization
        extractive_result = self.extractive_engine.summarize(
            text=cleaned_text,
            summary_type=norm_type,
            max_sentences=max_sentences,
        )
        if llm_fallback_error and extractive_result.status == "success":
            extractive_result.error = llm_fallback_error
        return extractive_result


# ============================================================================
# Functional Public Interface
# ============================================================================

def summarize_text(
    text: Any,
    summary_type: str = DEFAULT_SUMMARY_TYPE,
    llm_client: Optional[BaseLLMClient] = None,
    max_sentences: Optional[int] = None,
    raise_errors: bool = True,
) -> SummaryResult:
    """Convenience function to summarize text concisely or in detail.

    Args:
        text: Text to summarize.
        summary_type: 'concise' or 'detailed'.
        llm_client: Optional BaseLLMClient for generative summarization.
        max_sentences: Optional sentence count override.
        raise_errors: If True, raises exceptions on invalid input; if False, returns error result.

    Returns:
        Structured SummaryResult object.
    """
    summarizer = TextSummarizer(llm_client=llm_client, default_summary_type=summary_type)
    return summarizer.summarize(
        text=text,
        summary_type=summary_type,
        max_sentences=max_sentences,
        raise_errors=raise_errors,
    )
