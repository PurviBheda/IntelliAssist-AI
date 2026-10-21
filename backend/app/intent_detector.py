"""IntelliAssist AI - Intent Detection Module.

This module provides production-grade conversational intent classification and parameter extraction:
1. Recognizes 5 canonical intent classes:
   - 'document_question': Document-related queries and policy investigations.
   - 'summarize': Text, document, or conversation summarization requests.
   - 'sentiment_analysis': Requests to evaluate tone, sentiment, or emotional valence.
   - 'greeting': Conversational greetings, farewells, gratitude, and bot persona questions.
   - 'out_of_scope': Inquiries unrelated to document intelligence (e.g. coding, math, general trivia, recipes).
2. Extracts domain parameters (e.g. summary style, targets, inline text, sentiment targets).
3. Robust to variations in syntax, idioms, and slang (e.g. 'TL;DR', 'recap', 'vibe check').
4. Employs hybrid pattern-matching and optional semantic embedding similarity with prototype anchors.
5. Employs safe fallback to 'document_question' on ambiguous or low-confidence queries,
   ensuring grounding requirements are never bypassed for factual document inquiries.
6. Robust input validation and graceful error handling for empty, whitespace-only, or invalid inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from app.embedding_service import EmbeddingService


# ============================================================================
# Enums & Constants
# ============================================================================

class IntentType(str, Enum):
    """Supported intent classification categories."""

    DOCUMENT_QUESTION = "document_question"
    SUMMARIZE = "summarize"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    GREETING = "greeting"
    OUT_OF_SCOPE = "out_of_scope"


# Ambiguity confidence threshold: if top confidence < threshold or margin between top two < margin,
# mark is_ambiguous = True and route to safe fallback
DEFAULT_AMBIGUITY_THRESHOLD = 0.55
DEFAULT_AMBIGUITY_MARGIN = 0.15
SAFE_FALLBACK_INTENT = IntentType.DOCUMENT_QUESTION.value


# ============================================================================
# Response Model
# ============================================================================

@dataclass
class IntentResult:
    """Structured result returned by IntentDetector.

    Attributes:
        intent: Detected intent category (one of IntentType values).
        confidence: Calibrated certainty score between 0.0 and 1.0.
        parameters: Dictionary of extracted parameters (e.g. summary_type, target, text).
        is_ambiguous: True if intent resolution was ambiguous or close-call.
        fallback_used: True if the safe fallback intent was assigned due to uncertainty.
        raw_input: Cleaned original input text.
        alternative_intents: Ranked list of alternative intents with confidence scores.
        status: Operation status ('success' or 'error').
        error: Error message string if an error occurred, else None.
    """

    intent: str
    confidence: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False
    fallback_used: bool = False
    raw_input: str = ""
    alternative_intents: List[Dict[str, Any]] = field(default_factory=list)
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
# Input Validation Helper
# ============================================================================

def validate_intent_input(text: Any) -> str:
    """Validate user message input for intent detection.

    Args:
        text: Input string.

    Returns:
        Stripped string.

    Raises:
        ValueError: If text is None, empty, or whitespace-only.
        TypeError: If text is not a string.
    """
    if text is None:
        raise ValueError("Text for intent detection cannot be None.")
    if not isinstance(text, str):
        raise TypeError(f"Text for intent detection must be a string, got: {type(text).__name__}")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Text for intent detection cannot be empty or whitespace-only.")
    return cleaned


# ============================================================================
# Parameter Extraction Helpers
# ============================================================================

def extract_quoted_text(text: str) -> Optional[str]:
    """Extract text enclosed in single or double quotes, or backticks."""
    patterns = [
        r'"([^"\\]*(?:\\.[^"\\]*)*)"',
        r"'([^'\\]*(?:\\.[^'\\]*)*)'",
        r"`([^`]+)`",
        r"“([^”]+)”",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def extract_text_after_delimiter(text: str, delimiters: List[str]) -> Optional[str]:
    """Extract content trailing specific delimiters (e.g. ':', 'of', 'for')."""
    lower = text.lower()
    for delim in delimiters:
        if delim in lower:
            idx = lower.find(delim)
            trailing = text[idx + len(delim):].strip()
            # Strip trailing quotes if present
            trailing = trailing.strip("\"'`“”")
            if len(trailing) >= 3:
                return trailing
    return None


# ============================================================================
# Intent Archetype Exemplars (for Semantic Matching)
# ============================================================================

INTENT_EXEMPLARS: Dict[str, List[str]] = {
    IntentType.GREETING.value: [
        "hello", "hi there", "good morning", "good afternoon", "hey",
        "how are you doing", "who are you", "what can you do", "nice to meet you",
        "thank you very much", "thanks for your help", "goodbye", "bye", "see you later",
    ],
    IntentType.SUMMARIZE.value: [
        "summarize this document", "give me a concise summary of the policy",
        "can you summarize our conversation", "provide a detailed summary of this text",
        "tl;dr of this section", "recap what we discussed", "brief overview of the handbook",
        "summarize the main points", "short summary please", "summarize:",
    ],
    IntentType.SENTIMENT_ANALYSIS.value: [
        "analyze the sentiment of this text", "is this feedback positive or negative",
        "what is the sentiment score of this review", "detect the emotional tone of this message",
        "vibe check on this feedback", "classify the sentiment as positive negative or neutral",
        "sentiment analysis of:", "what's the sentiment of this statement",
    ],
    IntentType.OUT_OF_SCOPE.value: [
        "write a python script to scrape data", "how do I cook a chocolate cake recipe",
        "what is the weather in Seattle tomorrow", "write a C++ function to sort numbers",
        "who won the 1994 world cup", "debug this javascript code", "tell me a funny joke",
        "how do I hack a wifi password", "solve this calculus derivative equation",
    ],
    IntentType.DOCUMENT_QUESTION.value: [
        "what is the password expiration policy", "how many days of pto do employees get",
        "what are the requirements for multi factor authentication", "who is eligible for health insurance",
        "what does the expense policy state about travel meals", "where is the office located according to the handbook",
        "can employees carry over unused vacation days", "what is the remote work policy",
    ],
}


# ============================================================================
# Intent Detector Implementation
# ============================================================================

class IntentDetector:
    """Production-grade intent detector supporting hybrid rule-based and semantic classification."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        ambiguity_threshold: float = DEFAULT_AMBIGUITY_THRESHOLD,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        fallback_intent: str = SAFE_FALLBACK_INTENT,
    ) -> None:
        """Initialize the IntentDetector.

        Args:
            embedding_service: Optional EmbeddingService instance for semantic similarity scoring.
            ambiguity_threshold: Minimum confidence required before flagging ambiguity.
            ambiguity_margin: Minimum margin between top-1 and top-2 intents.
            fallback_intent: Safe fallback intent assigned on ambiguity (default: 'document_question').
        """
        self.embedding_service = embedding_service
        self.ambiguity_threshold = ambiguity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.fallback_intent = fallback_intent

        # Precompute exemplar embeddings if embedding_service is provided
        self._exemplar_embeddings: Dict[str, np.ndarray] = {}
        if self.embedding_service is not None:
            self._precompute_exemplar_embeddings()

    def _precompute_exemplar_embeddings(self) -> None:
        """Precompute normalized centroid embeddings for intent exemplar sets."""
        if self.embedding_service is None:
            return

        for intent_name, examples in INTENT_EXEMPLARS.items():
            try:
                vectors = [self.embedding_service.embed_text(ex) for ex in examples]
                stacked = np.vstack(vectors)
                centroid = np.mean(stacked, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid = centroid / norm
                self._exemplar_embeddings[intent_name] = centroid
            except Exception:
                # If embedding fails during initialization, proceed without embeddings
                self._exemplar_embeddings.clear()
                break

    # ========================================================================
    # Pattern Matchers & Feature Extractors
    # ========================================================================

    def _match_greeting_patterns(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for greeting, farewell, gratitude, and bot identity patterns."""
        t_clean = text.lower().strip()
        params: Dict[str, Any] = {}

        # If it starts with greeting but contains a document question:
        # e.g., "Hello, what is the password policy?" -> This is a DOCUMENT_QUESTION with a greeting prefix
        if re.search(r"^(hi|hello|hey|good morning|good evening)[,\s]+(what|where|how|who|can|is|are|tell|explain|does)\b", t_clean):
            # Do NOT treat as greeting; let it fall through to document_question
            return 0.10, {}

        # 1. Greetings: "hello", "hi there", "hey there!", "good morning", "hello! good morning"
        greeting_regex = re.compile(
            r"^(?:hi|hello|hey|howdy|greetings|good\s+morning|good\s+afternoon|good\s+evening|yo)"
            r"(?:[\s,!.]+(?:there|all|everyone|bot|assistant|good\s+morning|good\s+afternoon|good\s+evening))*[!.,\s]*$",
            re.IGNORECASE,
        )
        if greeting_regex.match(t_clean):
            params["greeting_type"] = "hello"
            return 0.98, params

        # 2. Gratitude: "thanks", "thank you", "thank you very much for your help!", "thanks a lot!"
        thanks_regex = re.compile(
            r"^(?:thanks|thank\s+you|thank\s+you\s+very\s+much|many\s+thanks|thx|appreciate\s+it|much\s+appreciated)"
            r"(?:[\s,!.]+(?:for\s+.*|so\s+much|a\s+lot|again))?[!.,\s]*$",
            re.IGNORECASE,
        )
        if thanks_regex.match(t_clean):
            params["greeting_type"] = "gratitude"
            return 0.98, params

        # 3. Goodbyes: "bye", "goodbye", "goodbye, have a great day!", "see you later, bye!"
        goodbye_regex = re.compile(
            r"^(?:bye|goodbye|see\s+you|see\s+ya|cya|farewell|have\s+a\s+good\s+(?:one|day)|take\s+care)"
            r"(?:[\s,!.]+(?:later|soon|bye|have\s+a\s+(?:great|good)\s+day|tomorrow))*[!.,\s]*$",
            re.IGNORECASE,
        )
        if goodbye_regex.match(t_clean):
            params["greeting_type"] = "goodbye"
            return 0.98, params

        # Also handle combined gratitude + goodbye: "Thank you, goodbye!"
        if re.search(r"\b(thank\s+you|thanks)\b.*?\b(bye|goodbye|see\s+you)\b", t_clean) or \
           re.search(r"\b(bye|goodbye)\b.*?\b(thank\s+you|thanks)\b", t_clean):
            params["greeting_type"] = "goodbye"
            return 0.98, params

        # 4. Bot identity / Chitchat
        chitchat_patterns = [
            r"^(who|what) are you\??$",
            r"^what is your name\??$",
            r"^how are you( doing)?\??$",
            r"^what can you do\??$",
            r"^help me\??$",
        ]
        t_punct_stripped = t_clean.strip("!?,. ")
        for pat in chitchat_patterns:
            if re.match(pat, t_punct_stripped):
                params["greeting_type"] = "chitchat"
                return 0.95, params

        return 0.0, params

    def _match_summarize_patterns(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for summarization requests and extract target, style, and text."""
        t_lower = text.lower().strip()
        params: Dict[str, Any] = {
            "summary_type": "concise",
            "target": "document",
            "text_to_summarize": None,
        }

        # Check requested style
        detailed_keywords = {"detailed", "in-depth", "comprehensive", "full", "elaborate", "bullet-point", "bullet points"}
        if any(w in t_lower for w in detailed_keywords):
            params["summary_type"] = "detailed"

        concise_keywords = {"concise", "brief", "short", "quick", "one-line", "high-level", "tldr", "tl;dr"}
        if any(w in t_lower for w in concise_keywords):
            params["summary_type"] = "concise"

        # Check target
        if any(w in t_lower for w in ["our conversation", "this conversation", "our chat", "the chat", "what we discussed", "recap our"]):
            params["target"] = "conversation"
        elif ":" in text or extract_quoted_text(text):
            quoted = extract_quoted_text(text)
            delimited = extract_text_after_delimiter(text, [":", "summarize:", "summarise:"])
            inline_text = quoted or delimited
            if inline_text and len(inline_text) > 10:
                params["target"] = "inline_text"
                params["text_to_summarize"] = inline_text

        # Strong summarization triggers
        strong_triggers = [
            r"\b(summarize|summarise|summary|tl;dr|tldr|recap)\b",
            r"\bgive me a (brief|quick|concise|detailed|short)?\s*summary\b",
            r"\bgive me an? (overview|recap)\b",
            r"\bprovide a (concise|detailed)?\s*summary\b",
            r"\bcan you summarize\b",
            r"\bcould you summarize\b",
            r"\bwhat is the summary of\b",
            r"\bbrief overview\b",
        ]

        for trig in strong_triggers:
            if re.search(trig, t_lower):
                return 0.95, params

        return 0.0, params

    def _match_sentiment_patterns(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for sentiment analysis requests and extract target and inline text."""
        t_lower = text.lower().strip()
        params: Dict[str, Any] = {
            "target": "inline_text",
            "text_to_analyze": None,
        }

        quoted = extract_quoted_text(text)
        delimited = extract_text_after_delimiter(
            text, [":", "sentiment of", "sentiment for", "tone of", "vibe of", "sentiment in", "mood of"]
        )
        inline_candidate = quoted or delimited
        if inline_candidate:
            params["text_to_analyze"] = inline_candidate

        # Check if targeting prior conversation
        if any(w in t_lower for w in ["our conversation", "last message", "previous message", "this chat", "what i said", "last response"]):
            params["target"] = "conversation"

        # Strong sentiment patterns
        sentiment_triggers = [
            r"\b(sentiment analysis|sentiment score|analyze sentiment|analyse sentiment|detect sentiment)\b",
            r"\bpositive\s+(?:or|and|vs)\s+negative\b",
            r"\b(is\s+this\s+.*?\b(positive|negative)\b)",
            r"\b(what is the (sentiment|tone|vibe|mood) of)\b",
            r"\b(classify the sentiment)\b",
            r"\b(rate the sentiment)\b",
            r"\b(sentiment of:?)\b",
            r"\b(vibe check)\b",
            r"\b(check the tone of)\b",
        ]

        for trig in sentiment_triggers:
            if re.search(trig, t_lower):
                return 0.95, params

        # General single-word keyword 'sentiment'
        if "sentiment" in t_lower:
            return 0.85, params

        return 0.0, params

    def _match_out_of_scope_patterns(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for explicitly out-of-scope tasks (coding, hacking, recipes, general trivia)."""
        t_lower = text.lower().strip()
        params: Dict[str, Any] = {}

        # 1. Code generation or programming assistance
        code_patterns = [
            r"\b(write|create|generate|debug|fix)\s+(a\s+)?(python|javascript|typescript|java|c\+\+|rust|go|sql|bash|html|css)\s+(script|code|function|program|app)\b",
            r"\b(write a program to|write code to|code for a|script to)\b",
        ]
        for pat in code_patterns:
            if re.search(pat, t_lower):
                params["category"] = "coding"
                return 0.92, params

        # 2. Hacking, exploits, or malicious activities
        hacking_patterns = [
            r"\b(how\s+(?:do\s+i\s+|can\s+i\s+|to\s+)?hack\b|crack\s+(?:a\s+)?password|exploit\s+a\s+vulnerability|bypass\s+security\s+controls|ddos)\b",
            r"\bhack\s+into\b",
        ]
        for pat in hacking_patterns:
            if re.search(pat, t_lower):
                params["category"] = "security_exploit"
                return 0.95, params

        # 3. Weather queries
        if re.search(r"\b(weather in|weather forecast|will it rain in|temperature in)\b", t_lower):
            params["category"] = "weather"
            return 0.95, params

        # 4. Recipes or cooking instructions
        if not re.search(r"^what\s+(is|are)\b", t_lower) and re.search(r"\b(recipe for|how to bake|how to cook|ingredients for)\b", t_lower) and not any(w in t_lower for w in ["policy", "expense", "allowance", "corporate"]):
            params["category"] = "recipe"
            return 0.90, params

        # 5. General sports/entertainment trivia
        trivia_patterns = [
            r"\b(who won the \d{4} (world cup|super bowl|nba|olympics))\b",
            r"\b(how tall is mount everest|capital of france|distance to the moon)\b",
        ]
        for pat in trivia_patterns:
            if re.search(pat, t_lower):
                params["category"] = "general_trivia"
                return 0.92, params

        return 0.0, params

    def _match_document_question_patterns(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """Check for document queries, policy checks, and standard informational questions."""
        t_lower = text.lower().strip()
        params: Dict[str, Any] = {"query": text}

        # Policy & enterprise keywords strongly indicate document questions
        doc_keywords = [
            "policy", "policies", "handbook", "guidelines", "procedure", "procedures",
            "password", "passwords", "pto", "vacation", "leave", "sick days", "holiday",
            "benefits", "mfa", "authentication", "security", "travel", "expense", "reimbursement",
            "employee", "employees", "hr", "manager", "allowance", "per diem", "confidential",
            "document", "documents", "section", "according to", "stated in", "file",
        ]
        has_doc_kw = any(w in t_lower for w in doc_keywords)

        # Interrogative starters
        question_starters = [
            r"^(what|where|when|why|how|who|which|whose)\b",
            r"^(is there|are there|can employees|can a user|do employees|does the policy)\b",
            r"^(tell me about|explain|describe)\b",
        ]
        has_question_starter = any(re.search(pat, t_lower) for pat in question_starters)
        ends_with_qmark = text.strip().endswith("?")

        # Check for vague demonstrative queries like "tell me about this", "tell me more"
        # without any substantive noun
        vague_demonstrative = bool(re.match(r"^(tell me about|explain|what about|describe)\s+(this|that|it|them|things|stuff)[.?!]*$", t_lower))
        if vague_demonstrative:
            # Low confidence to trigger ambiguity / fallback
            return 0.40, params

        if has_doc_kw and (has_question_starter or ends_with_qmark):
            return 0.95, params

        if has_doc_kw:
            return 0.85, params

        if has_question_starter or ends_with_qmark:
            return 0.70, params

        return 0.30, params

    # ========================================================================
    # Semantic Scoring via Exemplars
    # ========================================================================

    def _compute_semantic_scores(self, text: str) -> Dict[str, float]:
        """Compute cosine similarity scores against intent exemplar centroids."""
        if self.embedding_service is None or not self._exemplar_embeddings:
            return {}

        try:
            vec = self.embedding_service.embed_text(text)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm

            scores: Dict[str, float] = {}
            for intent_name, centroid in self._exemplar_embeddings.items():
                sim = float(np.dot(vec, centroid))
                # Map cosine similarity [-1.0, 1.0] to [0.0, 1.0]
                scores[intent_name] = max(0.0, min(1.0, (sim + 1.0) / 2.0))
            return scores
        except Exception:
            return {}

    # ========================================================================
    # Main Detection Method
    # ========================================================================

    def detect(self, text: Any, raise_errors: bool = True) -> IntentResult:
        """Detect the intent of the input text and extract associated parameters.

        Args:
            text: Input string to classify.
            raise_errors: If True, raises exceptions on empty/invalid inputs; if False,
                          returns a safe fallback IntentResult.

        Returns:
            Structured IntentResult object.

        Raises:
            ValueError: If text is empty or whitespace-only and raise_errors is True.
            TypeError: If text is not a string and raise_errors is True.
        """
        # 1. Validation
        try:
            cleaned = validate_intent_input(text)
        except Exception as e:
            if raise_errors:
                raise
            return IntentResult(
                intent=self.fallback_intent,
                confidence=0.0,
                parameters={},
                is_ambiguous=True,
                fallback_used=True,
                raw_input=str(text) if text is not None else "",
                status="error",
                error=str(e),
            )

        # 2. Rule-based feature evaluation
        rule_scores: Dict[str, float] = {}
        all_params: Dict[str, Dict[str, Any]] = {}

        # Greeting check
        g_score, g_params = self._match_greeting_patterns(cleaned)
        rule_scores[IntentType.GREETING.value] = g_score
        all_params[IntentType.GREETING.value] = g_params

        # Sentiment check
        s_score, s_params = self._match_sentiment_patterns(cleaned)
        rule_scores[IntentType.SENTIMENT_ANALYSIS.value] = s_score
        all_params[IntentType.SENTIMENT_ANALYSIS.value] = s_params

        # Summarize check
        sum_score, sum_params = self._match_summarize_patterns(cleaned)
        rule_scores[IntentType.SUMMARIZE.value] = sum_score
        all_params[IntentType.SUMMARIZE.value] = sum_params

        # Out-of-scope check
        oos_score, oos_params = self._match_out_of_scope_patterns(cleaned)
        rule_scores[IntentType.OUT_OF_SCOPE.value] = oos_score
        all_params[IntentType.OUT_OF_SCOPE.value] = oos_params

        # Document question check
        doc_score, doc_params = self._match_document_question_patterns(cleaned)
        rule_scores[IntentType.DOCUMENT_QUESTION.value] = doc_score
        all_params[IntentType.DOCUMENT_QUESTION.value] = doc_params

        # High-confidence pattern priority: If strong pattern triggers, prioritize it immediately
        # E.g. greetings, explicit sentiment commands, explicit summarization directives
        for priority_intent in [IntentType.GREETING.value, IntentType.SUMMARIZE.value, IntentType.SENTIMENT_ANALYSIS.value, IntentType.OUT_OF_SCOPE.value]:
            if rule_scores[priority_intent] >= 0.90:
                alt_intents = [
                    {"intent": k, "score": round(v, 4)}
                    for k, v in sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
                    if k != priority_intent
                ]
                return IntentResult(
                    intent=priority_intent,
                    confidence=round(rule_scores[priority_intent], 4),
                    parameters=all_params[priority_intent],
                    is_ambiguous=False,
                    fallback_used=False,
                    raw_input=cleaned,
                    alternative_intents=alt_intents,
                    status="success",
                )

        # 3. Hybrid scoring: combine rule-based with semantic similarity (if available)
        sem_scores = self._compute_semantic_scores(cleaned)
        combined_scores: Dict[str, float] = {}

        all_intents = [
            IntentType.DOCUMENT_QUESTION.value,
            IntentType.SUMMARIZE.value,
            IntentType.SENTIMENT_ANALYSIS.value,
            IntentType.GREETING.value,
            IntentType.OUT_OF_SCOPE.value,
        ]

        for intent in all_intents:
            r_score = rule_scores.get(intent, 0.0)
            if sem_scores:
                s_score_val = sem_scores.get(intent, 0.0)
                # Weighted fusion: 0.65 rules + 0.35 semantic
                combined_scores[intent] = (0.65 * r_score) + (0.35 * s_score_val)
            else:
                combined_scores[intent] = r_score

        # 4. Rank intents
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        top_intent, top_score = ranked[0]
        second_intent, second_score = ranked[1] if len(ranked) > 1 else (None, 0.0)

        # 5. Check ambiguity and safe fallback
        # If top score is low (< ambiguity_threshold) or margin is tiny (< ambiguity_margin):
        is_ambiguous = (top_score < self.ambiguity_threshold) or (
            second_intent is not None and (top_score - second_score < self.ambiguity_margin) and top_score < 0.75
        )

        final_intent = top_intent
        fallback_used = False

        if is_ambiguous:
            # Safe Fallback: In an enterprise document assistant, ambiguous queries
            # MUST be safely routed to document_question so that RAG evaluates vector context
            # and refuses safely if not found, rather than incorrectly executing a non-grounded path.
            final_intent = self.fallback_intent
            fallback_used = (top_intent != self.fallback_intent)

        alt_intents = [
            {"intent": k, "score": round(v, 4)}
            for k, v in ranked
            if k != final_intent
        ]

        return IntentResult(
            intent=final_intent,
            confidence=round(top_score, 4),
            parameters=all_params.get(final_intent, {}),
            is_ambiguous=is_ambiguous,
            fallback_used=fallback_used,
            raw_input=cleaned,
            alternative_intents=alt_intents,
            status="success",
        )


# ============================================================================
# Functional Public Interface
# ============================================================================

def detect_intent(
    text: Any,
    embedding_service: Optional[EmbeddingService] = None,
    raise_errors: bool = True,
) -> IntentResult:
    """Detect intent of input text.

    Args:
        text: Input string.
        embedding_service: Optional EmbeddingService instance.
        raise_errors: If True, raises exceptions on empty/invalid inputs; if False, returns safe fallback.

    Returns:
        Structured IntentResult object.
    """
    detector = IntentDetector(embedding_service=embedding_service)
    return detector.detect(text, raise_errors=raise_errors)
