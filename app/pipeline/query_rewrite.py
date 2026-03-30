from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.settings import settings
from app.deps.container import get_llm

logger = logging.getLogger(__name__)

_LATIN_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "than",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}

_DOMAIN_HINTS = {
    "blood",
    "breathing",
    "cough",
    "coughing",
    "dehydration",
    "diarrhea",
    "diarrhoea",
    "dog",
    "dogs",
    "ear",
    "ears",
    "eat",
    "eating",
    "fever",
    "flea",
    "fleas",
    "itch",
    "itching",
    "kitten",
    "lethargy",
    "parasite",
    "parasites",
    "pet",
    "pets",
    "poop",
    "puppy",
    "rash",
    "seizure",
    "stool",
    "tick",
    "ticks",
    "urine",
    "urinating",
    "vet",
    "vomit",
    "vomiting",
    "vomits",
    "worm",
    "worms",
    "wound",
}

_IRREGULAR_TOKEN_MAP = {
    "ate": "eat",
    "eaten": "eat",
    "drank": "drink",
    "drunk": "drink",
    "worse": "bad",
    "worsened": "worsen",
}

_STRONG_PRONOUN_PATTERNS = (
    r"\bit\b",
    r"\bits\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\btheir\b",
    r"\bhe\b",
    r"\bhim\b",
    r"\bhis\b",
    r"\bshe\b",
    r"\bher\b",
)

_FOLLOW_UP_PATTERNS = (
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\bwhat if\b",
    r"\bhow come\b",
    r"\bas well\b",
    r"\btoo\b",
    r"\binstead\b",
    r"\banother one\b",
    r"\bthe other one\b",
    r"\bthat one\b",
)

_GENERIC_REFERENCE_PATTERNS = (
    r"\b(this|that|these|those)\s+(one|ones|thing|issue|problem|case|situation)\b",
    r"\b(is|was)\s+(this|that)\s+(ok|okay|normal|safe|serious|dangerous)\b",
    r"\b(can|could|should)\s+(it|this|that)\b",
)

_REWRITE_PROMPT_TEMPLATE = """
You rewrite a user question into a retrieval-friendly search query for a pet-health RAG system.

Rules:
- Preserve the user's original meaning.
- Use conversation history only when it explicitly identifies the referent in the current question.
- Do not guess missing entities, symptoms, timelines, diagnoses, or treatments.
- Do not add facts that are not explicitly present in the current question or the conversation history.
- If the history is insufficient to resolve the reference, return the current question unchanged.
- Output only the rewritten query, with no explanation or quotation marks.

Examples:
- History:
  1. My dog ate chocolate this morning.
  Question: Is it dangerous?
  Output: Is eating chocolate dangerous for a dog?
- History:
  1. My dog has diarrhea.
  Question: What about vomiting?
  Output: dog diarrhea and vomiting

Conversation history:
{history}

Current question:
{question}

Rewritten query:
""".strip()


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    retrieval_query: str
    rewrite_needed: bool
    rule_score: int
    reasons: list[str] = field(default_factory=list)
    history_available: bool = False
    llm_used: bool = False
    rewrite_applied: bool = False


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _clean_conversation_context(
    conversation_context: Sequence[str] | None,
) -> list[str]:
    if not conversation_context:
        return []

    if isinstance(conversation_context, str):
        conversation_context = [conversation_context]

    cleaned: list[str] = []
    for item in conversation_context:
        normalized = _normalize_query(item)
        if normalized:
            cleaned.append(normalized)
    return cleaned


def _latin_tokens(query: str) -> list[str]:
    return [token.lower() for token in _LATIN_TOKEN_RE.findall(query)]


def _content_tokens(query: str) -> list[str]:
    return [
        token
        for token in _latin_tokens(query)
        if len(token) >= 3 and token not in _EN_STOPWORDS
    ]


def _normalize_token(token: str) -> str:
    token = token.lower()
    token = _IRREGULAR_TOKEN_MAP.get(token, token)

    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"

    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]

    if token.endswith("ous") and len(token) > 5:
        return token[:-3]

    return token


def _normalized_content_token_set(texts: Sequence[str]) -> set[str]:
    tokens: set[str] = set()
    for text in texts:
        for token in _content_tokens(text):
            tokens.add(_normalize_token(token))
    return tokens


def _compact_text(query: str) -> str:
    return _PUNCT_RE.sub("", query).strip().lower()


def _has_domain_hint(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in _DOMAIN_HINTS)


def _has_ambiguous_reference(query: str) -> bool:
    lowered = query.lower()
    if any(re.search(pattern, lowered) for pattern in _STRONG_PRONOUN_PATTERNS):
        return True
    return any(re.search(pattern, lowered) for pattern in _GENERIC_REFERENCE_PATTERNS)


def _has_follow_up_marker(query: str) -> bool:
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in _FOLLOW_UP_PATTERNS)


def _looks_short_or_fragmentary(query: str) -> bool:
    compact_len = len(_compact_text(query))
    content_count = len(_content_tokens(query))
    latin_count = len(_latin_tokens(query))

    if compact_len <= 8:
        return True

    return latin_count <= 3 and content_count <= 2


def _lacks_keyword_like_terms(query: str) -> bool:
    if _has_domain_hint(query):
        return False

    return len(_content_tokens(query)) < 2


def _looks_overlong_or_multi_intent(query: str) -> bool:
    lowered = query.lower()
    separators = sum(
        lowered.count(token)
        for token in (",", ";", " and ", " but ", " because ", " then ", " also ")
    )
    return len(query) >= 120 or separators >= 2 or lowered.count("?") >= 2


def score_query_rewrite_need(query: str) -> tuple[int, list[str]]:
    normalized = _normalize_query(query)
    if not normalized:
        return 0, []

    score = 0
    reasons: list[str] = []

    if _has_ambiguous_reference(normalized):
        score += 2
        reasons.append("ambiguous_reference")

    if _has_follow_up_marker(normalized):
        score += 2
        reasons.append("follow_up_marker")

    if _looks_short_or_fragmentary(normalized):
        score += 1
        reasons.append("short_or_fragmentary")

    if _lacks_keyword_like_terms(normalized):
        score += 1
        reasons.append("keyword_sparse")

    if _looks_overlong_or_multi_intent(normalized):
        score += 1
        reasons.append("overlong_or_multi_intent")

    return score, reasons


def _format_history(conversation_context: Sequence[str] | None) -> str:
    cleaned = _clean_conversation_context(conversation_context)
    if not cleaned:
        return "(none)"

    recent_turns = cleaned[-4:]
    return "\n".join(
        f"{idx}. {turn}" for idx, turn in enumerate(recent_turns, start=1)
    )


@lru_cache
def _get_rewrite_chain():
    prompt = PromptTemplate(
        template=_REWRITE_PROMPT_TEMPLATE,
        input_variables=["history", "question"],
    )
    return prompt | get_llm() | StrOutputParser()


def _sanitize_rewrite(candidate: str, original_query: str) -> str:
    cleaned = _normalize_query(candidate)
    if not cleaned:
        return original_query

    cleaned = cleaned.strip("`\"' ")
    lowered = cleaned.lower()
    prefixes = (
        "rewritten query:",
        "rewrite:",
        "query:",
        "search query:",
        "output:",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    if not cleaned:
        return original_query

    if "\n" in cleaned:
        cleaned = _normalize_query(cleaned.splitlines()[0])

    max_len = max(160, len(original_query) * 3)
    if len(cleaned) > max_len:
        return original_query

    return cleaned


def _is_safe_rewrite(
    original_query: str,
    candidate_query: str,
    conversation_context: Sequence[str] | None,
) -> bool:
    if candidate_query.casefold() == original_query.casefold():
        return True

    allowed_tokens = _normalized_content_token_set(
        [original_query, *_clean_conversation_context(conversation_context)]
    )
    candidate_tokens = _normalized_content_token_set([candidate_query])
    if candidate_tokens - allowed_tokens:
        return False

    if _has_domain_hint(candidate_query) and not any(
        _has_domain_hint(text)
        for text in [original_query, *_clean_conversation_context(conversation_context)]
    ):
        return False

    return True


def _try_llm_rewrite(
    query: str,
    conversation_context: Sequence[str] | None = None,
) -> str:
    cleaned_history = _clean_conversation_context(conversation_context)
    if not cleaned_history:
        return query

    try:
        raw = _get_rewrite_chain().invoke(
            {
                "history": _format_history(cleaned_history),
                "question": query,
            }
        )
    except Exception:
        logger.exception("Query rewrite LLM call failed")
        return query

    candidate = _sanitize_rewrite(str(raw), query)
    if not _is_safe_rewrite(query, candidate, cleaned_history):
        logger.info(
            "Reject unsafe query rewrite original=%r candidate=%r",
            query,
            candidate,
        )
        return query

    return candidate


def rewrite_query_for_retrieval(
    query: str,
    conversation_context: Sequence[str] | None = None,
) -> QueryRewriteResult:
    original_query = _normalize_query(query)
    cleaned_history = _clean_conversation_context(conversation_context)
    history_available = bool(cleaned_history)
    rule_score, reasons = score_query_rewrite_need(original_query)
    rewrite_needed = rule_score >= settings.QUERY_REWRITE_RULE_THRESHOLD

    if not original_query:
        return QueryRewriteResult(
            original_query="",
            retrieval_query="",
            rewrite_needed=False,
            rule_score=0,
            history_available=history_available,
        )

    if not settings.QUERY_REWRITE_ENABLED or not rewrite_needed or not history_available:
        return QueryRewriteResult(
            original_query=original_query,
            retrieval_query=original_query,
            rewrite_needed=rewrite_needed,
            rule_score=rule_score,
            reasons=reasons,
            history_available=history_available,
        )

    retrieval_query = _try_llm_rewrite(
        original_query,
        conversation_context=cleaned_history,
    )
    rewrite_applied = retrieval_query.casefold() != original_query.casefold()

    return QueryRewriteResult(
        original_query=original_query,
        retrieval_query=retrieval_query,
        rewrite_needed=rewrite_needed,
        rule_score=rule_score,
        reasons=reasons,
        history_available=history_available,
        llm_used=True,
        rewrite_applied=rewrite_applied,
    )
