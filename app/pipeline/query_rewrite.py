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
_USER_TURN_RE = re.compile(r"User:\s*(.+?)(?:\s*Assistant:|$)", re.IGNORECASE)

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
    "too",
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

_PET_REFERENCE_TOKENS = {
    "dog",
    "dogs",
    "cat",
    "cats",
    "pet",
    "pets",
    "puppy",
    "puppies",
    "kitten",
    "kittens",
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
    r"\bme too\b",
    r"\btoo[?.!,\s]*$",
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
- Use conversation history when the current question depends on earlier turns to identify the pet, symptom, event, timeline, or referent.
- Reuse the exact animal, symptom, food, medication, and timeline terms from the question or history whenever possible.
- Prefer a concise standalone query that still reads like natural English.
- Remove filler phrases such as "yeah", "I think", and "it seems like" when they do not change the meaning.
- Do not output awkward keyword bags or inverted phrases such as "weak and sleepy dog"; prefer short natural rewrites such as "dog is weak and sleepy".
- If the current question is a follow-up about the same pet or ongoing issue, carry forward the relevant prior issue, event, or timeline needed to make the query self-contained.
- When the follow-up adds a new symptom, status update, severity change, or time update to an existing case, keep both the prior issue and the new update in the rewrite.
- When a pronoun like "it", "he", "she", or "them" can be resolved from history, replace it with the resolved pet, symptom, event, or item instead of leaving the reference vague.
- Do not drop important context from history if that context is necessary for retrieval, such as the animal type, an ongoing symptom, or an explicit timeline.
- Preserve the user's actual intent or question, not just the background facts. The rewrite should still sound like a question when the user is asking a question.
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
  Output: dog with diarrhea and vomiting
- History:
  1. My dog has not eaten since yesterday.
  Question: yeah it seems like he is quite weak and sleepy
  Output: dog has not eaten since yesterday and is weak and sleepy
- History:
  1. I listed the common symptoms of parasites in dogs.
  Question: Please list them as bullet points
  Output: common symptoms of parasites in dogs
- History:
  1. My dog ate a chocolate bar.
  2. I said to induce vomiting if it happened within the past six hours.
  Question: I think it's over six hours, so does that mean it's ok right now?
  Output: is my dog okay after eating a chocolate bar more than 6 hours ago

Conversation history:
{history}

Current question:
{question}

Rewritten query:
""".strip()


@dataclass(slots=True)
class QueryRewriteDecision:
    original_query: str
    rewrite_needed: bool
    rule_score: int
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryRewriteResult:
    original_query: str
    rewrite_query: str
    rewrite_needed: bool
    rule_score: int
    reasons: list[str] = field(default_factory=list)
    history_available: bool = False
    llm_used: bool = False
    rewrite_applied: bool = False


def _print_rewrite_result(result: QueryRewriteResult) -> None:
    print(
        "[QUERY REWRITE] "
        f"needed={result.rewrite_needed} "
        f"applied={result.rewrite_applied} "
        f"history={result.history_available} "
        f"llm={result.llm_used} "
        f"score={result.rule_score} "
        f"reasons={','.join(result.reasons) or '-'} "
        f"original={result.original_query!r} "
        f"rewrite={result.rewrite_query!r}",
        flush=True,
    )


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


def _latest_user_issue(conversation_context: Sequence[str] | None) -> str:
    for item in reversed(_clean_conversation_context(conversation_context)):
        match = _USER_TURN_RE.search(item)
        if match:
            return _normalize_query(match.group(1))
    return ""


def _candidate_uses_recent_issue(candidate_query: str, recent_issue: str) -> bool:
    recent_tokens = {
        token for token in _content_tokens(recent_issue) if token not in _PET_REFERENCE_TOKENS
    }
    if not recent_tokens:
        return True

    candidate_tokens = set(_content_tokens(candidate_query))
    return bool(candidate_tokens & recent_tokens)


def _merge_recent_issue_into_rewrite(recent_issue: str, candidate_query: str) -> str:
    recent_issue = re.sub(r"^(?:my|our)\s+", "", recent_issue.strip(), flags=re.IGNORECASE)
    recent_issue = recent_issue.rstrip(".?! ")
    candidate_query = candidate_query.strip().rstrip(".?! ")
    if not recent_issue or not candidate_query:
        return candidate_query or recent_issue

    if candidate_query.casefold().startswith(recent_issue.casefold()):
        return candidate_query

    recent_tokens = _latin_tokens(recent_issue)
    candidate_tokens = _latin_tokens(candidate_query)
    if recent_tokens and candidate_tokens and recent_tokens[0] == candidate_tokens[0]:
        candidate_tail = re.sub(
            r"^[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\s+",
            "",
            candidate_query,
            count=1,
        )
        if candidate_tail:
            return f"{recent_issue} and {candidate_tail}"

    return f"{recent_issue} and {candidate_query}"


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


def inspect_query_rewrite(query: str) -> QueryRewriteDecision:
    original_query = _normalize_query(query)
    if not original_query:
        return QueryRewriteDecision(
            original_query="",
            rewrite_needed=False,
            rule_score=0,
        )

    rule_score, reasons = score_query_rewrite_need(original_query)
    return QueryRewriteDecision(
        original_query=original_query,
        rewrite_needed=rule_score >= settings.QUERY_REWRITE_RULE_THRESHOLD,
        rule_score=rule_score,
        reasons=reasons,
    )


def _format_history(conversation_context: Sequence[str] | None) -> str:
    cleaned = _clean_conversation_context(conversation_context)
    if not cleaned:
        return "(none)"

    return "\n".join(
        f"{idx}. {turn}" for idx, turn in enumerate(cleaned, start=1)
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
    candidate_query = _normalize_query(candidate_query)
    if not candidate_query:
        return False

    if candidate_query.casefold() == original_query.casefold():
        return True

    if len(_compact_text(candidate_query)) <= 4:
        return False

    if not _content_tokens(candidate_query) and not _has_domain_hint(candidate_query):
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
    recent_issue = _latest_user_issue(cleaned_history)
    if (
        recent_issue
        and (_has_ambiguous_reference(query) or _has_follow_up_marker(query))
        and not _candidate_uses_recent_issue(candidate, recent_issue)
    ):
        candidate = _sanitize_rewrite(
            _merge_recent_issue_into_rewrite(recent_issue, candidate),
            query,
        )

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
    rewrite_decision: QueryRewriteDecision | None = None,
) -> QueryRewriteResult:
    decision = rewrite_decision or inspect_query_rewrite(query)
    original_query = decision.original_query
    cleaned_history = _clean_conversation_context(conversation_context)
    history_available = bool(cleaned_history)
    rule_score = decision.rule_score
    reasons = list(decision.reasons)
    rewrite_needed = decision.rewrite_needed

    if not original_query:
        result = QueryRewriteResult(
            original_query="",
            rewrite_query="",
            rewrite_needed=False,
            rule_score=0,
            history_available=history_available,
        )
        _print_rewrite_result(result)
        return result

    if not settings.QUERY_REWRITE_ENABLED or not rewrite_needed or not history_available:
        result = QueryRewriteResult(
            original_query=original_query,
            rewrite_query=original_query,
            rewrite_needed=rewrite_needed,
            rule_score=rule_score,
            reasons=reasons,
            history_available=history_available,
        )
        _print_rewrite_result(result)
        return result

    rewrite_query = _try_llm_rewrite(
        original_query,
        conversation_context=cleaned_history,
    )
    rewrite_applied = rewrite_query.casefold() != original_query.casefold()

    result = QueryRewriteResult(
        original_query=original_query,
        rewrite_query=rewrite_query,
        rewrite_needed=rewrite_needed,
        rule_score=rule_score,
        reasons=reasons,
        history_available=history_available,
        llm_used=True,
        rewrite_applied=rewrite_applied,
    )
    _print_rewrite_result(result)
    return result
