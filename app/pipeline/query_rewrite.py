from __future__ import annotations

import json
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

_LLM_REWRITE_PROMPT_TEMPLATE = """
You decide whether a user question for a pet-health retrieval system should be rewritten, and you provide the rewritten query when needed.

Output exactly one valid JSON object with these keys:
- rewrite_needed: true or false
- reason: a short snake_case label
- rewritten_query: string

Decision rules:
- Set rewrite_needed to true when the question is not retrieval-ready on its own.
- Questions usually need rewrite when they contain ambiguous references, missing subject/topic, missing concrete noun details, shorthand follow-up wording, or status updates that must be merged with prior context.
- Examples that usually need rewrite: "Is it dangerous?", "What about vomiting?", "And now he seems weak", "Is that okay right now?", "What dosage for this?".
- Set rewrite_needed to false when the question is already standalone, explicit, and specific enough for retrieval.
- If history is insufficient to resolve an ambiguous reference, set rewrite_needed to false and keep the original question unchanged.
- Prefer tracked topic-state lines first, then structured memory lines, then recent turns.
- Preserve the user's original meaning.
- Reuse the exact animal, symptom, event, food, medication, and timeline terms from the question or history whenever possible.
- Prefer a concise standalone query that still reads like natural English.
- Remove filler phrases such as "yeah", "I think", and "it seems like" when they do not change the meaning.
- If rewrite_needed is false, rewritten_query must equal the original question.
- Do not invent facts or missing entities.
- Output JSON only.

Examples:
- History:
  1. pet_type=dog
  2. event=ate chocolate
  3. User: My dog ate chocolate this morning.
  Question: Is it dangerous?
  Output: {{"rewrite_needed": true, "reason": "ambiguous_reference", "rewritten_query": "Is eating chocolate dangerous for a dog?"}}
- History:
  1. pet_type=dog
  2. symptom=diarrhea
  Question: What about vomiting?
  Output: {{"rewrite_needed": true, "reason": "follow_up_missing_topic", "rewritten_query": "dog with diarrhea and vomiting"}}
- History:
  1. (none)
  Question: What are signs of dehydration in dogs?
  Output: {{"rewrite_needed": false, "reason": "already_specific", "rewritten_query": "What are signs of dehydration in dogs?"}}

Conversation history:
{history}

Current question:
{question}

JSON:
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
        template=_LLM_REWRITE_PROMPT_TEMPLATE,
        input_variables=["history", "question"],
    )
    return prompt | get_llm() | StrOutputParser()


def _extract_json_object(raw: str) -> str:
    cleaned = _normalize_query(raw)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("query rewrite model did not return a JSON object")
    return cleaned[start : end + 1]


def _sanitize_reason(reason: object) -> str:
    cleaned = _normalize_query(str(reason or "")).lower().replace(" ", "_")
    cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
    return cleaned or "unspecified"


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


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
    del conversation_context

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


def _should_backfill_recent_issue(reason: str) -> bool:
    return reason in {
        "ambiguous_reference",
        "follow_up_missing_topic",
        "missing_topic",
        "missing_specific_detail",
        "missing_subject_detail",
    }


def _llm_decide_and_rewrite(
    query: str,
    conversation_context: Sequence[str] | None = None,
) -> tuple[QueryRewriteDecision, str, bool]:
    original_query = _normalize_query(query)
    cleaned_history = _clean_conversation_context(conversation_context)

    if not original_query:
        return (
            QueryRewriteDecision(
                original_query="",
                rewrite_needed=False,
                rule_score=0,
            ),
            "",
            False,
        )

    try:
        raw = _get_rewrite_chain().invoke(
            {
                "history": _format_history(cleaned_history),
                "question": original_query,
            }
        )
        payload = json.loads(_extract_json_object(str(raw)))
        rewrite_needed = _parse_bool(payload.get("rewrite_needed"))
        reason = _sanitize_reason(payload.get("reason"))
        rewritten_query = _sanitize_rewrite(
            str(payload.get("rewritten_query", original_query)),
            original_query,
        )
    except Exception:
        logger.exception("Query rewrite LLM call failed")
        return (
            QueryRewriteDecision(
                original_query=original_query,
                rewrite_needed=False,
                rule_score=0,
                reasons=["llm_failure"],
            ),
            original_query,
            False,
        )

    if not rewrite_needed:
        rewritten_query = original_query

    recent_issue = _latest_user_issue(cleaned_history)
    if (
        rewrite_needed
        and recent_issue
        and _should_backfill_recent_issue(reason)
        and not _candidate_uses_recent_issue(rewritten_query, recent_issue)
    ):
        rewritten_query = _sanitize_rewrite(
            _merge_recent_issue_into_rewrite(recent_issue, rewritten_query),
            original_query,
        )

    decision = QueryRewriteDecision(
        original_query=original_query,
        rewrite_needed=rewrite_needed,
        rule_score=1 if rewrite_needed else 0,
        reasons=[reason],
    )
    return decision, rewritten_query, True


def inspect_query_rewrite(
    query: str,
    conversation_context: Sequence[str] | None = None,
) -> QueryRewriteDecision:
    decision, _, _ = _llm_decide_and_rewrite(
        query,
        conversation_context=conversation_context,
    )
    return decision


def rewrite_query_for_retrieval(
    query: str,
    conversation_context: Sequence[str] | None = None,
    rewrite_decision: QueryRewriteDecision | None = None,
) -> QueryRewriteResult:
    cleaned_history = _clean_conversation_context(conversation_context)
    history_available = bool(cleaned_history)

    if rewrite_decision is None:
        decision, candidate_query, llm_used = _llm_decide_and_rewrite(
            query,
            conversation_context=cleaned_history,
        )
    else:
        decision = rewrite_decision
        llm_used = False
        if not settings.QUERY_REWRITE_ENABLED:
            candidate_query = decision.original_query
        elif decision.rewrite_needed:
            _, candidate_query, llm_used = _llm_decide_and_rewrite(
                decision.original_query,
                conversation_context=cleaned_history,
            )
        else:
            candidate_query = decision.original_query

    original_query = decision.original_query
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

    if not settings.QUERY_REWRITE_ENABLED:
        result = QueryRewriteResult(
            original_query=original_query,
            rewrite_query=original_query,
            rewrite_needed=False,
            rule_score=0,
            reasons=["rewrite_disabled"],
            history_available=history_available,
            llm_used=llm_used,
        )
        _print_rewrite_result(result)
        return result

    rewrite_query = original_query
    rewrite_applied = False

    if rewrite_needed and _is_safe_rewrite(original_query, candidate_query, cleaned_history):
        rewrite_query = candidate_query
        rewrite_applied = rewrite_query.casefold() != original_query.casefold()
    elif rewrite_needed and not _is_safe_rewrite(original_query, candidate_query, cleaned_history):
        logger.info(
            "Reject unsafe query rewrite original=%r candidate=%r",
            original_query,
            candidate_query,
        )
        rewrite_needed = False
        reasons = reasons + ["unsafe_rewrite_rejected"]
        rule_score = 0

    result = QueryRewriteResult(
        original_query=original_query,
        rewrite_query=rewrite_query,
        rewrite_needed=rewrite_needed,
        rule_score=rule_score,
        reasons=reasons,
        history_available=history_available,
        llm_used=llm_used,
        rewrite_applied=rewrite_applied,
    )
    _print_rewrite_result(result)
    return result
