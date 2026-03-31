from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.settings import settings
from app.deps.container import get_llm

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_RECENT_TURNS_BY_USER: dict[int, list[str]] = {}
_SUMMARIES_BY_USER: dict[int, list[str]] = {}
_STORE_LOCK = RLock()

_SUMMARY_PROMPT_TEMPLATE = """
You summarize pet-health conversations for future query rewriting.

Rules:
- Preserve concrete entities, symptoms, events, foods, medications, timelines, and follow-up conclusions.
- Keep the summary concise but specific.
- Do not add new facts.
- Output only the summary text.

Conversation turns:
{turns}

Summary:
""".strip()


@dataclass(slots=True)
class HistoryMaterial:
    summary_items: list[str] = field(default_factory=list)
    recent_turns: list[str] = field(default_factory=list)

    @property
    def rewrite_context(self) -> list[str]:
        context: list[str] = []
        for idx, summary in enumerate(self.summary_items, start=1):
            context.append(f"Conversation summary {idx}: {summary}")
        context.extend(self.recent_turns)
        return context


def _normalize_text(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "").strip())


def _format_turn(user_query: str, assistant_answer: str) -> str:
    normalized_query = _normalize_text(user_query)
    normalized_answer = _normalize_text(assistant_answer)
    if not normalized_query and not normalized_answer:
        return ""
    return f"User: {normalized_query}\nAssistant: {normalized_answer}"


@lru_cache
def _get_summary_chain():
    prompt = PromptTemplate(
        template=_SUMMARY_PROMPT_TEMPLATE,
        input_variables=["turns"],
    )
    return prompt | get_llm() | StrOutputParser()


def _summarize_turns(turns: list[str]) -> str:
    cleaned_turns = [_normalize_text(turn) for turn in turns if _normalize_text(turn)]
    if not cleaned_turns:
        return ""

    try:
        raw = _get_summary_chain().invoke({"turns": "\n\n".join(cleaned_turns)})
    except Exception:
        logger.exception("History summarization failed")
        return _normalize_text(" ".join(cleaned_turns))[:800]

    summary = _normalize_text(str(raw))
    return summary or _normalize_text(" ".join(cleaned_turns))[:800]


def get_history_material(user_id: int) -> HistoryMaterial:
    with _STORE_LOCK:
        return HistoryMaterial(
            summary_items=list(_SUMMARIES_BY_USER.get(user_id, [])),
            recent_turns=list(_RECENT_TURNS_BY_USER.get(user_id, [])),
        )


def get_rewrite_context(user_id: int) -> list[str]:
    return get_history_material(user_id).rewrite_context


def append_conversation_turn(user_id: int, user_query: str, assistant_answer: str) -> None:
    formatted_turn = _format_turn(user_query, assistant_answer)
    if not formatted_turn:
        return

    turns_to_summarize: list[str] = []
    with _STORE_LOCK:
        recent_turns = _RECENT_TURNS_BY_USER.setdefault(user_id, [])
        if len(recent_turns) >= settings.HISTORY_RECENT_TURNS_LIMIT:
            turns_to_summarize = list(recent_turns)
            recent_turns.clear()

    if turns_to_summarize:
        summary = _summarize_turns(turns_to_summarize)
        if summary:
            with _STORE_LOCK:
                _SUMMARIES_BY_USER.setdefault(user_id, []).append(summary)

    with _STORE_LOCK:
        _RECENT_TURNS_BY_USER.setdefault(user_id, []).append(formatted_turn)


def clear_user_history(user_id: int) -> None:
    with _STORE_LOCK:
        _RECENT_TURNS_BY_USER.pop(user_id, None)
        _SUMMARIES_BY_USER.pop(user_id, None)
