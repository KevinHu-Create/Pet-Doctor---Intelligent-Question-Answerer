from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from threading import RLock
from typing import Mapping

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from app.core.settings import settings
from app.deps.container import get_llm

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_USER_TURN_RE = re.compile(r"User:\s*(.+?)(?:\s*Assistant:|$)", re.IGNORECASE)
_ASSISTANT_TURN_RE = re.compile(r"Assistant:\s*(.+)$", re.IGNORECASE)
_PROJECTION_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_ACTIVE_STATUS = "active"
_RESOLVED_STATUS = "resolved"
_NONE_SENTINELS = {"", "none", "null", "n/a"}
_TRACKED_FIELD_ORDER = (
    "pet_type",
    "pet_name",
    "symptom",
    "event",
    "food_or_toxin",
    "medication",
    "timeline",
    "severity",
)
_SUMMARY_FIELD_ORDER = (
    "user_reported_facts",
    "timeline_events",
    "user_actions_taken",
    "assistant_guidance_given",
    "state_transitions",
    "open_questions",
    "risk_flags",
)
_SUMMARY_ITEM_LIMIT = 6
_REWRITE_RECENT_TURNS_LIMIT = 2
_TOPIC_REWRITE_LIMITS = {
    "symptom": 3,
    "event": 2,
    "food_or_toxin": 2,
    "medication": 2,
}
_SUMMARY_REWRITE_LIMITS = {
    "user_actions_taken": 2,
    "timeline_events": 2,
    "state_transitions": 1,
    "risk_flags": 1,
}
_SINGLE_VALUE_FIELDS = frozenset({"pet_type", "pet_name", "timeline", "severity"})
_FIELD_MODES = {
    key: ("single" if key in _SINGLE_VALUE_FIELDS else "multi")
    for key in _TRACKED_FIELD_ORDER
}

_SUMMARY_PROMPT_TEMPLATE = """
You maintain a structured long-term memory table for a pet-health assistant.

Memory table fields:
- user_reported_facts
- timeline_events
- user_actions_taken
- assistant_guidance_given
- state_transitions
- open_questions
- risk_flags

Rules:
- Merge the existing memory table with the archived conversation turns.
- Preserve durable facts, timelines, actions, guidance, state changes, open questions, and risk flags that may matter later.
- Keep each item concise and concrete.
- Keep at most 6 items per field.
- If a field has no useful information, output "none" for that field.
- Output exactly one valid JSON object with all fields above.
- Each populated field must be an array of strings.
- Do not add new facts.
- Output JSON only.

Existing memory table:
{memory_snapshot}

Archived conversation turns:
{turns}

JSON:
""".strip()

_TOPIC_EXTRACTION_PROMPT_TEMPLATE = """
You extract incremental topic-tracking updates for a pet-health assistant.

Tracked fields:
- single fields: pet_type, pet_name, timeline, severity
- multi fields: symptom, event, food_or_toxin, medication

Rules:
- Use only the current user question and rewritten query as evidence.
- Use the previous tracked snapshot only to resolve pronouns or follow-up references.
- Output exactly one valid JSON object with these keys:
  pet_type, pet_name, symptom, event, food_or_toxin, medication, timeline, severity
- For single fields, output either {{"text": "...", "status": "active"|"resolved"}} or "none".
- For multi fields, output either an array of {{"text": "...", "status": "active"|"resolved"}} objects or "none".
- If the current turn does not update a field, output "none" for that field.
- Use "resolved" only when the user clearly says the issue stopped, is gone, is no longer happening, or is explicitly negated now.
- Do not infer diagnoses, treatments, or facts that are not explicitly present.
- Preserve the user's concrete wording when possible.
- Output JSON only.

Previous tracked snapshot:
{tracked_snapshot}

Current user question:
{question}

Current rewritten query:
{rewrite_query}

JSON:
""".strip()


@dataclass(slots=True)
class TrackedItem:
    text: str
    status: str = _ACTIVE_STATUS
    last_updated_turn: int = 0
    mention_count: int = 1


@dataclass(slots=True)
class TrackedField:
    key: str
    mode: str
    items: list[TrackedItem] = field(default_factory=list)

    @property
    def active_texts(self) -> list[str]:
        return [item.text for item in self.items if item.status == _ACTIVE_STATUS and item.text]


@dataclass(slots=True)
class StructuredSummaryField:
    key: str
    items: list[str] = field(default_factory=list)


@dataclass(slots=True)
class HistoryMaterial:
    recent_turns: list[str] = field(default_factory=list)
    tracked_fields: list[TrackedField] = field(default_factory=list)
    structured_summary_fields: list[StructuredSummaryField] = field(default_factory=list)

    @property
    def tracked_context(self) -> list[str]:
        return _tracked_fields_to_context(self.tracked_fields)

    @property
    def summary_items(self) -> list[str]:
        return _summary_fields_to_context(self.structured_summary_fields)

    @property
    def rewrite_recent_turns(self) -> list[str]:
        return list(self.recent_turns[-_REWRITE_RECENT_TURNS_LIMIT:])

    @property
    def rewrite_summary_items(self) -> list[str]:
        return _summary_fields_to_context(
            self.structured_summary_fields,
            allowed_keys=set(_SUMMARY_REWRITE_LIMITS),
            limits_by_key=_SUMMARY_REWRITE_LIMITS,
        )

    def projected_tracked_context(self, query: str | None = None) -> list[str]:
        return _project_tracked_fields_to_context(self.tracked_fields, query=query)

    def build_rewrite_context(self, query: str | None = None) -> list[str]:
        context = self.projected_tracked_context(query)
        context.extend(self.rewrite_recent_turns)
        context.extend(self.rewrite_summary_items)
        return context

    @property
    def rewrite_context(self) -> list[str]:
        return self.build_rewrite_context()


_RECENT_TURNS_BY_USER: dict[int, list[str]] = {}
_STRUCTURED_SUMMARIES_BY_USER: dict[int, dict[str, StructuredSummaryField]] = {}
_TOPIC_STATE_BY_USER: dict[int, dict[str, TrackedField]] = {}
_TOPIC_TURN_INDEX_BY_USER: dict[int, int] = {}
_STORE_LOCK = RLock()


def _normalize_text(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "").strip())


def _normalize_status(status: str | None) -> str:
    return _RESOLVED_STATUS if _normalize_text(status).lower() == _RESOLVED_STATUS else _ACTIVE_STATUS


def _normalized_item_key(text: str | None) -> str:
    return _normalize_text(text).casefold()


def _dedupe_text_items(items: list[str], *, limit: int | None = None) -> list[str]:
    ordered_keys: list[str] = []
    deduped: dict[str, str] = {}

    for item in items:
        normalized_item = _normalize_text(item)
        if not normalized_item:
            continue
        normalized_key = normalized_item.casefold()
        if normalized_key not in deduped:
            ordered_keys.append(normalized_key)
        deduped[normalized_key] = normalized_item

    deduped_items = [deduped[key] for key in ordered_keys]
    if limit is not None:
        return deduped_items[-limit:]
    return deduped_items


def _copy_tracked_item(item: TrackedItem) -> TrackedItem:
    return TrackedItem(
        text=item.text,
        status=item.status,
        last_updated_turn=item.last_updated_turn,
        mention_count=item.mention_count,
    )


def _empty_topic_state() -> dict[str, TrackedField]:
    return {
        key: TrackedField(key=key, mode=_FIELD_MODES[key])
        for key in _TRACKED_FIELD_ORDER
    }


def _copy_topic_state(
    state: Mapping[str, TrackedField] | None = None,
) -> dict[str, TrackedField]:
    copied = _empty_topic_state()
    if not state:
        return copied

    for key in _TRACKED_FIELD_ORDER:
        source = state.get(key)
        if not source:
            continue
        copied[key].items = [_copy_tracked_item(item) for item in source.items]
    return copied


def _empty_summary_state() -> dict[str, StructuredSummaryField]:
    return {
        key: StructuredSummaryField(key=key)
        for key in _SUMMARY_FIELD_ORDER
    }


def _copy_summary_state(
    state: Mapping[str, StructuredSummaryField] | None = None,
) -> dict[str, StructuredSummaryField]:
    copied = _empty_summary_state()
    if not state:
        return copied

    for key in _SUMMARY_FIELD_ORDER:
        source = state.get(key)
        if not source:
            continue
        copied[key].items = list(source.items)
    return copied


def _summary_state_from_items(
    items_by_key: Mapping[str, list[str]] | None = None,
) -> dict[str, StructuredSummaryField]:
    summary_state = _empty_summary_state()
    if not items_by_key:
        return summary_state

    for key in _SUMMARY_FIELD_ORDER:
        summary_state[key].items = _dedupe_text_items(
            list(items_by_key.get(key, [])),
            limit=_SUMMARY_ITEM_LIMIT,
        )
    return summary_state


def _tracked_fields_to_context(tracked_fields: list[TrackedField]) -> list[str]:
    context: list[str] = []
    for tracked_field in tracked_fields:
        texts = tracked_field.active_texts
        if texts:
            context.append(f"{tracked_field.key}={', '.join(texts)}")
    return context


def _projection_tokens(text: str | None) -> list[str]:
    normalized_text = _normalize_text(text)
    return [
        token.lower()
        for token in _PROJECTION_TOKEN_RE.findall(normalized_text)
        if len(token) >= 3
    ]


def _overlap_score(item_text: str, query_tokens: set[str]) -> int:
    if not query_tokens:
        return 0
    return len(set(_projection_tokens(item_text)) & query_tokens)


def _item_score(item: TrackedItem, query_tokens: set[str]) -> tuple[int, int, int, int]:
    return (
        1 if item.status == _ACTIVE_STATUS else 0,
        _overlap_score(item.text, query_tokens),
        item.last_updated_turn,
        item.mention_count,
    )


def _project_tracked_fields_to_context(
    tracked_fields: list[TrackedField],
    *,
    query: str | None = None,
) -> list[str]:
    query_tokens = set(_projection_tokens(query))
    context: list[str] = []

    for tracked_field in tracked_fields:
        if tracked_field.mode == "single":
            texts = tracked_field.active_texts
            if texts:
                context.append(f"{tracked_field.key}={', '.join(texts)}")
            continue

        active_items = [
            item
            for item in tracked_field.items
            if item.status == _ACTIVE_STATUS and _normalize_text(item.text)
        ]
        if not active_items:
            continue

        projected_items = sorted(
            active_items,
            key=lambda item: _item_score(item, query_tokens),
            reverse=True,
        )
        limit = _TOPIC_REWRITE_LIMITS.get(tracked_field.key, len(projected_items))
        projected_texts = [item.text for item in projected_items[:limit] if item.text]
        if projected_texts:
            context.append(f"{tracked_field.key}={', '.join(projected_texts)}")

    return context


def _summary_fields_to_context(
    summary_fields: list[StructuredSummaryField],
    *,
    allowed_keys: set[str] | frozenset[str] | None = None,
    prefix: str = "memory.",
    limits_by_key: Mapping[str, int] | None = None,
) -> list[str]:
    context: list[str] = []
    for summary_field in summary_fields:
        if allowed_keys is not None and summary_field.key not in allowed_keys:
            continue
        item_limit = (
            limits_by_key.get(summary_field.key, _SUMMARY_ITEM_LIMIT)
            if limits_by_key is not None
            else _SUMMARY_ITEM_LIMIT
        )
        items = _dedupe_text_items(summary_field.items, limit=item_limit)
        if items:
            context.append(f"{prefix}{summary_field.key}={'; '.join(items)}")
    return context


def _seed_single_field(fields_by_key: dict[str, TrackedField], key: str, value: str | None) -> None:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return

    field = fields_by_key[key]
    if field.items:
        return

    field.items = [TrackedItem(text=normalized_value, status=_ACTIVE_STATUS, last_updated_turn=0)]


def _seed_topic_state(
    fields_by_key: dict[str, TrackedField],
    *,
    pet_type: str | None = None,
    pet_name: str | None = None,
) -> dict[str, TrackedField]:
    _seed_single_field(fields_by_key, "pet_type", pet_type)
    _seed_single_field(fields_by_key, "pet_name", pet_name)
    return fields_by_key


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
        input_variables=["memory_snapshot", "turns"],
    )
    return prompt | get_llm() | StrOutputParser()


@lru_cache
def _get_topic_extraction_chain():
    prompt = PromptTemplate(
        template=_TOPIC_EXTRACTION_PROMPT_TEMPLATE,
        input_variables=["tracked_snapshot", "question", "rewrite_query"],
    )
    return prompt | get_llm() | StrOutputParser()


def _format_summary_snapshot(summary_state: Mapping[str, StructuredSummaryField]) -> str:
    lines = _summary_fields_to_context(
        [summary_state[key] for key in _SUMMARY_FIELD_ORDER],
        allowed_keys=set(_SUMMARY_FIELD_ORDER),
        prefix="",
    )
    return "\n".join(lines) if lines else "(none)"


def _extract_json_object(raw: str) -> str:
    cleaned = _normalize_text(raw)
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.removeprefix("json").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("history extraction did not return a JSON object")
    return cleaned[start : end + 1]


def _parse_summary_field(raw_value: object) -> list[str]:
    if _is_none_value(raw_value) or raw_value is None:
        return []

    if isinstance(raw_value, str):
        return _dedupe_text_items([raw_value], limit=_SUMMARY_ITEM_LIMIT)

    if not isinstance(raw_value, list):
        raise ValueError(f"invalid summary field payload: {raw_value!r}")

    parsed_items = []
    for raw_item in raw_value:
        if _is_none_value(raw_item):
            continue
        if not isinstance(raw_item, str):
            raise ValueError(f"summary field item must be a string: {raw_item!r}")
        parsed_items.append(raw_item)

    return _dedupe_text_items(parsed_items, limit=_SUMMARY_ITEM_LIMIT)


def _parse_summary_state(
    raw: str,
    previous_state: Mapping[str, StructuredSummaryField],
) -> dict[str, StructuredSummaryField]:
    payload = json.loads(_extract_json_object(raw))
    if not isinstance(payload, dict):
        raise ValueError("summary payload must be a JSON object")

    items_by_key: dict[str, list[str]] = {}
    for key in _SUMMARY_FIELD_ORDER:
        if key in payload:
            items_by_key[key] = _parse_summary_field(payload[key])
        else:
            items_by_key[key] = list(previous_state.get(key, StructuredSummaryField(key=key)).items)
    return _summary_state_from_items(items_by_key)


def _fallback_summary_state(
    previous_state: Mapping[str, StructuredSummaryField],
    turns: list[str],
) -> dict[str, StructuredSummaryField]:
    items_by_key = {
        key: list(previous_state.get(key, StructuredSummaryField(key=key)).items)
        for key in _SUMMARY_FIELD_ORDER
    }

    for turn in turns:
        normalized_turn = _normalize_text(turn)
        if not normalized_turn:
            continue

        user_match = _USER_TURN_RE.search(normalized_turn)
        assistant_match = _ASSISTANT_TURN_RE.search(normalized_turn)

        if user_match:
            user_text = _normalize_text(user_match.group(1))
            if user_text:
                items_by_key["user_reported_facts"].append(user_text)
                if "?" in user_text:
                    items_by_key["open_questions"].append(user_text)

        if assistant_match:
            assistant_text = _normalize_text(assistant_match.group(1))
            if assistant_text:
                items_by_key["assistant_guidance_given"].append(assistant_text)

    return _summary_state_from_items(items_by_key)


def _summarize_turns(
    previous_state: Mapping[str, StructuredSummaryField] | None,
    turns: list[str],
) -> dict[str, StructuredSummaryField]:
    cleaned_turns = [_normalize_text(turn) for turn in turns if _normalize_text(turn)]
    current_state = _copy_summary_state(previous_state)
    if not cleaned_turns:
        return current_state

    try:
        raw = _get_summary_chain().invoke(
            {
                "memory_snapshot": _format_summary_snapshot(current_state),
                "turns": "\n\n".join(cleaned_turns),
            }
        )
        return _parse_summary_state(str(raw), current_state)
    except Exception:
        logger.exception("History summarization failed")
        return _fallback_summary_state(current_state, cleaned_turns)


def _format_tracked_snapshot(fields_by_key: Mapping[str, TrackedField]) -> str:
    lines = _tracked_fields_to_context([fields_by_key[key] for key in _TRACKED_FIELD_ORDER])
    return "\n".join(lines) if lines else "(none)"


def _is_none_value(value: object) -> bool:
    return isinstance(value, str) and _normalize_text(value).lower() in _NONE_SENTINELS


def _parse_delta_item(raw_item: object) -> TrackedItem | None:
    if _is_none_value(raw_item):
        return None

    if isinstance(raw_item, str):
        normalized_text = _normalize_text(raw_item)
        if not normalized_text:
            return None
        return TrackedItem(text=normalized_text, status=_ACTIVE_STATUS)

    if not isinstance(raw_item, dict):
        raise ValueError(f"invalid tracked item payload: {raw_item!r}")

    normalized_text = _normalize_text(raw_item.get("text"))
    if not normalized_text:
        raise ValueError(f"tracked item missing text: {raw_item!r}")

    return TrackedItem(
        text=normalized_text,
        status=_normalize_status(raw_item.get("status")),
    )


def _dedupe_delta_items(items: list[TrackedItem]) -> list[TrackedItem]:
    ordered_keys: list[str] = []
    deduped: dict[str, TrackedItem] = {}

    for item in items:
        normalized_key = _normalized_item_key(item.text)
        if not normalized_key:
            continue
        if normalized_key not in deduped:
            ordered_keys.append(normalized_key)
        deduped[normalized_key] = item

    return [deduped[key] for key in ordered_keys]


def _parse_delta_field(key: str, raw_value: object) -> list[TrackedItem]:
    if _is_none_value(raw_value) or raw_value is None:
        return []

    mode = _FIELD_MODES[key]
    if mode == "single" and isinstance(raw_value, list):
        raw_items = raw_value
    elif mode == "single":
        raw_items = [raw_value]
    elif isinstance(raw_value, list):
        raw_items = raw_value
    else:
        raw_items = [raw_value]

    parsed_items: list[TrackedItem] = []
    for raw_item in raw_items:
        parsed_item = _parse_delta_item(raw_item)
        if parsed_item:
            parsed_items.append(parsed_item)

    if mode == "single" and parsed_items:
        return [parsed_items[-1]]

    return _dedupe_delta_items(parsed_items)


def _parse_topic_delta(raw: str) -> dict[str, list[TrackedItem]]:
    payload = json.loads(_extract_json_object(raw))
    if not isinstance(payload, dict):
        raise ValueError("topic extraction payload must be a JSON object")

    return {
        key: _parse_delta_field(key, payload.get(key, "none"))
        for key in _TRACKED_FIELD_ORDER
    }


def _extract_topic_delta(
    tracked_snapshot: str,
    user_query: str,
    rewrite_query: str,
) -> dict[str, list[TrackedItem]] | None:
    normalized_question = _normalize_text(user_query)
    normalized_rewrite = _normalize_text(rewrite_query) or normalized_question
    if not normalized_question and not normalized_rewrite:
        return None

    try:
        raw = _get_topic_extraction_chain().invoke(
            {
                "tracked_snapshot": tracked_snapshot or "(none)",
                "question": normalized_question or "(none)",
                "rewrite_query": normalized_rewrite or "(none)",
            }
        )
        return _parse_topic_delta(str(raw))
    except Exception:
        logger.exception("Topic tracking extraction failed")
        return None


def _upsert_multi_item(field: TrackedField, incoming_item: TrackedItem, turn_index: int) -> None:
    normalized_key = _normalized_item_key(incoming_item.text)
    if not normalized_key:
        return

    for existing_item in field.items:
        if _normalized_item_key(existing_item.text) == normalized_key:
            existing_item.text = incoming_item.text
            existing_item.status = incoming_item.status
            existing_item.last_updated_turn = turn_index
            existing_item.mention_count += 1
            return

    field.items.append(
        TrackedItem(
            text=incoming_item.text,
            status=incoming_item.status,
            last_updated_turn=turn_index,
            mention_count=max(incoming_item.mention_count, 1),
        )
    )


def _apply_topic_delta(
    fields_by_key: dict[str, TrackedField],
    topic_delta: Mapping[str, list[TrackedItem]],
    turn_index: int,
) -> None:
    for key in _TRACKED_FIELD_ORDER:
        incoming_items = topic_delta.get(key, [])
        if not incoming_items:
            continue

        field = fields_by_key[key]
        if field.mode == "single":
            latest_item = incoming_items[-1]
            mention_count = max(latest_item.mention_count, 1)
            if field.items and _normalized_item_key(field.items[0].text) == _normalized_item_key(latest_item.text):
                mention_count = field.items[0].mention_count + 1
            field.items = [
                TrackedItem(
                    text=latest_item.text,
                    status=latest_item.status,
                    last_updated_turn=turn_index,
                    mention_count=mention_count,
                )
            ]
            continue

        for incoming_item in incoming_items:
            _upsert_multi_item(field, incoming_item, turn_index)


def _has_topic_updates(topic_delta: Mapping[str, list[TrackedItem]]) -> bool:
    return any(topic_delta.get(key) for key in _TRACKED_FIELD_ORDER)


def get_history_material(
    user_id: int,
    *,
    pet_type: str | None = None,
    pet_name: str | None = None,
) -> HistoryMaterial:
    with _STORE_LOCK:
        tracked_state = _seed_topic_state(
            _copy_topic_state(_TOPIC_STATE_BY_USER.get(user_id)),
            pet_type=pet_type,
            pet_name=pet_name,
        )
        summary_state = _copy_summary_state(_STRUCTURED_SUMMARIES_BY_USER.get(user_id))
        return HistoryMaterial(
            recent_turns=list(_RECENT_TURNS_BY_USER.get(user_id, [])),
            tracked_fields=[tracked_state[key] for key in _TRACKED_FIELD_ORDER],
            structured_summary_fields=[summary_state[key] for key in _SUMMARY_FIELD_ORDER],
        )


def get_rewrite_context(
    user_id: int,
    *,
    query: str | None = None,
    pet_type: str | None = None,
    pet_name: str | None = None,
) -> list[str]:
    return get_history_material(
        user_id,
        pet_type=pet_type,
        pet_name=pet_name,
    ).build_rewrite_context(query)


def append_conversation_turn(user_id: int, user_query: str, assistant_answer: str) -> None:
    formatted_turn = _format_turn(user_query, assistant_answer)
    if not formatted_turn:
        return

    turns_to_summarize: list[str] = []
    summary_state: dict[str, StructuredSummaryField] | None = None
    with _STORE_LOCK:
        recent_turns = _RECENT_TURNS_BY_USER.setdefault(user_id, [])
        if len(recent_turns) >= settings.HISTORY_RECENT_TURNS_LIMIT:
            turns_to_summarize = list(recent_turns)
            recent_turns.clear()
            summary_state = _copy_summary_state(_STRUCTURED_SUMMARIES_BY_USER.get(user_id))

    if turns_to_summarize:
        summarized_state = _summarize_turns(summary_state, turns_to_summarize)
        with _STORE_LOCK:
            _STRUCTURED_SUMMARIES_BY_USER[user_id] = summarized_state

    with _STORE_LOCK:
        _RECENT_TURNS_BY_USER.setdefault(user_id, []).append(formatted_turn)


def update_topic_tracking(
    user_id: int,
    user_query: str,
    rewrite_query: str,
    *,
    pet_type: str | None = None,
    pet_name: str | None = None,
) -> None:
    with _STORE_LOCK:
        tracked_state = _seed_topic_state(
            _copy_topic_state(_TOPIC_STATE_BY_USER.get(user_id)),
            pet_type=pet_type,
            pet_name=pet_name,
        )
        tracked_snapshot = _format_tracked_snapshot(tracked_state)

    topic_delta = _extract_topic_delta(tracked_snapshot, user_query, rewrite_query)
    if topic_delta is None or not _has_topic_updates(topic_delta):
        return

    with _STORE_LOCK:
        current_state = _seed_topic_state(
            _copy_topic_state(_TOPIC_STATE_BY_USER.get(user_id)),
            pet_type=pet_type,
            pet_name=pet_name,
        )
        next_turn_index = _TOPIC_TURN_INDEX_BY_USER.get(user_id, 0) + 1
        _TOPIC_TURN_INDEX_BY_USER[user_id] = next_turn_index
        _apply_topic_delta(current_state, topic_delta, next_turn_index)
        _TOPIC_STATE_BY_USER[user_id] = current_state


def clear_user_history(user_id: int) -> None:
    with _STORE_LOCK:
        _RECENT_TURNS_BY_USER.pop(user_id, None)
        _STRUCTURED_SUMMARIES_BY_USER.pop(user_id, None)
        _TOPIC_STATE_BY_USER.pop(user_id, None)
        _TOPIC_TURN_INDEX_BY_USER.pop(user_id, None)
