import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.prompts import PromptTemplate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app
from app.pipeline import query_rewrite
from app.pipeline.query_rewrite import QueryRewriteDecision
from app.routes import qa
from app.services import history_service


class StubChain:
    def __init__(self, handler):
        self._handler = handler

    def invoke(self, payload):
        return self._handler(payload)


def _topic_payload(**overrides):
    payload = {key: "none" for key in history_service._TRACKED_FIELD_ORDER}
    payload.update(overrides)
    return json.dumps(payload)


def _summary_payload(**overrides):
    payload = {key: "none" for key in history_service._SUMMARY_FIELD_ORDER}
    payload.update(overrides)
    return json.dumps(payload)


def _rewrite_payload(*, rewrite_needed, reason, rewritten_query):
    return json.dumps(
        {
            "rewrite_needed": rewrite_needed,
            "reason": reason,
            "rewritten_query": rewritten_query,
        }
    )


@pytest.fixture(autouse=True)
def reset_history_state():
    history_service._RECENT_TURNS_BY_USER.clear()
    history_service._STRUCTURED_SUMMARIES_BY_USER.clear()
    history_service._TOPIC_STATE_BY_USER.clear()
    history_service._TOPIC_TURN_INDEX_BY_USER.clear()
    history_service._get_summary_chain.cache_clear()
    history_service._get_topic_extraction_chain.cache_clear()
    app.dependency_overrides.clear()
    yield
    history_service._RECENT_TURNS_BY_USER.clear()
    history_service._STRUCTURED_SUMMARIES_BY_USER.clear()
    history_service._TOPIC_STATE_BY_USER.clear()
    history_service._TOPIC_TURN_INDEX_BY_USER.clear()
    history_service._get_summary_chain.cache_clear()
    history_service._get_topic_extraction_chain.cache_clear()
    app.dependency_overrides.clear()


def test_history_material_seeds_profile_and_orders_tracked_context(monkeypatch):
    def extraction_handler(payload):
        assert payload["tracked_snapshot"] == "pet_type=dog\npet_name=Buddy"
        return _topic_payload(
            symptom=[{"text": "diarrhea", "status": "active"}],
        )

    monkeypatch.setattr(
        history_service,
        "_get_topic_extraction_chain",
        lambda: StubChain(extraction_handler),
    )

    history_service.update_topic_tracking(
        1,
        "My dog Buddy has diarrhea",
        "dog Buddy has diarrhea",
        pet_type="dog",
        pet_name="Buddy",
    )
    history_service._STRUCTURED_SUMMARIES_BY_USER[1] = history_service._summary_state_from_items(
        {
            "user_reported_facts": ["owner reports appetite loss"],
            "timeline_events": ["symptoms started this morning"],
        }
    )
    history_service.append_conversation_turn(
        1,
        "My dog Buddy has diarrhea",
        "Keep an eye on hydration.",
    )

    material = history_service.get_history_material(1, pet_type="dog", pet_name="Buddy")

    assert material.tracked_context == [
        "pet_type=dog",
        "pet_name=Buddy",
        "symptom=diarrhea",
    ]
    recent_turn = "User: My dog Buddy has diarrhea\nAssistant: Keep an eye on hydration."
    assert material.rewrite_recent_turns == [recent_turn]
    assert material.rewrite_summary_items == [
        "memory.timeline_events=symptoms started this morning",
    ]
    assert material.rewrite_context == [
        "pet_type=dog",
        "pet_name=Buddy",
        "symptom=diarrhea",
        recent_turn,
        "memory.timeline_events=symptoms started this morning",
    ]
    assert material.summary_items == [
        "memory.user_reported_facts=owner reports appetite loss",
        "memory.timeline_events=symptoms started this morning",
    ]


def test_topic_tracking_accumulates_and_hides_resolved_items(monkeypatch):
    def extraction_handler(payload):
        question = payload["question"]
        if question == "My dog has diarrhea":
            return _topic_payload(
                symptom=[{"text": "diarrhea", "status": "active"}],
            )
        if question == "What about vomiting?":
            assert "symptom=diarrhea" in payload["tracked_snapshot"]
            return _topic_payload(
                symptom=[{"text": "vomiting", "status": "active"}],
            )
        if question == "She is not vomiting anymore":
            assert "symptom=diarrhea, vomiting" in payload["tracked_snapshot"]
            return _topic_payload(
                symptom=[{"text": "vomiting", "status": "resolved"}],
            )
        raise AssertionError(f"unexpected question payload: {question!r}")

    monkeypatch.setattr(
        history_service,
        "_get_topic_extraction_chain",
        lambda: StubChain(extraction_handler),
    )

    history_service.update_topic_tracking(
        2,
        "My dog has diarrhea",
        "dog has diarrhea",
        pet_type="dog",
    )
    history_service.update_topic_tracking(
        2,
        "What about vomiting?",
        "dog with diarrhea and vomiting",
        pet_type="dog",
    )

    second_turn_material = history_service.get_history_material(2, pet_type="dog")
    assert second_turn_material.tracked_context == [
        "pet_type=dog",
        "symptom=diarrhea, vomiting",
    ]

    history_service.update_topic_tracking(
        2,
        "She is not vomiting anymore",
        "dog has diarrhea and is not vomiting anymore",
        pet_type="dog",
    )

    final_material = history_service.get_history_material(2, pet_type="dog")
    assert final_material.tracked_context == [
        "pet_type=dog",
        "symptom=diarrhea",
    ]

    symptom_items = history_service._TOPIC_STATE_BY_USER[2]["symptom"].items
    assert [(item.text, item.status, item.mention_count) for item in symptom_items] == [
        ("diarrhea", "active", 1),
        ("vomiting", "resolved", 2),
    ]


@pytest.mark.parametrize(
    "mode",
    ["bad_json", "chain_error"],
)
def test_topic_tracking_failures_do_not_raise_or_clear_seed(monkeypatch, mode):
    def extraction_handler(_payload):
        if mode == "bad_json":
            return "not valid json"
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr(
        history_service,
        "_get_topic_extraction_chain",
        lambda: StubChain(extraction_handler),
    )

    history_service.update_topic_tracking(
        3,
        "My dog has diarrhea",
        "dog has diarrhea",
        pet_type="dog",
    )

    material = history_service.get_history_material(3, pet_type="dog")
    assert material.tracked_context == ["pet_type=dog"]


def test_recent_turn_rollup_builds_structured_summary(monkeypatch):
    monkeypatch.setattr(history_service.settings, "HISTORY_RECENT_TURNS_LIMIT", 2)
    monkeypatch.setattr(
        history_service,
        "_get_summary_chain",
        lambda: StubChain(
            lambda payload: (
                _summary_payload(
                    user_reported_facts=["dog had diarrhea", "dog became lethargic"],
                    timeline_events=["symptoms started yesterday"],
                    user_actions_taken=["owner switched to bland food"],
                    risk_flags=["watch for dehydration"],
                )
            )
        ),
    )

    history_service.append_conversation_turn(10, "My dog has diarrhea", "Offer water.")
    history_service.append_conversation_turn(10, "He is now lethargic", "Monitor closely.")
    history_service.append_conversation_turn(10, "He ate some rice", "That can be okay.")

    material = history_service.get_history_material(10)

    assert material.summary_items == [
        "memory.user_reported_facts=dog had diarrhea; dog became lethargic",
        "memory.timeline_events=symptoms started yesterday",
        "memory.user_actions_taken=owner switched to bland food",
        "memory.risk_flags=watch for dehydration",
    ]
    assert material.rewrite_summary_items == [
        "memory.timeline_events=symptoms started yesterday",
        "memory.user_actions_taken=owner switched to bland food",
        "memory.risk_flags=watch for dehydration",
    ]
    assert material.recent_turns == [
        "User: He ate some rice\nAssistant: That can be okay."
    ]


@pytest.mark.parametrize("summary_mode", ["bad_json", "chain_error"])
def test_summary_fallback_keeps_structured_memory(summary_mode, monkeypatch):
    monkeypatch.setattr(history_service.settings, "HISTORY_RECENT_TURNS_LIMIT", 2)

    def summary_handler(_payload):
        if summary_mode == "bad_json":
            return "not valid json"
        raise RuntimeError("summary llm unavailable")

    monkeypatch.setattr(
        history_service,
        "_get_summary_chain",
        lambda: StubChain(summary_handler),
    )

    history_service.append_conversation_turn(11, "My dog has diarrhea", "Offer water.")
    history_service.append_conversation_turn(11, "Should I worry?", "Watch hydration.")
    history_service.append_conversation_turn(11, "He is still active", "Continue monitoring.")

    material = history_service.get_history_material(11)

    assert material.summary_items == [
        "memory.user_reported_facts=My dog has diarrhea; Should I worry?",
        "memory.assistant_guidance_given=Offer water.; Watch hydration.",
        "memory.open_questions=Should I worry?",
    ]
    assert material.rewrite_summary_items == []
    assert material.recent_turns == [
        "User: He is still active\nAssistant: Continue monitoring."
    ]


def test_rewrite_context_projects_topic_state_and_limits_recent_turns():
    topic_state = history_service._empty_topic_state()
    topic_state["pet_type"].items = [history_service.TrackedItem(text="dog", last_updated_turn=0)]
    topic_state["symptom"].items = [
        history_service.TrackedItem(
            text="diarrhea",
            status="active",
            last_updated_turn=1,
            mention_count=4,
        ),
        history_service.TrackedItem(
            text="vomiting",
            status="active",
            last_updated_turn=4,
            mention_count=1,
        ),
        history_service.TrackedItem(
            text="lethargy",
            status="active",
            last_updated_turn=3,
            mention_count=2,
        ),
        history_service.TrackedItem(
            text="coughing",
            status="active",
            last_updated_turn=2,
            mention_count=1,
        ),
    ]
    history_service._TOPIC_STATE_BY_USER[12] = topic_state
    history_service._STRUCTURED_SUMMARIES_BY_USER[12] = history_service._summary_state_from_items(
        {
            "timeline_events": ["symptoms started yesterday"],
            "user_actions_taken": ["owner switched to bland food"],
            "assistant_guidance_given": ["watch hydration carefully"],
        }
    )
    history_service._RECENT_TURNS_BY_USER[12] = [
        "User: first turn\nAssistant: first answer",
        "User: second turn\nAssistant: second answer",
        "User: third turn\nAssistant: third answer",
    ]

    material = history_service.get_history_material(12, pet_type="dog")
    rewrite_context = material.build_rewrite_context("What about vomiting?")

    assert rewrite_context == [
        "pet_type=dog",
        "symptom=vomiting, lethargy, coughing",
        "User: second turn\nAssistant: second answer",
        "User: third turn\nAssistant: third answer",
        "memory.timeline_events=symptoms started yesterday",
        "memory.user_actions_taken=owner switched to bland food",
    ]


def test_rewrite_query_can_use_tracked_symptom_context(monkeypatch):
    def extraction_handler(payload):
        assert payload["tracked_snapshot"] == "pet_type=dog"
        return _topic_payload(
            symptom=[{"text": "diarrhea", "status": "active"}],
        )

    monkeypatch.setattr(
        history_service,
        "_get_topic_extraction_chain",
        lambda: StubChain(extraction_handler),
    )

    history_service.update_topic_tracking(
        4,
        "My dog has diarrhea",
        "dog has diarrhea",
        pet_type="dog",
    )

    captured = {}

    def rewrite_handler(payload):
        captured.update(payload)
        return _rewrite_payload(
            rewrite_needed=True,
            reason="follow_up_missing_topic",
            rewritten_query="dog with diarrhea and vomiting",
        )

    monkeypatch.setattr(
        query_rewrite,
        "_get_rewrite_chain",
        lambda: StubChain(rewrite_handler),
    )

    result = query_rewrite.rewrite_query_for_retrieval(
        "What about vomiting?",
        conversation_context=history_service.get_rewrite_context(
            4,
            query="What about vomiting?",
            pet_type="dog",
        ),
        rewrite_decision=QueryRewriteDecision(
            original_query="What about vomiting?",
            rewrite_needed=True,
            rule_score=1,
            reasons=["follow_up_missing_topic"],
        ),
    )

    assert result.rewrite_query == "dog with diarrhea and vomiting"
    assert result.rewrite_applied is True
    assert "pet_type=dog" in captured["history"]
    assert "symptom=diarrhea" in captured["history"]


def test_rewrite_query_can_use_tracked_event_context(monkeypatch):
    def extraction_handler(_payload):
        return _topic_payload(
            event=[{"text": "ate chocolate", "status": "active"}],
            food_or_toxin=[{"text": "chocolate", "status": "active"}],
        )

    monkeypatch.setattr(
        history_service,
        "_get_topic_extraction_chain",
        lambda: StubChain(extraction_handler),
    )

    history_service.update_topic_tracking(
        5,
        "My dog ate chocolate this morning",
        "dog ate chocolate this morning",
        pet_type="dog",
    )

    captured = {}

    def rewrite_handler(payload):
        captured.update(payload)
        return _rewrite_payload(
            rewrite_needed=True,
            reason="ambiguous_reference",
            rewritten_query="Is eating chocolate dangerous for a dog?",
        )

    monkeypatch.setattr(
        query_rewrite,
        "_get_rewrite_chain",
        lambda: StubChain(rewrite_handler),
    )

    result = query_rewrite.rewrite_query_for_retrieval(
        "Is it dangerous?",
        conversation_context=history_service.get_rewrite_context(
            5,
            query="Is it dangerous?",
            pet_type="dog",
        ),
        rewrite_decision=QueryRewriteDecision(
            original_query="Is it dangerous?",
            rewrite_needed=True,
            rule_score=1,
            reasons=["ambiguous_reference"],
        ),
    )

    assert result.rewrite_query == "Is eating chocolate dangerous for a dog?"
    assert result.rewrite_applied is True
    assert "event=ate chocolate" in captured["history"]
    assert "food_or_toxin=chocolate" in captured["history"]


def test_inspect_query_rewrite_is_llm_driven(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_get_rewrite_chain",
        lambda: StubChain(
            lambda payload: _rewrite_payload(
                rewrite_needed=True,
                reason="missing_specific_detail",
                rewritten_query="dog has diarrhea and vomiting",
            )
        ),
    )

    decision = query_rewrite.inspect_query_rewrite(
        "What about vomiting?",
        conversation_context=["pet_type=dog", "symptom=diarrhea"],
    )

    assert decision.rewrite_needed is True
    assert decision.rule_score == 1
    assert decision.reasons == ["missing_specific_detail"]


def test_llm_can_decide_no_rewrite(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_get_rewrite_chain",
        lambda: StubChain(
            lambda payload: _rewrite_payload(
                rewrite_needed=False,
                reason="already_specific",
                rewritten_query=payload["question"],
            )
        ),
    )

    result = query_rewrite.rewrite_query_for_retrieval(
        "What are signs of dehydration in dogs?",
        conversation_context=[],
    )

    assert result.rewrite_needed is False
    assert result.rewrite_query == "What are signs of dehydration in dogs?"
    assert result.rewrite_applied is False
    assert result.llm_used is True


def test_llm_rewrite_failure_falls_back_to_original(monkeypatch):
    monkeypatch.setattr(
        query_rewrite,
        "_get_rewrite_chain",
        lambda: StubChain(lambda payload: "not valid json"),
    )

    result = query_rewrite.rewrite_query_for_retrieval(
        "Is it dangerous?",
        conversation_context=["pet_type=dog", "event=ate chocolate"],
    )

    assert result.rewrite_needed is False
    assert result.rewrite_query == "Is it dangerous?"
    assert result.rewrite_applied is False
    assert "llm_failure" in result.reasons


def test_rewrite_prompt_template_escapes_json_examples():
    prompt = PromptTemplate(
        template=query_rewrite._LLM_REWRITE_PROMPT_TEMPLATE,
        input_variables=["history", "question"],
    )

    rendered = prompt.format(history="pet_type=dog", question="Is it dangerous?")

    assert '"rewrite_needed"' in rendered
    assert '{"rewrite_needed": true' in rendered


def test_topic_tracking_prompt_template_escapes_json_examples():
    prompt = PromptTemplate(
        template=history_service._TOPIC_EXTRACTION_PROMPT_TEMPLATE,
        input_variables=["tracked_snapshot", "question", "rewrite_query"],
    )

    rendered = prompt.format(
        tracked_snapshot="pet_type=dog",
        question="Is it dangerous?",
        rewrite_query="Is eating chocolate dangerous for a dog?",
    )

    assert '{"text": "...", "status": "active"|"resolved"}' in rendered


def test_ask_route_keeps_api_shape_and_updates_topic_tracking(monkeypatch):
    call_log: list[str] = []

    app.dependency_overrides[qa.require_current_user] = lambda: SimpleNamespace(
        id=99,
        pet_type="dog",
        pet_name="Buddy",
    )

    monkeypatch.setattr(
        qa,
        "get_history_material",
        lambda user_id, pet_type=None, pet_name=None: call_log.append("get_history_material")
        or SimpleNamespace(build_rewrite_context=lambda query: ["pet_type=dog", "pet_name=Buddy"]),
    )
    monkeypatch.setattr(
        qa,
        "rewrite_query_for_retrieval",
        lambda question, conversation_context=None: call_log.append(
            "rewrite_query_for_retrieval"
        )
        or SimpleNamespace(
            rewrite_query="Is eating chocolate dangerous for a dog?",
            rewrite_applied=True,
        ),
    )
    monkeypatch.setattr(
        qa,
        "answer_question",
        lambda question, rewrite_query=None: call_log.append("answer_question") or "answer",
    )
    monkeypatch.setattr(
        qa,
        "append_conversation_turn",
        lambda user_id, question, answer: call_log.append("append_conversation_turn"),
    )
    monkeypatch.setattr(
        qa,
        "update_topic_tracking",
        lambda user_id, question, rewrite_query, pet_type=None, pet_name=None: call_log.append(
            "update_topic_tracking"
        ),
    )

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Is it dangerous?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "answer"}
    assert call_log == [
        "get_history_material",
        "rewrite_query_for_retrieval",
        "answer_question",
        "append_conversation_turn",
        "update_topic_tracking",
    ]
