from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps.auth import require_current_user
from app.db.models import User
from app.pipeline.query_rewrite import rewrite_query_for_retrieval
from app.services.history_service import (
    append_conversation_turn,
    get_history_material,
    update_topic_tracking,
)
from app.services.rag_service import answer_question

router = APIRouter(tags=["qa"])

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: User = Depends(require_current_user)):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    history_material = get_history_material(
        user.id,
        pet_type=user.pet_type,
        pet_name=user.pet_name,
    )
    rewrite_result = rewrite_query_for_retrieval(
        q,
        conversation_context=history_material.build_rewrite_context(q),
    )
    answer_query = rewrite_result.rewrite_query if rewrite_result.rewrite_applied else q
    ans = answer_question(
        question=answer_query,
        rewrite_query=rewrite_result.rewrite_query,
    )
    append_conversation_turn(user.id, q, ans)
    update_topic_tracking(
        user.id,
        q,
        rewrite_result.rewrite_query,
        pet_type=user.pet_type,
        pet_name=user.pet_name,
    )
    return AskResponse(answer=ans)
