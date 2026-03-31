from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps.auth import require_current_user
from app.db.models import User
from app.pipeline.query_rewrite import inspect_query_rewrite, rewrite_query_for_retrieval
from app.services.history_service import append_conversation_turn, get_rewrite_context
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

    rewrite_decision = inspect_query_rewrite(q)
    rewrite_context = (
        get_rewrite_context(user.id) if rewrite_decision.rewrite_needed else []
    )
    rewrite_result = rewrite_query_for_retrieval(
        q,
        conversation_context=rewrite_context,
        rewrite_decision=rewrite_decision,
    )
    ans = answer_question(
        question=q,
        rewrite_query=rewrite_result.rewrite_query,
    )
    append_conversation_turn(user.id, q, ans)
    return AskResponse(answer=ans)
