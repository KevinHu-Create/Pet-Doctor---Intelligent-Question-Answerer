from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.deps.auth import require_current_user
from app.db.models import User
from app.services.rag_service import answer_question

router = APIRouter(tags=["qa"])

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _: User = Depends(require_current_user)):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    ans = answer_question(q)
    return AskResponse(answer=ans)
