from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.rag_service import answer_question

router = APIRouter(tags=["qa"])

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    ans = answer_question(q)
    return AskResponse(answer=ans)