# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from deps import get_rag_chain

app = FastAPI(title="Pet Doctor RAG API", version="1.0")

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is empty")

    chain = get_rag_chain()
    ans = chain.invoke(q)
    return AskResponse(answer=ans)