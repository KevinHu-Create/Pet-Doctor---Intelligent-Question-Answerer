from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.qa import router as qa_router

app = FastAPI(title="Pet Doctor RAG API", version="1.0")

app.include_router(health_router)
app.include_router(qa_router)