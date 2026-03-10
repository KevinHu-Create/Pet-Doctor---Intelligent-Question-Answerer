from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.qa import router as qa_router
from app.routes.auth import router as auth_router
from app.routes.admin import router as admin_router

app = FastAPI(title="Pet Doctor RAG API", version="1.0")

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(qa_router)
app.include_router(admin_router)