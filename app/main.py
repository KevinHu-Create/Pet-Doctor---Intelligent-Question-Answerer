from fastapi import FastAPI
from app.routes.health import router as health_router
from app.routes.qa import router as qa_router
from app.routes.auth import router as auth_router
from app.routes.user import router as user_router
from app.db.database import engine
from app.db.models import Base


Base.metadata.create_all(bind=engine)
app = FastAPI(title="Pet Doctor RAG API", version="1.0")

app.include_router(health_router)
app.include_router(qa_router)
app.include_router(auth_router)
app.include_router(user_router)