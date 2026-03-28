from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.health import router as health_router
from app.routes.qa import router as qa_router
from app.routes.auth import router as auth_router
from app.routes.user import router as user_router
from app.core.settings import settings
from app.db.database import AdminSessionLocal, admin_engine, engine
from app.db.models import AdminBase, Base
from app.services.user_service import seed_admin


Base.metadata.create_all(bind=engine)
AdminBase.metadata.create_all(bind=admin_engine)

with AdminSessionLocal() as admin_db:
    seed_admin(admin_db)

app = FastAPI(title="Pet Doctor RAG API", version="1.0")
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret_key,
    same_site="lax",
    https_only=False,
)
STATIC_DIR = Path(__file__).resolve().parent / "static"

app.include_router(health_router)
app.include_router(qa_router)
app.include_router(auth_router)
app.include_router(user_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/chat", include_in_schema=False)
def chat_page():
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/profile", include_in_schema=False)
def profile_page():
    return FileResponse(STATIC_DIR / "profile.html")


@app.get("/admin", include_in_schema=False)
def admin_page():
    return FileResponse(STATIC_DIR / "admin.html")
