from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.settings import settings

def _engine_kwargs(database_url: str) -> dict:
    if database_url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {}


engine = create_engine(settings.database_url, echo=False, **_engine_kwargs(settings.database_url))
admin_engine = create_engine(
    settings.admin_database_url,
    echo=False,
    **_engine_kwargs(settings.admin_database_url),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AdminSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=admin_engine)
Base = declarative_base()
AdminBase = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_admin_db():
    db = AdminSessionLocal()
    try:
        yield db
    finally:
        db.close()
