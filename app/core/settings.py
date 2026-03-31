import os
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _get_bool_env(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

class Settings(BaseSettings):
    database_url: str = "sqlite:///./users.db"
    admin_database_url: str = "sqlite:///./admins.db"
    session_secret_key: str = os.getenv("SESSION_SECRET_KEY", "pet-doctor-dev-secret")
    admin_seed_username: str = os.getenv("ADMIN_SEED_USERNAME", "admin")
    admin_seed_email: str = os.getenv("ADMIN_SEED_EMAIL", "admin@petdoctor.com")
    admin_seed_password: str = os.getenv("ADMIN_SEED_PASSWORD", "admin123456")
    DATA_DIR: Path = os.getenv("DATA_DIR", PROJECT_ROOT / "data")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    HF_CACHE_DIR: Path = Path(
        os.getenv("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")
    )
    HF_HUB_ETAG_TIMEOUT: int = int(os.getenv("HF_HUB_ETAG_TIMEOUT", "60"))
    HF_HUB_DOWNLOAD_TIMEOUT: int = int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "120"))
    HF_EMBED_MODEL: str = os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3")
    TORCH_DEVICE: str = os.getenv("TORCH_DEVICE", "auto")
    RAG_DENSE_TOP_K: int = int(os.getenv("RAG_DENSE_TOP_K", "12"))
    RAG_RERANK_TOP_N: int = int(os.getenv("RAG_RERANK_TOP_N", "4"))
    RAG_RERANK_MODEL: str = os.getenv(
        "RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
    )
    QUERY_REWRITE_ENABLED: bool = _get_bool_env("QUERY_REWRITE_ENABLED", "true")
    QUERY_REWRITE_RULE_THRESHOLD: int = int(
        os.getenv("QUERY_REWRITE_RULE_THRESHOLD", "2")
    )
    HISTORY_RECENT_TURNS_LIMIT: int = int(
        os.getenv("HISTORY_RECENT_TURNS_LIMIT", "10")
    )
    OLLAMA_CHAT_MODEL: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "dog_owner_handbook")

settings = Settings()
