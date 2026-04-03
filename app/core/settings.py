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
    BGE_M3_BATCH_SIZE: int = int(os.getenv("BGE_M3_BATCH_SIZE", "8"))
    BGE_M3_USE_FP16: bool = _get_bool_env("BGE_M3_USE_FP16", "false")
    RAG_DENSE_TOP_K: int = int(os.getenv("RAG_DENSE_TOP_K", "12"))
    RAG_RERANK_TOP_N: int = int(os.getenv("RAG_RERANK_TOP_N", "4"))
    RAG_HYBRID_ENABLED: bool = _get_bool_env(
        "RAG_HYBRID_ENABLED",
        "true" if "bge-m3" in os.getenv("HF_EMBED_MODEL", "BAAI/bge-m3").lower() else "false",
    )
    RAG_HYBRID_RANKER: str = os.getenv("RAG_HYBRID_RANKER", "weighted")
    RAG_HYBRID_DENSE_WEIGHT: float = float(
        os.getenv("RAG_HYBRID_DENSE_WEIGHT", "1.0")
    )
    RAG_HYBRID_SPARSE_WEIGHT: float = float(
        os.getenv("RAG_HYBRID_SPARSE_WEIGHT", "1.0")
    )
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
    MILVUS_CONSISTENCY_LEVEL: str = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Session")
    MILVUS_DENSE_VECTOR_FIELD: str = os.getenv(
        "MILVUS_DENSE_VECTOR_FIELD", "dense_vector"
    )
    MILVUS_SPARSE_VECTOR_FIELD: str = os.getenv(
        "MILVUS_SPARSE_VECTOR_FIELD", "sparse_vector"
    )
    MILVUS_DENSE_INDEX_TYPE: str = os.getenv("MILVUS_DENSE_INDEX_TYPE", "HNSW")
    MILVUS_DENSE_METRIC_TYPE: str = os.getenv("MILVUS_DENSE_METRIC_TYPE", "IP")
    MILVUS_DENSE_INDEX_M: int = int(os.getenv("MILVUS_DENSE_INDEX_M", "8"))
    MILVUS_DENSE_EF_CONSTRUCTION: int = int(
        os.getenv("MILVUS_DENSE_EF_CONSTRUCTION", "64")
    )
    MILVUS_DENSE_SEARCH_EF: int = int(os.getenv("MILVUS_DENSE_SEARCH_EF", "64"))
    MILVUS_DENSE_INDEX_NLIST: int = int(
        os.getenv("MILVUS_DENSE_INDEX_NLIST", "1024")
    )
    MILVUS_DENSE_SEARCH_NPROBE: int = int(
        os.getenv("MILVUS_DENSE_SEARCH_NPROBE", "32")
    )
    MILVUS_SPARSE_INDEX_TYPE: str = os.getenv(
        "MILVUS_SPARSE_INDEX_TYPE", "SPARSE_INVERTED_INDEX"
    )
    MILVUS_SPARSE_METRIC_TYPE: str = os.getenv("MILVUS_SPARSE_METRIC_TYPE", "IP")
    MILVUS_SPARSE_DROP_RATIO_BUILD: float = float(
        os.getenv("MILVUS_SPARSE_DROP_RATIO_BUILD", "0.2")
    )
    MILVUS_SPARSE_DROP_RATIO_SEARCH: float = float(
        os.getenv("MILVUS_SPARSE_DROP_RATIO_SEARCH", "0.0")
    )

settings = Settings()
