import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings:
    DATA_DIR = os.getenv("DATA_DIR", PROJECT_ROOT / "data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

    HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus:19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "wiki_insulin_st_minilm_v1")

settings = Settings()