import os
from functools import lru_cache

from app.core.settings import settings

os.environ.setdefault("HF_HOME", str(settings.HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(settings.HF_CACHE_DIR / "hub"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    str(settings.HF_CACHE_DIR / "sentence_transformers"),
)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(settings.HF_HUB_ETAG_TIMEOUT))
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    str(settings.HF_HUB_DOWNLOAD_TIMEOUT),
)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama

@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.HF_EMBED_MODEL,
        cache_folder=str(settings.HF_CACHE_DIR / "sentence_transformers"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache
def get_vectorstore():
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": settings.MILVUS_URI},
        collection_name=settings.COLLECTION_NAME,
    )

@lru_cache
def get_llm():
    return ChatOllama(model=settings.OLLAMA_CHAT_MODEL, base_url=settings.OLLAMA_BASE_URL)
