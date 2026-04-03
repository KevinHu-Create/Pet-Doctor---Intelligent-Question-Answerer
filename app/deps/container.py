import os
from functools import lru_cache

from app.core.device import get_torch_device
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

from app.deps.milvus_hybrid import (
    get_dense_embeddings,
    get_vectorstore_embedding_function,
    get_vectorstore_kwargs,
)

@lru_cache
def get_embeddings():
    return get_dense_embeddings()

@lru_cache
def get_vectorstore():
    return Milvus(
        embedding_function=get_vectorstore_embedding_function(),
        connection_args={"uri": settings.MILVUS_URI},
        collection_name=settings.COLLECTION_NAME,
        **get_vectorstore_kwargs(),
    )

@lru_cache
def get_reranker():
    return CrossEncoder(
        settings.RAG_RERANK_MODEL,
        device=get_torch_device(),
        cache_dir=str(settings.HF_CACHE_DIR / "sentence_transformers"),
    )

@lru_cache
def get_llm():
    return ChatOllama(model=settings.OLLAMA_CHAT_MODEL, base_url=settings.OLLAMA_BASE_URL)
