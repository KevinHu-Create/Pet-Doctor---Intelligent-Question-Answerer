from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama

from app.core.settings import settings

@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=settings.HF_EMBED_MODEL,
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