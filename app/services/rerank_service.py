from typing import Any

from langchain_core.documents import Document

from app.core.settings import settings
from app.deps.container import get_reranker, get_vectorstore


def _normalize_search_param(search_param: dict[str, Any], top_k: int) -> dict[str, Any]:
    normalized = dict(search_param)
    params = dict(normalized.get("params") or {})
    if "ef" in params:
        params["ef"] = max(int(params["ef"]), top_k)
    if "nprobe" in params:
        params["nprobe"] = max(int(params["nprobe"]), min(top_k, 64))
    normalized["params"] = params
    return normalized


def _build_search_param(vectorstore: Any, top_k: int) -> dict | list[dict] | None:
    search_param = getattr(vectorstore, "search_params", None)
    if isinstance(search_param, list):
        return [
            _normalize_search_param(param, top_k)
            if isinstance(param, dict)
            else param
            for param in search_param
        ]

    if not isinstance(search_param, dict):
        return None

    return _normalize_search_param(search_param, top_k)


def _build_ranker_kwargs(vectorstore: Any) -> dict[str, Any]:
    if not getattr(vectorstore, "_is_multi_vector", False):
        return {}

    ranker_type = settings.RAG_HYBRID_RANKER.strip().lower()
    if ranker_type == "weighted":
        return {
            "ranker_type": "weighted",
            "ranker_params": {
                "weights": [
                    settings.RAG_HYBRID_DENSE_WEIGHT,
                    settings.RAG_HYBRID_SPARSE_WEIGHT,
                ]
            },
        }

    if ranker_type == "rrf":
        return {"ranker_type": "rrf"}

    return {}


def rerank_documents(
    question: str,
    docs: list[Document],
    top_n: int = settings.RAG_RERANK_TOP_N,
) -> list[Document]:
    if not docs or top_n <= 0:
        return []

    ranked_results = get_reranker().rank(
        query=question,
        documents=[doc.page_content for doc in docs],
        top_k=min(top_n, len(docs)),
        return_documents=False,
    )

    reranked_docs = []
    for ranked_result in ranked_results:
        doc = docs[ranked_result["corpus_id"]]
        doc.metadata = {
            **(doc.metadata or {}),
            "rerank_score": float(ranked_result["score"]),
        }
        reranked_docs.append(doc)

    return reranked_docs


def retrieve_documents(
    rewrite_query: str,
    dense_top_k: int = settings.RAG_DENSE_TOP_K,
    top_n: int = settings.RAG_RERANK_TOP_N,
) -> list[Document]:
    if dense_top_k <= 0 or top_n <= 0:
        return []

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(
        rewrite_query,
        k=dense_top_k,
        param=_build_search_param(vectorstore, dense_top_k),
        **_build_ranker_kwargs(vectorstore),
    )
    return rerank_documents(rewrite_query, docs, top_n=top_n)
