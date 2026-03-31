from langchain_core.documents import Document

from app.core.settings import settings
from app.deps.container import get_reranker, get_vectorstore


def _build_search_param(dense_top_k: int) -> dict | None:
    vectorstore = get_vectorstore()
    search_param = getattr(vectorstore, "search_params", None)
    if not isinstance(search_param, dict):
        return None

    normalized = dict(search_param)
    params = dict(normalized.get("params") or {})
    if "ef" in params:
        params["ef"] = max(int(params["ef"]), dense_top_k)
    normalized["params"] = params
    return normalized


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

    docs = get_vectorstore().similarity_search(
        rewrite_query,
        k=dense_top_k,
        param=_build_search_param(dense_top_k),
    )
    return rerank_documents(rewrite_query, docs, top_n=top_n)
