import logging
from typing import Sequence

from langchain_core.documents import Document

from app.core.settings import settings
from app.deps.container import get_reranker, get_vectorstore
from app.pipeline.query_rewrite import rewrite_query_for_retrieval

logger = logging.getLogger(__name__)


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
    question: str,
    dense_top_k: int = settings.RAG_DENSE_TOP_K,
    top_n: int = settings.RAG_RERANK_TOP_N,
    conversation_context: Sequence[str] | None = None,
) -> list[Document]:
    if dense_top_k <= 0 or top_n <= 0:
        return []

    rewrite_result = rewrite_query_for_retrieval(
        question,
        conversation_context=conversation_context,
    )
    retrieval_query = rewrite_result.retrieval_query or question

    if rewrite_result.rewrite_needed or rewrite_result.rewrite_applied:
        logger.info(
            "Query rewrite rewrite_needed=%s rule_score=%s reasons=%s history_available=%s llm_used=%s rewrite_applied=%s original=%r retrieval=%r",
            rewrite_result.rewrite_needed,
            rewrite_result.rule_score,
            ",".join(rewrite_result.reasons),
            rewrite_result.history_available,
            rewrite_result.llm_used,
            rewrite_result.rewrite_applied,
            rewrite_result.original_query,
            retrieval_query,
        )

    docs = get_vectorstore().similarity_search(retrieval_query, k=dense_top_k)
    return rerank_documents(retrieval_query, docs, top_n=top_n)
