from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.deps.container import get_llm
from app.services.rerank_service import retrieve_documents

PROMPT_TEMPLATE = """
Human: You are a pet health assistant.
Answer the question using only the provided context.

Rules:
- Keep the answer under 80 words.
- Do not include statistics or numbers unless they are explicitly stated in the context.
- Do not add facts or symptoms not supported by the context.
- Prefer the most directly relevant guidance from the context instead of combining multiple speculative causes.
- If the context contains emergency warning signs that match the question, say that immediate veterinary attention is needed.
- Do not invent differential diagnoses unless they are explicitly and directly supported by the most relevant context.
- Do not list multiple possible diseases or causes unless the user explicitly asks for causes.
- Prefer concrete next-step guidance over speculative explanation.
- When in doubt, choose the safest context-supported recommendation instead of summarizing every retrieved possibility.
- If the context is insufficient, say so briefly.
- Do not mention the handbook or the source of the information in your answer, do not say see page.

<context>
{context}
</context>

<question>
{question}
</question>

Assistant:
"""

def format_docs(docs):
    print("\n=== RETRIEVED DOCS ===")
    for i, doc in enumerate(docs):
        print(f"\n--- doc {i} ---")
        print(doc.page_content[:300])  # 防止太长
    print("=== END DOCS ===\n")

    return "\n\n".join(doc.page_content for doc in docs)


def _normalize_chain_input(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return {"question": payload.strip(), "rewrite_query": payload.strip()}

    if isinstance(payload, dict):
        question = str(payload.get("question", "")).strip()
        rewrite_query = str(payload.get("rewrite_query", question)).strip()
        return {
            "question": question,
            "rewrite_query": rewrite_query or question,
        }

    raise TypeError(f"Unsupported RAG chain input type: {type(payload)!r}")


def _retrieve_docs_for_chain(payload: Any):
    normalized = _normalize_chain_input(payload)
    return retrieve_documents(normalized["rewrite_query"])


def _question_for_chain(payload: Any) -> str:
    return _normalize_chain_input(payload)["question"]


@lru_cache
def get_rag_chain():
    context_pipeline = RunnableLambda(_retrieve_docs_for_chain) | RunnableLambda(format_docs)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": context_pipeline, "question": RunnableLambda(_question_for_chain)}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain

def answer_question(
    question: str,
    rewrite_query: str | None = None,
) -> str:
    chain = get_rag_chain()
    return chain.invoke(
        {
            "question": str(question or "").strip(),
            "rewrite_query": str(rewrite_query or question or "").strip(),
        }
    )
