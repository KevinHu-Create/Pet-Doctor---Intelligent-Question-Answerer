from functools import lru_cache
from typing import Any, Sequence

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.deps.container import get_llm
from app.services.retrieval_service import retrieve_documents

PROMPT_TEMPLATE = """
Human: You are a pet health assistant.
Answer the question using only the provided context.

Rules:
- Keep the answer under 80 words.
- Do not include statistics or numbers unless they are explicitly stated in the context.
- Do not add facts or symptoms not supported by the context.
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
        return {"question": payload, "conversation_context": []}

    if isinstance(payload, dict):
        question = str(payload.get("question", "")).strip()
        raw_context = payload.get("conversation_context") or []
        if isinstance(raw_context, str):
            raw_context = [raw_context]
        conversation_context = [
            str(item).strip() for item in raw_context if str(item).strip()
        ]
        return {
            "question": question,
            "conversation_context": conversation_context,
        }

    raise TypeError(f"Unsupported RAG chain input type: {type(payload)!r}")


def _retrieve_docs_for_chain(payload: Any):
    normalized = _normalize_chain_input(payload)
    return retrieve_documents(
        normalized["question"],
        conversation_context=normalized["conversation_context"],
    )


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
    conversation_context: Sequence[str] | None = None,
) -> str:
    chain = get_rag_chain()
    return chain.invoke(
        {
            "question": question,
            "conversation_context": list(conversation_context or []),
        }
    )
