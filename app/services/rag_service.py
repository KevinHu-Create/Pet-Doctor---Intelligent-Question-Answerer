from functools import lru_cache
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
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

@lru_cache
def get_rag_chain():
    context_pipeline = RunnableLambda(retrieve_documents) | RunnableLambda(format_docs)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": context_pipeline, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain

def answer_question(question: str) -> str:
    chain = get_rag_chain()
    return chain.invoke(question)
