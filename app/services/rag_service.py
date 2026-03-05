from functools import lru_cache
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.deps.container import get_vectorstore, get_llm

PROMPT_TEMPLATE = """
Human: You are a Pet doctor AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
Don't say you don't know the answer unless there is completely no any relevant information in the context, if you need more information just ask the user for more reletive information.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@lru_cache
def get_rag_chain():
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain

def answer_question(question: str) -> str:
    chain = get_rag_chain()
    return chain.invoke(question)