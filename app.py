# app.py
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import (
    HF_EMBED_MODEL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_BASE_URL,
    MILVUS_URI,
    COLLECTION_NAME,
)

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

def build_rag_chain():
    embeddings = HuggingFaceEmbeddings(
    model_name=HF_EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    )

    # 关键：这里不是 from_documents（不写入），而是“连接已有 collection”
    vectorstore = Milvus(
        embedding_function=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    llm = ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def main():
    rag_chain = build_rag_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()
        if query.lower() in ("exit", "quit"):
            break
        res = rag_chain.invoke(query)
        print("\n--- Answer ---")
        print(res)

if __name__ == "__main__":
    main()