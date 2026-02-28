# deps.py
from functools import lru_cache

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

@lru_cache
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

@lru_cache
def get_vectorstore():
    # 连接已有 collection（不写入）
    return Milvus(
        embedding_function=get_embeddings(),
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
    )

@lru_cache
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": 4})

@lru_cache
def get_llm():
    return ChatOllama(model=OLLAMA_CHAT_MODEL, base_url=OLLAMA_BASE_URL)

@lru_cache
def get_rag_chain():
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain = (
        {"context": get_retriever() | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain