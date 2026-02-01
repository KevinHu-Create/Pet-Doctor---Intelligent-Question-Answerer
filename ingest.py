from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    WIKI_URLS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HF_EMBED_MODEL,
    MILVUS_URI,
    COLLECTION_NAME,
)

def load_and_split(urls):
    loader = WebBaseLoader(web_paths=tuple(urls))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(documents)
    return docs

def ingest(drop_old: bool = False):
    docs = load_and_split(WIKI_URLS)
    print(f"[ingest] loaded & split into {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=HF_EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 写入 Milvus（docker service）
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={"uri": MILVUS_URI},
        collection_name=COLLECTION_NAME,
        drop_old=drop_old,
    )

    print(f"[ingest] saved to Milvus collection='{COLLECTION_NAME}', uri='{MILVUS_URI}'")
    return vectorstore

if __name__ == "__main__":
    ingest(drop_old=True)