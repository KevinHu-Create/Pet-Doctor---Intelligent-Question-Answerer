from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pathlib import Path
from config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    HF_EMBED_MODEL,
    MILVUS_URI,
    COLLECTION_NAME,
)

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}

def load_local_documents(data_dir: str):
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"DATA_DIR not found: {base.resolve()}")

    docs = []
    files = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]

    for path in files:
        suffix = path.suffix.lower()

        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()
            elif suffix == ".docx":
                loaded = Docx2txtLoader(str(path)).load()
            elif suffix in [".txt", ".md"]:
                loaded = TextLoader(str(path), encoding="utf-8").load()
            else:
                continue
        except Exception as e:
            print(f"[ingest] skip (failed to load): {path} | err={e}")
            continue

        # 给每个 Document 打上“可追溯”的 metadata（后面做 Recall@K 要用）
        for d in loaded:
            d.metadata["source_path"] = str(path)
            d.metadata["source_folder"] = str(path.parent.relative_to(base))  # 保留子目录层级
            d.metadata["file_type"] = suffix

        docs.extend(loaded)

    print(f"[ingest] loaded {len(docs)} raw document pages/sections")
    return docs
    
def load_and_split(paths):
    documents = load_local_documents(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = splitter.split_documents(documents)
    return docs

def ingest(drop_old: bool = False):
    docs = load_and_split(DATA_DIR)
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