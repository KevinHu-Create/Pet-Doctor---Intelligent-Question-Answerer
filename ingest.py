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

# Per-file chapter ranges. PyPDFLoader already provides page metadata, so we
# only need a page -> chapter lookup during ingestion.
CHAPTER_MAPS = {
    "Dog-Owners-Home-Veterinary-Handbook.pdf": [
        {"chapter": "Chapter 1: Emergencies", "startpage": 1, "endpage": 47},
         {"chapter": "Chapter 2: Gastrointestinal Parasites", "startpage": 51, "endpage": 63},
         {"chapter": "Chapter 3: Infectious Diseases", "startpage": 65, "endpage": 98},
         {"chapter": "Chapter 4: The Skin and Coat", "startpage": 101, "endpage": 168},
         {"chapter": "Chapter 5: The Eyes", "startpage": 169, "endpage": 204},
         {"chapter": "Chapter 6: The Ears", "startpage": 205, "endpage": 218},
         {"chapter": "Chapter 7: The Nose", "startpage": 221, "endpage": 229},
         {"chapter": "Chapter 8: The Mouse and Throat", "startpage": 231, "endpage": 254},
         {"chapter": "Chapter 9: The Digestive System", "startpage": 255, "endpage": 308},
         {"chapter": "Chapter 10: The Respiratory System", "startpage": 311, "endpage": 326},
         {"chapter": "Chapter 11: The Circulatory System", "startpage": 329, "endpage": 352},
         {"chapter": "Chapter 12: The Nervous System", "startpage": 355, "endpage": 381},
         {"chapter": "Chapter 13: The Musculoskeletal System", "startpage": 383, "endpage": 408},
         {"chapter": "Chapter 14: The Urinary System", "startpage": 411, "endpage": 426},
         {"chapter": "Chapter 15: Sex and Reproduction", "startpage": 427, "endpage": 466},
         {"chapter": "Chapter 16: Pregnancy and Whelping", "startpage": 467, "endpage": 487},
         {"chapter": "Chapter 17: Pediatrics", "startpage": 489, "endpage": 520},
         {"chapter": "Chapter 18: Tumors and Cancers", "startpage": 525, "endpage": 544},
         {"chapter": "Chapter 19: Geriatrics", "startpage": 545, "endpage": 558},
         {"chapter": "Chapter 20: Medications", "startpage": 559, "endpage": 570},
     ]
}

# Per-file section ranges. This is prepared now so section metadata can be
# added later without changing the ingestion structure again.
#
# Annotation:
# We are not writing `section` into document metadata yet because the current
# evaluation plan only uses `chapter`. Keep this map in sync with CHAPTER_MAPS
# so it is ready when section-level evaluation or filtering is needed.
#SECTION_MAPS = {
#    "Dog-Owners-Home-Veterinary-Handbook.pdf": [
#        # {"section": "Section Name", "startpage": 1, "endpage": 10},
#    ]
#}


def get_label_for_page(page_number: int, page_map: list[dict], label_key: str) -> str | None:
    for entry in page_map:
        if entry["startpage"] <= page_number <= entry["endpage"]:
            return entry[label_key]
    return None

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

        chapter_map = CHAPTER_MAPS.get(path.name, [])
        section_map = SECTION_MAPS.get(path.name, [])

        # 给每个 Document 打上“可追溯”的 metadata（后面做 Recall@K 要用）
        for d in loaded:
            d.metadata["source_path"] = str(path)
            d.metadata["source_folder"] = str(path.parent.relative_to(base))  # 保留子目录层级
            d.metadata["file_type"] = suffix
            if suffix == ".pdf":
                page_number = d.metadata.get("page", -1)
                chapter = get_label_for_page(page_number, chapter_map, "chapter")
                if chapter:
                    d.metadata["chapter"] = chapter

                # Annotation:
                # Section metadata is intentionally not written yet. The lookup
                # is prepared here for future use once evaluation starts using
                # section-level metadata.
                # section = get_label_for_page(page_number, section_map, "section")
                # if section:
                #     d.metadata["section"] = section

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
