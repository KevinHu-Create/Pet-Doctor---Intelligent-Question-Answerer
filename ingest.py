import os
from pathlib import Path

from app.core.settings import settings

os.environ.setdefault("HF_HOME", str(settings.HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(settings.HF_CACHE_DIR / "hub"))
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    str(settings.HF_CACHE_DIR / "sentence_transformers"),
)
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(settings.HF_HUB_ETAG_TIMEOUT))
os.environ.setdefault(
    "HF_HUB_DOWNLOAD_TIMEOUT",
    str(settings.HF_HUB_DOWNLOAD_TIMEOUT),
)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}

# Per-file chapter ranges using the handbook's printed page numbers.
CHAPTER_MAPS = {
    "Dog-Owners-Home-Veterinary-Handbook.pdf": [
        {"chapter": "Chapter 1: Emergencies", "startpage": 1, "endpage": 50},
         {"chapter": "Chapter 2: Gastrointestinal Parasites", "startpage": 51, "endpage": 64},
         {"chapter": "Chapter 3: Infectious Diseases", "startpage": 65, "endpage": 100},
         {"chapter": "Chapter 4: The Skin and Coat", "startpage": 101, "endpage": 168},
         {"chapter": "Chapter 5: The Eyes", "startpage": 169, "endpage": 204},
         {"chapter": "Chapter 6: The Ears", "startpage": 205, "endpage": 220},
         {"chapter": "Chapter 7: The Nose", "startpage": 221, "endpage": 230},
         {"chapter": "Chapter 8: The Mouth and Throat", "startpage": 231, "endpage": 254},
         {"chapter": "Chapter 9: The Digestive System", "startpage": 255, "endpage": 310},
         {"chapter": "Chapter 10: The Respiratory System", "startpage": 311, "endpage": 328},
         {"chapter": "Chapter 11: The Circulatory System", "startpage": 329, "endpage": 354},
         {"chapter": "Chapter 12: The Nervous System", "startpage": 355, "endpage": 382},
         {"chapter": "Chapter 13: The Musculoskeletal System", "startpage": 383, "endpage": 410},
         {"chapter": "Chapter 14: The Urinary System", "startpage": 411, "endpage": 426},
         {"chapter": "Chapter 15: Sex and Reproduction", "startpage": 427, "endpage": 466},
         {"chapter": "Chapter 16: Pregnancy and Whelping", "startpage": 467, "endpage": 488},
         {"chapter": "Chapter 17: Pediatrics", "startpage": 489, "endpage": 524},
         {"chapter": "Chapter 18: Tumors and Cancers", "startpage": 525, "endpage": 544},
         {"chapter": "Chapter 19: Geriatrics", "startpage": 545, "endpage": 558},
         {"chapter": "Chapter 20: Medications", "startpage": 559, "endpage": 570},
     ]
}

# Per-file section ranges using the handbook's printed page numbers.
#
# Annotation:
# We are not writing `section` into document metadata yet because the current
# evaluation plan only uses `chapter`. During ingestion we convert the PDF page
# index from PyPDFLoader into a printed page number before doing the lookup.
SECTION_MAPS = {
    "Dog-Owners-Home-Veterinary-Handbook.pdf": [
        {"section": "Handling and Restraint", "startpage": 2, "endpage": 6},
        {"section": "Artificial Respiration and Heart Massage", "startpage": 7, "endpage": 10},
        {"section": "Shock", "startpage": 11, "endpage": 13},
        {"section": "Acute Painful Abdomen", "startpage": 14, "endpage": 14},
        {"section": "Broken Bones", "startpage": 15, "endpage": 15},
        {"section": "Burns", "startpage": 16, "endpage": 16},
        {"section": "Cold Exposure", "startpage": 17, "endpage": 18},
        {"section": "Dehydration", "startpage": 19, "endpage": 19},
        {"section": "Drowning and Suffocation", "startpage": 20, "endpage": 20},
        {"section": "Electric Shock", "startpage": 21, "endpage": 21},
        {"section": "Heat Stroke", "startpage": 22, "endpage": 22},
        {"section": "Poisoning", "startpage": 23, "endpage": 38},
        {"section": "Insect Stings and Bites", "startpage": 39, "endpage": 39},
        {"section": "Snakebites", "startpage": 40, "endpage": 41},
        {"section": "Wounds", "startpage": 42, "endpage": 50},
        {"section": "Deworming Puppies and Adult Dogs", "startpage": 51, "endpage": 52},
        {"section": "How to Control Worms", "startpage": 53, "endpage": 55},
        {"section": "Ascarids (Roundworms)", "startpage": 56, "endpage": 57},
        {"section": "Hookworms (Ancylostoma)", "startpage": 58, "endpage": 59},
        {"section": "Tapeworms", "startpage": 60, "endpage": 60},
        {"section": "Whipworms", "startpage": 61, "endpage": 61},
        {"section": "Threadworms (Strongyloides)", "startpage": 62, "endpage": 62},
        {"section": "Other Worm Parasites", "startpage": 62, "endpage": 64},
        {"section": "Bacterial Diseases", "startpage": 65, "endpage": 70},
        {"section": "Viral Diseases", "startpage": 71, "endpage": 79},
        {"section": "Fungal Diseases", "startpage": 80, "endpage": 82},
        {"section": "Protozoan Diseases", "startpage": 83, "endpage": 86},
        {"section": "Rickettsial Diseases", "startpage": 87, "endpage": 89},
        {"section": "Antibodies and Immunity", "startpage": 90, "endpage": 90},
        {"section": "Vaccinations", "startpage": 91, "endpage": 93},
        {"section": "Available Vaccines", "startpage": 94, "endpage": 97},
        {"section": "Vaccination Recommendations", "startpage": 98, "endpage": 101},
        {"section": "The Dog’s Coat", "startpage": 102, "endpage": 102},
        {"section": "Avoiding Coat and Skin Problems", "startpage": 103, "endpage": 106},
        {"section": "Beyond the Brush", "startpage": 107, "endpage": 108},
        {"section": "Bathing", "startpage": 109, "endpage": 111},
        {"section": "Sorting Out Skin Diseases", "startpage": 112, "endpage": 116},
        {"section": "Fleas", "startpage": 117, "endpage": 122},
        {"section": "Ticks", "startpage": 123, "endpage": 125},
        {"section": "Other External Parasites", "startpage": 126, "endpage": 130},
        {"section": "Using Insecticides", "startpage": 131, "endpage": 133},
        {"section": "Lick Granuloma (Acral Pruritic Dermatitis)", "startpage": 134, "endpage": 134},
        {"section": "Allergies", "startpage": 135, "endpage": 141},
        {"section": "Disorders with Hair Loss", "startpage": 142, "endpage": 156},
        {"section": "Pyoderma (Skin Infections)", "startpage": 157, "endpage": 163},
        {"section": "Autoimmune and Immune-Mediated Skin Diseases", "startpage": 164, "endpage": 170},
        {"section": "How Dogs See", "startpage": 171, "endpage": 171},
        {"section": "What to Do if Your Dog Has an Eye Problem", "startpage": 172, "endpage": 174},
        {"section": "The Eyeball", "startpage": 175, "endpage": 176},
        {"section": "Eyelids", "startpage": 177, "endpage": 181},
        {"section": "The Nictitating Membrane (Third Eyelid)", "startpage": 182, "endpage": 183},
        {"section": "The Outer Eye", "startpage": 184, "endpage": 185},
        {"section": "The Tearing Mechanism", "startpage": 186, "endpage": 190},
        {"section": "The Cornea", "startpage": 191, "endpage": 195},
        {"section": "The Inner Eye", "startpage": 196, "endpage": 200},
        {"section": "Retinal Diseases", "startpage": 201, "endpage": 205},
        {"section": "Basic Ear Care", "startpage": 206, "endpage": 208},
        {"section": "The Ear Flap", "startpage": 209, "endpage": 211},
        {"section": "The Ear Canal", "startpage": 212, "endpage": 215},
        {"section": "The Middle Ear", "startpage": 216, "endpage": 216},
        {"section": "The Inner Ear", "startpage": 217, "endpage": 217},
        {"section": "Deafness", "startpage": 218, "endpage": 221},
        {"section": "Signs of Nasal Irritation", "startpage": 222, "endpage": 222},
        {"section": "The Nose", "startpage": 223, "endpage": 225},
        {"section": "The Nasal Cavity", "startpage": 226, "endpage": 230},
        {"section": "How to Examine the Mouth", "startpage": 231, "endpage": 231},
        {"section": "Signs of Mouth and Throat Disease", "startpage": 232, "endpage": 232},
        {"section": "The Mouth", "startpage": 233, "endpage": 238},
        {"section": "Teeth and Gums", "startpage": 239, "endpage": 248},
        {"section": "Taking Care of Your Dog’s Teeth", "startpage": 249, "endpage": 250},
        {"section": "The Throat", "startpage": 251, "endpage": 251},
        {"section": "Salivary Glands", "startpage": 252, "endpage": 252},
        {"section": "The Head", "startpage": 253, "endpage": 255},
        {"section": "Endoscopy", "startpage": 256, "endpage": 256},
        {"section": "The Esophagus", "startpage": 257, "endpage": 260},
        {"section": "The Stomach", "startpage": 261, "endpage": 269},
        {"section": "Vomiting", "startpage": 270, "endpage": 272},
        {"section": "Small and Large Bowels", "startpage": 273, "endpage": 282},
        {"section": "Diarrhea", "startpage": 283, "endpage": 285},
        {"section": "The Anus and Rectum", "startpage": 286, "endpage": 293},
        {"section": "The Liver", "startpage": 294, "endpage": 296},
        {"section": "The Pancreas", "startpage": 297, "endpage": 301},
        {"section": "Feeding and Nutrition", "startpage": 302, "endpage": 310},
        {"section": "Abnormal Breathing", "startpage": 311, "endpage": 313},
        {"section": "The Larynx (Voice Box)", "startpage": 314, "endpage": 318},
        {"section": "Coughing", "startpage": 319, "endpage": 319},
        {"section": "Trachea and Bronchi", "startpage": 320, "endpage": 323},
        {"section": "The Lungs", "startpage": 324, "endpage": 325},
        {"section": "Tumors of the Larynx, Trachea, and Lungs", "startpage": 326, "endpage": 329},
        {"section": "The Normal Heart", "startpage": 330, "endpage": 332},
        {"section": "Heart Disease", "startpage": 333, "endpage": 337},
        {"section": "Arrhythmias", "startpage": 338, "endpage": 338},
        {"section": "Congestive Heart Failure", "startpage": 339, "endpage": 340},
        {"section": "Heartworms", "startpage": 341, "endpage": 348},
        {"section": "Anemia and Clotting Disorders", "startpage": 349, "endpage": 355},
        {"section": "Neurological Evaluation", "startpage": 356, "endpage": 356},
        {"section": "Head Injuries", "startpage": 357, "endpage": 359},
        {"section": "Brain Diseases", "startpage": 360, "endpage": 362},
        {"section": "Hereditary Diseases", "startpage": 363, "endpage": 366},
        {"section": "Seizure Disorders", "startpage": 367, "endpage": 370},
        {"section": "Coma", "startpage": 371, "endpage": 371},
        {"section": "Weakness or Paralysis", "startpage": 372, "endpage": 373},
        {"section": "Spinal Cord Diseases", "startpage": 374, "endpage": 379},
        {"section": "Peripheral Nerve Injuries", "startpage": 380, "endpage": 383},
        {"section": "Limping or Lameness", "startpage": 384, "endpage": 385},
        {"section": "Bone and Joint Injuries", "startpage": 386, "endpage": 391},
        {"section": "Inherited Orthopedic Diseases", "startpage": 392, "endpage": 400},
        {"section": "Arthritis", "startpage": 401, "endpage": 406},
        {"section": "Metabolic Bone Diseases", "startpage": 407, "endpage": 407},
        {"section": "Vitamin and Mineral Supplements", "startpage": 408, "endpage": 411},
        {"section": "Signs of Urinary Tract Disease", "startpage": 412, "endpage": 412},
        {"section": "Diagnosing Urinary Tract Diseases", "startpage": 413, "endpage": 413},
        {"section": "Diseases of the Bladder and Urethra", "startpage": 414, "endpage": 417},
        {"section": "Enlarged Prostate", "startpage": 418, "endpage": 419},
        {"section": "Kidney Disease", "startpage": 420, "endpage": 427},
        {"section": "The Basics of Genetics", "startpage": 428, "endpage": 429},
        {"section": "Breeding Purebred Dogs", "startpage": 430, "endpage": 440},
        {"section": "Mating Procedures", "startpage": 441, "endpage": 443},
        {"section": "Artificial Insemination", "startpage": 444, "endpage": 445},
        {"section": "Male Infertility", "startpage": 446, "endpage": 447},
        {"section": "Female Infertility", "startpage": 448, "endpage": 453},
        {"section": "Pseudocyesis (False Pregnancy)", "startpage": 454, "endpage": 454},
        {"section": "Unwanted Pregnancy", "startpage": 455, "endpage": 455},
        {"section": "Preventing Pregnancy", "startpage": 456, "endpage": 458},
        {"section": "Diseases of the Female Genital Tract", "startpage": 459, "endpage": 462},
        {"section": "Diseases of the Male Genital Tract", "startpage": 463, "endpage": 466},
        {"section": "Pregnancy", "startpage": 467, "endpage": 471},
        {"section": "Whelping", "startpage": 472, "endpage": 476},
        {"section": "Dystocia (Prolonged or Difficult Labor)", "startpage": 477, "endpage": 480},
        {"section": "Postpartum Care of the Dam", "startpage": 481, "endpage": 481},
        {"section": "Postpartum Problems", "startpage": 482, "endpage": 486},
        {"section": "Maternal Neglect", "startpage": 487, "endpage": 489},
        {"section": "Caring for Newborns", "startpage": 490, "endpage": 495},
        {"section": "Raising Puppies by Hand", "startpage": 496, "endpage": 502},
        {"section": "Diseases of Newborn Puppies", "startpage": 503, "endpage": 508},
        {"section": "Weaning", "startpage": 509, "endpage": 510},
        {"section": "Feeding Growing Puppies", "startpage": 511, "endpage": 511},
        {"section": "The Importance of Early Socialization", "startpage": 512, "endpage": 512},
        {"section": "Acquiring a New Puppy", "startpage": 513, "endpage": 516},
        {"section": "Training Advice for Your Puppy", "startpage": 517, "endpage": 525},
        {"section": "What Causes Cancer?", "startpage": 526, "endpage": 526},
        {"section": "Treating Tumors and Cancers", "startpage": 527, "endpage": 529},
        {"section": "Common Surface Tumors", "startpage": 530, "endpage": 535},
        {"section": "Soft Tissue Sarcomas", "startpage": 536, "endpage": 537},
        {"section": "Transitional Cell Carcinoma of the Bladder", "startpage": 538, "endpage": 538},
        {"section": "Bone Tumors", "startpage": 538, "endpage": 539},
        {"section": "Reproductive Tract Tumors", "startpage": 540, "endpage": 543},
        {"section": "Leukemia", "startpage": 544, "endpage": 545},
        {"section": "The Geriatric Checkup", "startpage": 546, "endpage": 546},
        {"section": "Behavior Changes", "startpage": 546, "endpage": 547},
        {"section": "Physical Changes", "startpage": 548, "endpage": 551},
        {"section": "Functional Changes", "startpage": 552, "endpage": 553},
        {"section": "Weight Changes", "startpage": 554, "endpage": 554},
        {"section": "Temperature, Pulse, and Respiration", "startpage": 554, "endpage": 554},
        {"section": "Diet and Nutrition", "startpage": 554, "endpage": 555},
        {"section": "Hospice Care", "startpage": 556, "endpage": 556},
        {"section": "Euthanasia", "startpage": 557, "endpage": 558},
        {"section": "Anesthetics and Anesthesia", "startpage": 559, "endpage": 559},
        {"section": "Analgesics", "startpage": 560, "endpage": 561},
        {"section": "Antibiotics", "startpage": 562, "endpage": 563},
        {"section": "Drug Complications", "startpage": 564, "endpage": 565},
        {"section": "How to Give Medications", "startpage": 566, "endpage": 572},
        {"section": "Normal Body Temperature", "startpage": 573, "endpage": 573},
        {"section": "Normal Heart Rate", "startpage": 574, "endpage": 574},
        {"section": "Normal Respiratory Rate", "startpage": 574, "endpage": 574},
        {"section": "Gestation", "startpage": 574, "endpage": 576},
        {"section": "Complete Blood Count (CBC) or Hemogram", "startpage": 577, "endpage": 577},
        {"section": "Blood Chemistry Panel", "startpage": 578, "endpage": 579},
        {"section": "Urinalysis", "startpage": 580, "endpage": 582},
        {"section": "Web Sites for General Health Information", "startpage": 583, "endpage": 583},
        {"section": "Web Sites for Dog Food and Nutrition Information", "startpage": 583, "endpage": 627},
    ]
}

PRINTED_PAGE_OFFSETS = {
    "Dog-Owners-Home-Veterinary-Handbook.pdf": 29,
}


def get_label_for_page(page_number: int, page_map: list[dict], label_key: str) -> str | None:
    for entry in page_map:
        if entry["startpage"] <= page_number <= entry["endpage"]:
            return entry[label_key]
    return None


def get_printed_page(path_name: str, pdf_page_number: int) -> int | None:
    offset = PRINTED_PAGE_OFFSETS.get(path_name)
    if offset is None:
        return None
    printed_page = pdf_page_number - offset
    if printed_page < 1 or printed_page > 627:
        return None
    return printed_page

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
                pdf_page_number = d.metadata.get("page", -1)
                printed_page = get_printed_page(path.name, pdf_page_number)
                if printed_page is not None:
                    d.metadata["printed_page"] = printed_page
                    chapter = get_label_for_page(printed_page, chapter_map, "chapter")
                    d.metadata["chapter"] = chapter if chapter else "Unknown"
                else:
                    d.metadata["chapter"] = "Front Matter"
                

                # Annotation:
                # Section metadata is intentionally not written yet. The lookup
                # is prepared here for future use once evaluation starts using
                # section-level metadata.
                # section = get_label_for_page(lookup_page, section_map, "section")
                # if section:
                #     d.metadata["section"] = section

        docs.extend(loaded)

    print(f"[ingest] loaded {len(docs)} raw document pages/sections")
    return docs
    
def load_and_split(paths):
    documents = load_local_documents(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    docs = splitter.split_documents(documents)
    return docs

def ingest(drop_old: bool = False):
    docs = load_and_split(settings.DATA_DIR)
    print(f"[ingest] loaded & split into {len(docs)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.HF_EMBED_MODEL,
        cache_folder=str(settings.HF_CACHE_DIR / "sentence_transformers"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 写入 Milvus（docker service）
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={"uri": settings.MILVUS_URI},
        collection_name=settings.COLLECTION_NAME,
        drop_old=drop_old,
    )

    print(
        f"[ingest] saved to Milvus collection='{settings.COLLECTION_NAME}', "
        f"uri='{settings.MILVUS_URI}'"
    )
    return vectorstore

if __name__ == "__main__":
    ingest(drop_old=True)
