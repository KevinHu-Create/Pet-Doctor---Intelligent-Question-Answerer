# Pet-Doctor---Intelligent-Question-Answerer
## Pet Doctor – Intelligent Question Answerer (CLI Prototype)

This project implements a Retrieval-Augmented Generation (RAG) pipeline for
domain-specific question answering, currently demonstrated with medical
knowledge about insulin.

### Architecture
- **Embedding**: HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: Milvus (Dockerized)
- **LLM**: Ollama-hosted `llama3.2`
- **Document Loader**: Web-based ingestion using LangChain
- **Interface**: Interactive CLI (command-line interface)

### Workflow
1. Web documents are loaded and split into overlapping text chunks.
2. Chunks are embedded using sentence-transformer models.
3. Embeddings are stored in Milvus as a persistent vector collection.
4. User queries retrieve relevant context via vector similarity search.
5. Retrieved context is injected into a prompt for LLM-based answer generation.

### Key Features
- Fully containerized backend services (Milvus, MinIO, etcd, Ollama)
- Decoupled ingestion and query pipelines
- Configurable embedding and collection settings
- CLI-based interaction for fast debugging and verification

### Current Status
- ✅ End-to-end RAG pipeline verified
- ✅ Successful ingestion and retrieval from Milvus
- 🔜 Planned upgrade to API-based service (FastAPI)