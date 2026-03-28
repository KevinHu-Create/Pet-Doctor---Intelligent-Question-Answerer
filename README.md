# Pet-Doctor---Intelligent-Question-Answerer
## Pet Doctor – Intelligent Question Answerer

This project implements a Retrieval-Augmented Generation (RAG) pipeline for
domain-specific pet health question answering.

### Architecture
- **Embedding**: HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Vector Database**: Milvus (Dockerized with MinIO + etcd)
- **LLM**: Local Ollama-hosted `llama3.2`
- **Document Loader**: Local data documents using LangChain PDFLoader
- **Interface**: FastAPI service

### Workflow
1. Local documents are loaded and split into overlapping text chunks.
2. Chunks are embedded using sentence-transformer models.
3. Embeddings are stored in Milvus as a persistent vector collection.
4. User queries retrieve relevant context via vector similarity search.
5. Retrieved context is injected into a prompt for LLM-based answer generation.

### Key Features
- Dockerized vector database services
- Decoupled ingestion and query pipelines
- Configurable embedding and collection settings
- Local Ollama integration for answer generation
- Evaluation scripts for retrieval and RAG quality checks

### Current Status
- ✅ End-to-end RAG pipeline verified
- ✅ Successful ingestion and retrieval from Milvus
- ✅ FastAPI service is the primary application entrypoint
