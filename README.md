# RagEngine

**RagEngine** is a modular, extensible framework for building Retrieval-Augmented Generation (RAG) pipelines. It combines sparse (BM25) and dense (vector) retrieval techniques to fetch relevant documents and generate grounded answers using large language models (LLMs).

This project is designed to:
- Serve as a learning resource for hybrid retrieval and RAG
- Act as a practical base for local or remote LLM deployments
- Support plug-and-play retrievers, scoring logic, and generation backends

### 🔧 Features
- ✅ BM25-based sparse retrieval using `rank_bm25`
- ✅ Dense vector search using SentenceTransformers + Faiss
- ✅ Hybrid scoring combiner (weighted, rank fusion, etc.)
- ✅ LLM-ready response pipeline (for future integration)
- ✅ Modular design for swapping components
