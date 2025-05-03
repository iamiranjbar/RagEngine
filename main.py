
from utils.loader import DocumentLoader
from retrievers.bm25_retriever import BM25Retriever

loaded_documents = DocumentLoader.load_documents_from_json("data/sample_docs.json")
retriever = BM25Retriever(loaded_documents)
query = "how does retrieval augmented generation work?"
retriever.show_fancy_results(query)
