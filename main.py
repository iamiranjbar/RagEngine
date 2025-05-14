
from utils.loader import DocumentLoader
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever

loaded_documents = DocumentLoader.load_documents_from_json("data/sample_docs.json")
bm25_retriever = BM25Retriever(loaded_documents)
dense_retriever = DenseRetriever()
dense_retriever.build_index(loaded_documents)

query = "how does retrieval augmented generation work?"
alpha = 0.5

hybrid_retriever = HybridRetriever(bm25_retriever, dense_retriever, alpha)
hybrid_retriever.show_fancy_results(query)

