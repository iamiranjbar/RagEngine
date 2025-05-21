
from utils.loader import DocumentLoader
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_feedback_retriever import HybridRetrieverWithFeedback

ALPHA = 0.5

loaded_documents = DocumentLoader.load_documents_from_json("data/sample_docs.json")
bm25_retriever = BM25Retriever(loaded_documents)
dense_retriever = DenseRetriever()
dense_retriever.build_index(loaded_documents)
hybrid_retriever_with_feedback = HybridRetrieverWithFeedback(bm25_retriever, dense_retriever, ALPHA)


hybrid_retriever_with_feedback.log_feedback()
hybrid_retriever_with_feedback.save_feedback()

while True:
    query = input("🔍 Your question: ")
    hybrid_retriever_with_feedback.show_result_get_feedback(query)
    hybrid_retriever_with_feedback.save_feedback()
