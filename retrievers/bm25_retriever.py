import json
import os
from rank_bm25 import BM25Okapi

def load_documents_from_json(data_path: str) -> list[str]:
    """
    Load documents from a JSON file.
    
    Args:
        data_path: Path to the JSON file containing documents
        
    Returns:
        List of document strings
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    with open(data_path, 'r') as f:
        documents = json.load(f)
    
    return documents

def preprocess_documents(documents: list[str]) -> list[str]:
    """
    Preprocess documents by tokenizing them.
    
    Args:
        documents: List of document strings
    
    Returns:
        List of preprocessed document strings
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    return tokenized_docs

class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.tokenized_docs = preprocess_documents(documents)
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_raw_scores(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = list(zip(scores, self.documents))
        top_results = sorted(scored_docs, key=lambda x: x[0], reverse=True)[:top_k]
        return top_results

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        raw_results = self.get_raw_scores(query, top_k)
        if not raw_results:
            return []
        max_score = max(score for score, _ in raw_results)
        if max_score == 0:
            return raw_results
        normalized_results = [(score / max_score, doc) for score, doc in raw_results]
        return normalized_results

    def show_fancy_results(self, query: str, top_k: int = 3) -> None:
        results = self.retrieve(query, top_k)
        for score, doc in results:
            print(f"Score: {score:.2f}\nDoc: {doc}\n")


loaded_documents = load_documents_from_json("data/sample_docs.json")
retriever = BM25Retriever(loaded_documents)

query = "how does retrieval augmented generation work?"
retriever.show_fancy_results(query)
