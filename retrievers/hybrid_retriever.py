class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever, alpha=0.5):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.alpha = alpha  # weight for dense (1 - alpha) for BM25

    def retrieve(self, query, top_k=3):
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        
        # Convert to 1-score format to align with BM25 (higher is better)
        dense_results = [(1 - score, doc) for score, doc in dense_results]

        # Build score maps
        scores = {}
        for score, doc in bm25_results:
            scores[doc] = scores.get(doc, 0) + (1 - self.alpha) * score
        for score, doc in dense_results:
            scores[doc] = scores.get(doc, 0) + self.alpha * score

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def show_fancy_results(self, query: str, top_k: int = 3) -> None:
        results = self.retrieve(query, top_k)
        for doc, score in results:
            print(f"Score: {score:.2f}\nDoc: {doc}\n")
