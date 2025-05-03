from rank_bm25 import BM25Okapi
from utils.preprocessor import TextPreprocessor

class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        self.text_preprocessor = TextPreprocessor()
        self.tokenized_docs = [self.text_preprocessor.process(doc) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def get_raw_scores(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        tokenized_query = self.text_preprocessor.process(query)
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
