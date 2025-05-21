import os
import json
from retrievers.hybrid_retriever import HybridRetriever

class HybridRetrieverWithFeedback(HybridRetriever):
    def __init__(self, *args, feedback_path="feedback.json", **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_path = feedback_path
        self.feedback_log = self._load_feedback()

    def record_feedback(self, query, doc, is_relevant):
        key = (query.strip(), doc.strip())
        self.feedback_log[key] = self.feedback_log.get(key, 0) + (1 if is_relevant else -1)

    def save_feedback(self):
        # Convert tuple keys to string for JSON
        json_friendly = {f"{q}||{d}": score for (q, d), score in self.feedback_log.items()}
        with open(self.feedback_path, "w") as f:
            json.dump(json_friendly, f, indent=2)

    def _load_feedback(self):
        if not os.path.exists(self.feedback_path):
            return {}
        with open(self.feedback_path, "r") as f:
            data = json.load(f)
        # Convert back to tuple keys
        return {tuple(k.split("||")): v for k, v in data.items()}

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

        # Apply feedback effect: boost or penalize scores based on feedback
        for doc in scores:
            feedback_score = self.feedback_log.get((query, doc), 0)
            # You can tune the feedback_weight as needed
            feedback_weight = 0.1
            scores[doc] += feedback_weight * feedback_score

        # Sort by combined score (including feedback)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def show_result_get_feedback(self, query: str, top_k: int = 3) -> None:
        results = self.retrieve(query, top_k)
        for doc, score in results:
            print(f"Score: {score:.2f}\nDoc: {doc}\n")
            feedback = input("👍👎 Was helpful? (y/n): ").strip().lower()
            self.record_feedback(query, doc, feedback == "y")
        
    def log_feedback(self):
        print("Feedback Summary:")
        for (query, doc), feedback in self.feedback_log.items():
            print(f"{query[:30]}... → {doc[:30]}... have Feedback: {feedback}")
