import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.preprocessor import TextPreprocessor
from utils.loader import DocumentLoader

class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = []
        self.corpus = []
        self.text_preprocessor = TextPreprocessor()
        self.index = None
        
    def build_index(self, documents):
        """
        Build the FAISS index from documents
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        self.corpus = [self.text_preprocessor.process(doc) for doc in self.documents]
        corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=False, show_progress_bar=True)
        dimension = corpus_embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(corpus_embeddings))
        
    def retrieve(self, query, top_k=3):
        """
        Retrieve the most similar documents to the query
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, similarity_score)
        """
        query_embedding = self.model.encode([query])[0]
        D, I = self.index.search(np.array([query_embedding]), top_k)
        found_indices = I[0]
        similarity_scores = D[0]
        raw_results = [(float(similarity_scores[rank]), self.documents[i]) for rank, i in enumerate(found_indices)]
        max_score = max(score for score, _ in raw_results)
        if max_score == 0:
            return raw_results
        normalized_results = [(score / max_score, doc) for score, doc in raw_results]
        return normalized_results

    def show_fancy_results(self, query: str, top_k: int = 3) -> None:
        results = self.retrieve(query, top_k)
        for score, doc in results:
            print(f"Score: {score:.2f}\nDoc: {doc}\n")
