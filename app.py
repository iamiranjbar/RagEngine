
import streamlit as st
from utils.loader import DocumentLoader
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_feedback_retriever import HybridRetrieverWithFeedback

ALPHA = 0.5

def run():
    st.set_page_config(page_title="RAG Engine", layout="centered")
    st.title("🔍 RAG Engine")
    st.markdown("Hybrid Retrieval + Feedback System")

    loaded_documents = DocumentLoader.load_documents_from_json("data/sample_docs.json")
    bm25_retriever = BM25Retriever(loaded_documents)
    dense_retriever = DenseRetriever()
    dense_retriever.build_index(loaded_documents)
    retriever = HybridRetrieverWithFeedback(bm25_retriever, dense_retriever, ALPHA)

    query = st.text_input("Enter your question:")

    if query:
        top_k = st.slider("Top K Results", 1, 10, 3)
        results = retriever.retrieve(query, top_k=top_k)

        st.subheader("🔎 Top Results")
        for i, (doc, score) in enumerate(results):
            st.markdown(f"**{i+1}.** _Score: {score:.2f}_  \n{doc}")
            key = f"feedback_{i}"
            if st.button("👍 Relevant", key=f"{key}_yes"):
                retriever.record_feedback(query, doc, True)
                retriever.save_feedback()
            if st.button("👎 Not Relevant", key=f"{key}_no"):
                retriever.record_feedback(query, doc, False)
                retriever.save_feedback()

if __name__ == "__main__":
  run()
