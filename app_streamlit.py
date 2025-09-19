import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS index
index = faiss.read_index("faiss_index/index.faiss")

# Load stored documents (chunks)
with open("faiss_index/chunks.pkl", "rb") as f:
    documents = pickle.load(f)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load QA model (distilbert for simplicity, runs on CPU)
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

st.title("📚 Chat with your Documents")

query = st.text_input("Ask a question:")

if query:
    # Encode query
    query_vec = embedder.encode([query])
    
    # Search FAISS
    D, I = index.search(query_vec, k=3)  # top-3 chunks
    context = " ".join([documents[i] for i in I[0]])

    # Run QA pipeline
    result = qa_pipeline(question=query, context=context)

    st.subheader("Answer:")
    st.write(result["answer"])

    st.subheader("Context used:")
    st.write(context)
