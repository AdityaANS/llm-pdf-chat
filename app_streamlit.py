import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
from requests.exceptions import RequestException

# Load FAISS index
index = faiss.read_index("faiss_index/index.faiss")

# Load stored documents
with open("faiss_index/chunks.pkl", "rb") as f:
    documents = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Ollama settings
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

st.title("📚 Chat with your Documents (Local LLM Edition)")

def generate_answer_ollama(query, context, base_url=OLLAMA_BASE, model=OLLAMA_MODEL, timeout=60):
    """
    Generates a natural answer using the LLM.
    Allows reasoning and paraphrasing, rather than verbatim extraction.
    """
    prompt = f"""
Answer the question using the context below.
You can summarize, explain, or rephrase the answer in natural language.
If the answer is not in the context, say 'I don't know'.

Context:
{context}

Question: {query}
Answer:
"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        text = msg.get("content") or data.get("response") or ""
        return text.strip()
    except RequestException:
        return None

# Input
query = st.text_input("Ask a question:")

if query:
    # Encode query
    query_vec = embedder.encode([query])
    
    # Retrieve top-k chunks
    D, I = index.search(query_vec, k=5)  # increased k for richer context
    context = " ".join([documents[i] for i in I[0]])
    
    # Generate answer via Ollama
    answer = generate_answer_ollama(query, context)
    
    # Fallback if LLM fails
    if answer is None:
        from transformers import pipeline
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_pipeline(question=query, context=context)
        answer = result.get("answer", "(no answer)")
    
    # Show results
    st.subheader("Answer:")
    st.write(answer)
    
    st.subheader("Context used:")
    st.write(context)


