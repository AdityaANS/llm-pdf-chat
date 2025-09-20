import os
import streamlit as st
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import requests
from requests.exceptions import RequestException
from transformers import pipeline  # fallback

# --- Load FAISS index ---
index = faiss.read_index("faiss_index/index.faiss")

# Load stored documents
with open("faiss_index/chunks.pkl", "rb") as f:
    documents = pickle.load(f)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Fallback QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Ollama settings
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

st.title("📚 Chat with your Documents (Local LLM Edition)")

def generate_answer_ollama(query, context, base_url=OLLAMA_BASE, model=OLLAMA_MODEL, timeout=60):
    if not context.strip():
        return "I don't know (no context found)."

    # Build a safe prompt
    prompt = f"""
You are a helpful assistant. Answer the question using ONLY the context below. 
If the answer is not present in the context, say 'I don't know'.

Context:
{context}

Question:
{query}
Answer:"""
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

# --- Streamlit UI ---
query = st.text_input("Ask a question:")

if query:
    # Encode query and search FAISS
    query_vec = embedder.encode([query])
    D, I = index.search(query_vec, k=3)

    # Collect retrieved documents safely
    retrieved_docs = [documents[i] for i in I[0] if i < len(documents)]
    context = " ".join(retrieved_docs)

    # Generate answer
    answer = generate_answer_ollama(query, context)

    # Fallback if Ollama fails
    if answer is None or answer.strip() == "":
        if context.strip():
            result = qa_pipeline(question=query, context=context)
            answer = result.get("answer", "I don't know")
        else:
            answer = "I don't know (no context found)."

    # Display answer and context
    st.subheader("Answer:")
    st.write(answer)

    st.subheader("Context used:")
    st.write(context if context.strip() else "(No context found)")

