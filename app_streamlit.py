import os
import streamlit as st
import faiss
import pickle
import re
from sentence_transformers import SentenceTransformer
import requests
from requests.exceptions import RequestException
from transformers import pipeline  # fallback QA

# --------------------------
# Load FAISS index & documents
# --------------------------
index = faiss.read_index("faiss_index/index.faiss")
with open("faiss_index/chunks.pkl", "rb") as f:
    documents = pickle.load(f)

# --------------------------
# Embedding model
# --------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Ollama settings
# --------------------------
OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

# --------------------------
# Streamlit UI
# --------------------------
st.title("📚 Chat with your Documents (Local LLM Edition)")

# --------------------------
# Helper functions
# --------------------------
def clean_text(text):
    """Clean PDF text chunks"""
    text = re.sub(r'\S+@\S+', '', text)         # remove emails
    text = re.sub(r'\[\d+\]', '', text)         # remove references like [1]
    text = re.sub(r'Fig\.\s*\d+', '', text)     # remove figure numbers
    text = re.sub(r'http\S+', '', text)         # remove URLs
    text = re.sub(r'\s+', ' ', text)            # collapse whitespace
    return text.strip()

def generate_answer_ollama(query, context, base_url=OLLAMA_BASE, model=OLLAMA_MODEL, timeout=60):
    """
    Generates a natural answer using the LLM.
    Allows reasoning and paraphrasing, rather than verbatim extraction.
    """
    prompt = f"""
Using the context below, answer the question naturally in your own words.
You can summarize, explain, or paraphrase.
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

def summarize_chunk(chunk):
    """
    Optional: summarize each chunk before sending to LLM to reduce noise
    """
    return generate_answer_ollama("Summarize this text briefly:", chunk) or chunk

# --------------------------
# Main query input
# --------------------------
query = st.text_input("Ask a question:")

if query:
    # Encode query
    query_vec = embedder.encode([query])
    
    # Retrieve top-k chunks (k can be adjusted)
    k = 3
    D, I = index.search(query_vec, k=k)
    
    # Clean chunks
    context_chunks = [clean_text(documents[i]) for i in I[0]]
    
    # Optional: summarize each chunk
    summarized_chunks = []
    for chunk in context_chunks:
        summarized_chunks.append(summarize_chunk(chunk))
    context = " ".join(summarized_chunks)
    
    # Generate answer via Ollama
    answer = generate_answer_ollama(query, context)
    
    # Fallback QA if LLM fails
    if answer is None:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        result = qa_pipeline(question=query, context=context)
        answer = result.get("answer", "(no answer)")
    
    # Display
    st.subheader("Answer:")
    st.write(answer)
    
    st.subheader("Context used:")
    st.write(context)
