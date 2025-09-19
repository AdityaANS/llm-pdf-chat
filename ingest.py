import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import pdfplumber

# Paths
DATA_DIR = "data/Uploads"
INDEX_DIR = "faiss_index"
os.makedirs(INDEX_DIR, exist_ok=True)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []

def extract_text_from_pdf(path):
    text_chunks = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                # Split into smaller chunks (e.g., 500 chars)
                for i in range(0, len(text), 500):
                    text_chunks.append(text[i:i+500])
            else:
                print(f"⚠️ Page {page_num} in {os.path.basename(path)} has no text")
    return text_chunks

# Collect text chunks from PDFs
for filename in os.listdir(DATA_DIR):
    print("📄 Found file:", filename)
    if filename.lower().endswith(".pdf"):  # case-insensitive check
        path = os.path.join(DATA_DIR, filename)
        print("➡️ Processing:", path)
        file_chunks = extract_text_from_pdf(path)
        print(f"✅ Extracted {len(file_chunks)} chunks from {filename}")
        chunks.extend(file_chunks)

print(f"📊 Total extracted chunks: {len(chunks)}")

if len(chunks) == 0:
    print("⚠️ No text found in PDFs. Make sure your PDFs are in the 'data' folder and contain selectable text.")
    exit()

# Embed chunks
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

# Save chunks so we can retrieve text later
with open(os.path.join(INDEX_DIR, "chunks.pkl"), "wb") as f:
    pickle.dump(chunks, f)

print(f"✅ Index built with {len(chunks)} chunks and saved to {INDEX_DIR}")
