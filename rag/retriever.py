import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────
EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


# ── Load embedder, FAISS index, chunks ───────────────────
print("Loading RAG components...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index    = faiss.read_index(str(EMBEDDINGS_DIR / "index.faiss"))

with open(EMBEDDINGS_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("RAG components loaded.")

# ── Retrieve relevant chunks ──────────────────────────────
def retrieve(query, top_k=5):
    query_vector = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, top_k)
    
    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunks[idx])
    return results

# ── Generate response ─────────────────────────────────────
def generate_response(disease_name, query, retrieved_chunks):
    context = "\n\n".join([c["text"] for c in retrieved_chunks])

    prompt = f"""You are an agricultural expert assistant.

Detected disease: {disease_name}

Knowledge base context:
{context}

User's specific question: {query}

Answer ONLY the user's specific question above. Do not give a generic treatment summary unless asked. Be direct and specific."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content

# ── Main RAG function ─────────────────────────────────────
def get_disease_info(disease_name, user_query=None):
    # Default query if none provided
    if not user_query:
        user_query = f"What are the symptoms, treatment and prevention for {disease_name}?"
    
    # Retrieve relevant chunks
    retrieved = retrieve(f"{disease_name} {user_query}", top_k=5)
    
    # Generate response
    response = generate_response(disease_name, user_query, retrieved)
    
    return {
        "disease"   : disease_name,
        "query"     : user_query,
        "response"  : response,
        "sources"   : list(set([c["source"] for c in retrieved]))
    }

# ── Test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with Potato Early Blight
    result = get_disease_info(
        disease_name="Potato___Early_blight",
        user_query="What fungicide should I use and how do I prevent spread?"
    )
    
    print(f"Disease  : {result['disease']}")
    print(f"Query    : {result['query']}")
    print(f"Sources  : {result['sources']}")
    print(f"\nResponse :\n{result['response']}")