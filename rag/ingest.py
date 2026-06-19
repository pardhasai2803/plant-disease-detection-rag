import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ─────────────────────────────────────────────────
KB_DIR        = Path("knowledge_base")
EMBEDDINGS_DIR = Path("embeddings")
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# ── Load embedding model ──────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

# ── Chunk each document ───────────────────────────────────
def chunk_document(text, filename):
    """Split document into section-based chunks."""
    chunks = []
    lines  = text.strip().split('\n')
    
    current_section = ""
    current_content = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Each line starting with a keyword is a new section
        sections = ["DISEASE:", "CROP:", "TYPE:", "SYMPTOMS:",
                    "CAUSES:", "FAVORABLE_CONDITIONS:", "CHEMICAL_TREATMENT:",
                    "ORGANIC_TREATMENT:", "PREVENTION:", "SEVERITY:", "ADDITIONAL_INFO:"]
        
        is_section = any(line.startswith(s) for s in sections)
        
        if is_section:
            # Save previous section as chunk
            if current_content.strip():
                chunks.append({
                    "text"    : f"{current_section} {current_content}".strip(),
                    "source"  : filename,
                    "section" : current_section
                })
            # Start new section
            parts           = line.split(":", 1)
            current_section = parts[0].strip()
            current_content = parts[1].strip() if len(parts) > 1 else ""
        else:
            current_content += " " + line

    # Save last section
    if current_content.strip():
        chunks.append({
            "text"    : f"{current_section} {current_content}".strip(),
            "source"  : filename,
            "section" : current_section
        })

    return chunks

# ── Process all documents ─────────────────────────────────
all_chunks = []
txt_files  = sorted(KB_DIR.glob("*.txt"))

print(f"\nChunking {len(txt_files)} documents...")
for txt_file in txt_files:
    with open(txt_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = chunk_document(text, txt_file.stem)
    all_chunks.extend(chunks)
    print(f"  {txt_file.stem}: {len(chunks)} chunks")

print(f"\nTotal chunks: {len(all_chunks)}")

# ── Embed all chunks ──────────────────────────────────────
print("\nEmbedding chunks...")
texts      = [chunk["text"] for chunk in all_chunks]
embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings).astype("float32")

print(f"Embeddings shape: {embeddings.shape}")

# ── Build FAISS index ─────────────────────────────────────
print("\nBuilding FAISS index...")
dimension  = embeddings.shape[1]
index      = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print(f"FAISS index built — {index.ntotal} vectors, dimension {dimension}")

# ── Save index + chunks ───────────────────────────────────
faiss.write_index(index, str(EMBEDDINGS_DIR / "index.faiss"))

with open(EMBEDDINGS_DIR / "chunks.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print(f"\nSaved:")
print(f"  {EMBEDDINGS_DIR / 'index.faiss'}")
print(f"  {EMBEDDINGS_DIR / 'chunks.pkl'}")
print("\nIngestion complete!")