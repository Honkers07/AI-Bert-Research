import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# === SETTINGS ===
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
QUERY = "Lens system for optical imaging with variable focus"
SIMILARITY_MODE = "cosine"  # Options: "cosine" or "euclidean"
TOP_K = 5

# === LOAD MODEL ===
print("📦 Loading model...")
model = SentenceTransformer(MODEL_NAME)

# === LOAD INDEX + METADATA BASED ON SIMILARITY MODE ===
if SIMILARITY_MODE == "cosine":
    index_path = "patent_index_cosine.faiss"
    normalize_query = True
elif SIMILARITY_MODE == "euclidean":
    index_path = "patent_index_euclidean.faiss"
    normalize_query = False
else:
    raise ValueError("❌ Invalid SIMILARITY_MODE. Use 'cosine' or 'euclidean'.")

print(f"📂 Loading {SIMILARITY_MODE} index from {index_path}...")
index = faiss.read_index(index_path)

with open("patent_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === EMBED QUERY ===
print("🧠 Embedding query...")
query_vec = model.encode([QUERY], convert_to_numpy=True).astype("float32")

if normalize_query:
    query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

# === SEARCH ===
print(f"🔍 Searching top {TOP_K} most similar patents...")
D, I = index.search(query_vec, TOP_K)

# === SHOW RESULTS ===
print(f"\n🔎 Top {TOP_K} similar patents to: '{QUERY}'\n")
for rank, idx in enumerate(I[0]):
    match = metadata[idx]
    print(f"{rank+1}. {match['title']} (📄 {match['publication_number']}, 📅 {match['publication_date']})")
