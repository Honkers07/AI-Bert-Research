import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# === SETTINGS ===
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"

QUERY_TITLE = "Adjustable lens with microfluidic actuator"
QUERY_CLAIM = "Adjustable lens with microfluidic actuator"
QUERY_DESCRIPTION = "An optical lens comprising a deformable membrane"  # Leave empty to skip description

SIMILARITY_MODE = "cosine"  # Options: "cosine" or "euclidean"
CLAIM_WEIGHT = 0.7
DESC_WEIGHT = 0.3
USE_FULL_SEARCH = True      # True = search all, False = limited NN search
NN_K = 1000                 # Number of nearest neighbors to retrieve (if USE_FULL_SEARCH is False)
TOP_K = 5                   # Final top results

assert abs(CLAIM_WEIGHT + DESC_WEIGHT - 1.0) < 1e-5, "❌ Weights must sum to 1.0"

# === LOAD MODEL ===
print("📦 Loading model...")
model = SentenceTransformer(MODEL_NAME)

# === LOAD INDICES ===
if SIMILARITY_MODE == "cosine":
    claim_index_path = "data/index_claim_cosine.faiss"
    desc_index_path = "data/index_desc_cosine.faiss"
    normalize_query = True
elif SIMILARITY_MODE == "euclidean":
    claim_index_path = "data/index_claim_euclidean.faiss"
    desc_index_path = "data/index_desc_euclidean.faiss"
    normalize_query = False
else:
    raise ValueError("❌ Invalid SIMILARITY_MODE. Use 'cosine' or 'euclidean'.")

print("📂 Loading FAISS indices...")
claim_index = faiss.read_index(claim_index_path)
desc_index = faiss.read_index(desc_index_path)

# === LOAD METADATA ===
with open("data/patent_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

claim_metadata = [m for m in metadata if m["section"] == "claim"]
desc_metadata = [m for m in metadata if m["section"] == "description"]

# === EMBED QUERIES ===
query_vectors = {}

if QUERY_CLAIM.strip():
    vec = model.encode([QUERY_CLAIM], convert_to_numpy=True).astype("float32")
    if normalize_query:
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    query_vectors["claim"] = vec

if QUERY_DESCRIPTION.strip():
    vec = model.encode([QUERY_DESCRIPTION], convert_to_numpy=True).astype("float32")
    if normalize_query:
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    query_vectors["description"] = vec

if not query_vectors:
    raise ValueError("❌ At least one of QUERY_CLAIM or QUERY_DESCRIPTION must be non-empty.")

# === SEARCH FUNCTION ===
def search_index(index, metadata_list, vec, weight, section_name):
    k = len(metadata_list) if USE_FULL_SEARCH else min(NN_K, len(metadata_list))
    D, I = index.search(vec, k)

    results = {}
    for score, idx in zip(D[0], I[0]):
        item = metadata_list[idx]
        pub = item["publication_number"]
        results[pub] = {
            "score": weight * score,
            "title": item["title"],
            "section": section_name,
            "date": item.get("publication_date", "")
        }
    return results

# === SEARCH & COMBINE ===
combined = {}

if "claim" in query_vectors:
    print("🔎 Searching claim index...")
    claim_results = search_index(claim_index, claim_metadata, query_vectors["claim"], CLAIM_WEIGHT if "description" in query_vectors else 1.0, "claim")
    for pub, entry in claim_results.items():
        combined[pub] = {
            "score": entry["score"],
            "title": entry["title"],
            "date": entry["date"],
            "sections": ["claim"]
        }

if "description" in query_vectors:
    print("🔎 Searching description index...")
    desc_results = search_index(desc_index, desc_metadata, query_vectors["description"], DESC_WEIGHT if "claim" in query_vectors else 1.0, "description")
    for pub, entry in desc_results.items():
        if pub in combined:
            combined[pub]["score"] += entry["score"]
            combined[pub]["sections"].append("description")
        else:
            combined[pub] = {
                "score": entry["score"],
                "title": entry["title"],
                "date": entry["date"],
                "sections": ["description"]
            }

# === SORT & FILTER ===
sorted_results = sorted(combined.items(), key=lambda x: -x[1]["score"])[:TOP_K]

# === PRINT RESULTS ===
print(f"\n🔎 Top {TOP_K} results for query title: '{QUERY_TITLE}'\n")

if not sorted_results:
    print("No matches found.")
else:
    for rank, (pub, info) in enumerate(sorted_results, 1):
        sections = ", ".join(info["sections"])
        print(f"{rank}. {info['title']} (📄 {pub}, 📅 {info['date']}) | Score: {info['score']:.4f} | Sections: {sections}")
