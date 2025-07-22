import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# === LOAD MODEL & INDEX ONCE ===
print("📦 Loading model...")
model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")

print("📂 Loading FAISS indices...")
claim_index = faiss.read_index("data/index_claim_cosine.faiss")
desc_index = faiss.read_index("data/index_desc_cosine.faiss")

with open("data/patent_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

claim_metadata = [m for m in metadata if m["section"] == "claim"]
desc_metadata = [m for m in metadata if m["section"] == "description"]

# === SEARCH FUNCTION ===
def search_patents(
    query_title: str,
    query_claim: str,
    query_desc: str,
    similarity_mode: str = "cosine",
    claim_weight: float = 0.7,
    description_weight: float = 0.3,
    top_k: int = 5
):
    claim_vec = desc_vec = None

    if query_claim:
        claim_vec = model.encode([query_claim], convert_to_numpy=True).astype("float32")
    if query_desc:
        desc_vec = model.encode([query_desc], convert_to_numpy=True).astype("float32")

    # Normalize vectors for cosine similarity
    if similarity_mode == "cosine":
        if claim_vec is not None:
            claim_vec /= np.linalg.norm(claim_vec, axis=1, keepdims=True)
        if desc_vec is not None:
            desc_vec /= np.linalg.norm(desc_vec, axis=1, keepdims=True)

    def search_index(index, metadata_list, vector, weight, section):
        k = min(top_k * 10, len(metadata_list))  # Retrieve more for reranking
        D, I = index.search(vector, k)

        results = {}
        for score, idx in zip(D[0], I[0]):
            item = metadata_list[idx]
            pub_num = item["publication_number"]
            results[pub_num] = {
                "score": float(weight * score),
                "title": item["title"],
                "section": section,
                "date": item.get("publication_date", ""),
            }
        return results

    # Perform searches
    claim_results = search_index(claim_index, claim_metadata, claim_vec, claim_weight, "claim") if claim_vec is not None else {}
    desc_results = search_index(desc_index, desc_metadata, desc_vec, description_weight, "description") if desc_vec is not None else {}

    # Combine results
    combined = {}

    for pub, entry in claim_results.items():
        combined[pub] = {
            "score": entry["score"],
            "title": entry["title"],
            "date": entry["date"],
            "sections": ["claim"]
        }

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

    # If both vectors exist, filter to entries that matched both
    if claim_vec is not None and desc_vec is not None:
        filtered = {k: v for k, v in combined.items() if len(v["sections"]) == 2}
    else:
        filtered = combined

    # Sort and return top_k
    top_results = sorted(filtered.items(), key=lambda x: -x[1]["score"])[:top_k]

    return [
        {
            "rank": i + 1,
            "publication_number": pub,
            **info
        }
        for i, (pub, info) in enumerate(top_results)
    ]
