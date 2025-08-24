import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === SETTINGS ===
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
CLAIM_WEIGHT = 0.7
DESC_WEIGHT = 0.3

# Files (adjust paths if needed)
XCITED_FILE = r"C:\Users\erico\Downloads\XCitedArticle(with claim and description for x cited articles).json"
CLAIM_FILE = r"C:\Users\erico\Downloads\claims_matched.jsonl"        # optional fallback
DESCRIPTION_FILE = r"C:\Users\erico\Downloads\descriptions_matched.jsonl"  # optional fallback

OUTPUT_FILE = "xcited_search_results.json"
USE_FULL_SEARCH = True
BATCH_SAVE_INTERVAL = 50

# === LOAD FAISS INDICES (cosine) ===
print("📂 Loading FAISS cosine indices...")
claim_index = faiss.read_index("data/index_claim_cosine.faiss")
desc_index = faiss.read_index("data/index_desc_cosine.faiss")

# === LOAD METADATA ===
with open("data/patent_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

claim_metadata = [m for m in metadata if m["section"] == "claim"]
desc_metadata  = [m for m in metadata if m["section"] == "description"]

claim_pub_to_idx = {m["publication_number"]: i for i, m in enumerate(claim_metadata)}
desc_pub_to_idx  = {m["publication_number"]: i for i, m in enumerate(desc_metadata)}

VECTOR_DIM = claim_index.d  # dimension (should equal desc_index.d)

# === OPTIONAL: load claim/description text files as fallback (if you want text-based embedding) ===
claims_text = {}
descs_text = {}
if os.path.exists(CLAIM_FILE):
    print("📖 Loading claims file for fallbacks...")
    with open(CLAIM_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                pub = obj.get("publication_number")
                if pub:
                    # adjust key names if your JSONL differs
                    claims_text[pub] = obj.get("claim_text", "") or obj.get("claims","") or ""
            except Exception:
                continue

if os.path.exists(DESCRIPTION_FILE):
    print("📖 Loading descriptions file for fallbacks...")
    with open(DESCRIPTION_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                pub = obj.get("publication_number")
                if pub:
                    descs_text[pub] = obj.get("description", "") or obj.get("desc","") or ""
            except Exception:
                continue

# === Embedding model (lazy loaded if needed) ===
_model = None
def ensure_model():
    global _model
    if _model is None:
        print("📦 Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)

def embed_and_normalize(text):
    ensure_model()
    vec = _model.encode([text], convert_to_numpy=True).astype("float32")
    vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
    return vec  # shape (1, dim)

# === Helper: reconstruct vector from FAISS index if possible ===
def reconstruct_from_index(index, idx):
    try:
        out = np.empty(VECTOR_DIM, dtype="float32")
        index.reconstruct(idx, out)        # for IndexFlat* this works
        out = out.reshape(1, -1)
        # normalize to be safe (index built normalized for cosine)
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out
    except Exception:
        return None

# === Add missing patent into FAISS (both claim & desc) ===
def add_patent_to_faiss(pub_num, title, claim_text, desc_text):
    # use text if provided, otherwise try to use fallback text maps
    if not claim_text:
        claim_text = claims_text.get(pub_num, "")
    if not desc_text:
        desc_text = descs_text.get(pub_num, "")

    if not claim_text and not desc_text:
        raise ValueError(f"No text available to add patent {pub_num}. Provide claim/description text or files.")

    # create embeddings (normalize for cosine)
    if claim_text:
        cvec = embed_and_normalize(claim_text)  # (1, dim)
        claim_index.add(cvec)
        claim_metadata.append({"publication_number": pub_num, "title": title, "section": "claim"})
        claim_pub_to_idx[pub_num] = len(claim_metadata) - 1

    if desc_text:
        dvec = embed_and_normalize(desc_text)
        desc_index.add(dvec)
        desc_metadata.append({"publication_number": pub_num, "title": title, "section": "description"})
        desc_pub_to_idx[pub_num] = len(desc_metadata) - 1

# === Build query vector for original patent (tries: provided text -> reconstruct from index -> fallback to title) ===
def build_query_vectors(orig_pub, orig_claim_text="", orig_desc_text="", orig_title=""):
    claim_q = None
    desc_q  = None

    # 1) claim: try provided claim text
    if orig_claim_text and orig_claim_text.strip():
        claim_q = embed_and_normalize(orig_claim_text)
    else:
        # try reconstruct from claim index
        idx = claim_pub_to_idx.get(orig_pub)
        if idx is not None:
            claim_q = reconstruct_from_index(claim_index, idx)
        else:
            # final fallback: try claim text file
            ct = claims_text.get(orig_pub)
            if ct:
                claim_q = embed_and_normalize(ct)

    # 2) description
    if orig_desc_text and orig_desc_text.strip():
        desc_q = embed_and_normalize(orig_desc_text)
    else:
        idx = desc_pub_to_idx.get(orig_pub)
        if idx is not None:
            desc_q = reconstruct_from_index(desc_index, idx)
        else:
            dt = descs_text.get(orig_pub)
            if dt:
                desc_q = embed_and_normalize(dt)

    # last resort: if both are None, embed the title as a single query (not ideal, but prevents empty runs)
    if claim_q is None and desc_q is None:
        if orig_title and orig_title.strip():
            print(f"⚠️ No claim/desc for {orig_pub}. Falling back to title embedding.")
            claim_q = embed_and_normalize(orig_title)
        else:
            # truly no data — return None, None
            return None, None

    return claim_q, desc_q

# === Weighted search using prebuilt query vectors (already normalized for cosine) ===
def weighted_search_from_vecs(claim_vec, desc_vec):
    combined_scores = {}

    if claim_vec is not None:
        k = len(claim_metadata) if USE_FULL_SEARCH else min(1000, len(claim_metadata))
        D, I = claim_index.search(claim_vec, k)
        for score, idx in zip(D[0], I[0]):
            pub = claim_metadata[idx]["publication_number"]
            combined_scores[pub] = combined_scores.get(pub, 0.0) + CLAIM_WEIGHT * float(score)

    if desc_vec is not None:
        k = len(desc_metadata) if USE_FULL_SEARCH else min(1000, len(desc_metadata))
        D, I = desc_index.search(desc_vec, k)
        for score, idx in zip(D[0], I[0]):
            pub = desc_metadata[idx]["publication_number"]
            combined_scores[pub] = combined_scores.get(pub, 0.0) + DESC_WEIGHT * float(score)

    # sort descending (best scores first)
    sorted_results = sorted(combined_scores.items(), key=lambda x: -x[1])
    return sorted_results

# === MAIN LOOP: process x-cited file ===
results = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

processed_pairs = {(r["original_patent"], r["cited_patent"]) for r in results}

with open(XCITED_FILE, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        data = json.loads(line)
        orig_pub = data.get("original_patent")
        orig_title = data.get("original_title", "")
        orig_claim = data.get("original_claim", "") or data.get("original_claims", "")
        orig_desc  = data.get("original_description", "") or data.get("original_desc", "")

        cited_pub = data.get("cited_patent")
        cited_title = data.get("cited_title", "")
        cited_claim = data.get("cited_claim", "") or data.get("cited_claims", "")
        cited_desc  = data.get("cited_description", "") or data.get("cited_desc", "")

        if (orig_pub, cited_pub) in processed_pairs:
            continue

        # Ensure cited patent exists in FAISS; if not, add it (requires claim/desc text)
        if (cited_pub not in claim_pub_to_idx) or (cited_pub not in desc_pub_to_idx):
            print(f"➕ Adding missing cited patent {cited_pub} to FAISS...")
            add_patent_to_faiss(cited_pub, cited_title, cited_claim, cited_desc)

        # Build query vectors for original patent
        claim_vec, desc_vec = build_query_vectors(orig_pub, orig_claim, orig_desc, orig_title)

        if claim_vec is None and desc_vec is None:
            print(f"❌ Skipping {orig_pub} — no way to build query vectors (no text and not in index).")
            results.append({
                "original_patent": orig_pub,
                "original_title": orig_title,
                "cited_patent": cited_pub,
                "cited_title": cited_title,
                "rank": None,
                "total_results": 0,
                "note": "no_query_vectors"
            })
            continue

        sorted_results = weighted_search_from_vecs(claim_vec, desc_vec)
        total_results = len(sorted_results)

        # Find rank (1-based) of cited patent
        rank = next((i+1 for i, (pub, _) in enumerate(sorted_results) if pub == cited_pub), None)

        results.append({
            "original_patent": orig_pub,
            "original_title": orig_title,
            "cited_patent": cited_pub,
            "cited_title": cited_title,
            "rank": rank,
            "total_results": total_results
        })

        # periodic save
        if line_num % BATCH_SAVE_INTERVAL == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
                json.dump(results, out_f, ensure_ascii=False, indent=2)
            print(f"💾 Saved progress at {line_num} lines.")

# final save
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, ensure_ascii=False, indent=2)

print("✅ Done. Results saved to", OUTPUT_FILE)
