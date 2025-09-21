#!/usr/bin/env python3
"""
Robust FAISS + SentenceTransformer search script.
Usage examples:
  python Matching.py
  python Matching.py --claim-index data/index_claim_cosine.faiss --claim-idmap data/idmap_claim.json \
                     --desc-index data/index_desc_cosine.faiss --desc-idmap data/idmap_desc.json \
                     --input "C:/Users/wilbe/OneDrive/Desktop/Filtered Patents/XCitedArticle(...).json" \
                     --output data/x_cited_ranking_results.jsonl
"""
import os
import sys
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time # For progress indication

# -------------------------
# Helpers
# -------------------------
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def check_file(path, friendly_name=None):
    if not os.path.isfile(path):
        name = friendly_name or path
        eprint(f"❌ Required file not found: {name}\n   Path checked: {path}")
        return False
    return True

def load_jsonl(file_path):
    items = []
    if not check_file(file_path, "Input JSONL"):
        return items
    with open(file_path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                eprint(f"⚠️ Skipping bad JSON line {n} in {file_path}: {e}")
    return items

def save_jsonl(file_path, items):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def load_idmap(idmap_path):
    if not check_file(idmap_path, "ID map"):
        return None
    try:
        with open(idmap_path, "r", encoding="utf-8") as f:
            idmap = json.load(f)
        # Ensure all keys are strings for consistent lookup
        idmap_str = {str(k): v for k, v in idmap.items()}
        return idmap_str
    except Exception as e:
        eprint(f"❌ Failed to load idmap {idmap_path}: {e}")
        return None

def load_index(index_path, idmap_path):
    if not check_file(index_path, "FAISS index"):
        return None, None
    idmap = load_idmap(idmap_path)
    if idmap is None:
        return None, None
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        eprint(f"❌ Failed to read FAISS index '{index_path}': {e}")
        return None, None
    return index, idmap

def normalize_vec(vec):
    # vec: numpy array shape (1, dim) or (n, dim)
    if vec is None or vec.size == 0: # Handle empty vectors
        return None
    vec = np.array(vec, dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors
    norms[norms == 0] = 1.0 # If a vector is all zeros, its norm is 0. Normalizing it to 1.0 prevents NaNs.
    return vec / norms

def embed_text(model, text):
    if not text or not str(text).strip():
        return None
    # sentence-transformers returns (n, dim) when given list
    vec = model.encode([str(text)], convert_to_numpy=True)
    return normalize_vec(vec)

def query_index(index, idmap, vec, top_k=10):
    if index is None or idmap is None or vec is None:
        return []
    # faiss expects float32 and 2D array
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    try:
        D, I = index.search(vec, top_k)
    except Exception as e:
        eprint(f"⚠️ FAISS search failed: {e}")
        return []
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: # Invalid index
            continue
        # ID map keys are now guaranteed to be strings from load_idmap
        pub_id = idmap.get(str(idx))
        if pub_id:
            results.append({"publication_number": pub_id, "score": float(dist)})
    return results

# -------------------------
# Main script
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Rank X-cited patents using FAISS + SBERT embeddings")
    parser.add_argument("--claim-index", default="data/index_claim_cosine.faiss", help="FAISS index for claims")
    parser.add_argument("--desc-index",  default="data/index_desc_cosine.faiss",  help="FAISS index for descriptions")
    parser.add_argument("--claim-idmap", default="data/idmap_claim.json", help="ID map JSON for claim index (idx -> pubid)")
    parser.add_argument("--desc-idmap",  default="data/idmap_desc.json",  help="ID map JSON for desc index (idx -> pubid)")
    parser.add_argument("--input",       default=r"C:\Users\wilbe\OneDrive\Desktop\Filtered Patents\XCitedArticle(with claim and description for x cited articles).json",
                        help="Input JSONL (or single JSON) file with x-cited records")
    parser.add_argument("--output",      default="data/x_cited_ranking_results.jsonl", help="Output JSONL results")
    parser.add_argument("--topk",        type=int, default=10, help="Top-K results from each index")
    args = parser.parse_args()

    # Validate input file exists
    if not check_file(args.input, "Input file"):
        eprint("\nPlease provide a valid --input path to your XCitedArticle JSON/JSONL file.")
        sys.exit(2)

    # Load indices and idmaps (with clear messages if missing)
    eprint("📂 Loading FAISS indices and idmaps...")
    claim_index, claim_idmap = load_index(args.claim_index, args.claim_idmap)
    desc_index, desc_idmap = load_index(args.desc_index, args.desc_idmap)

    if claim_index is None or claim_idmap is None:
        eprint("\n❌ Claim index or idmap failed to load. Please verify the paths and that the files exist.")
        eprint(f"Expected claim index: {args.claim_index}")
        eprint(f"Expected claim idmap: {args.claim_idmap}")
        # list files in data folder to help debugging
        data_dir = os.path.dirname(args.claim_index) or "."
        eprint(f"Files in '{data_dir}': {os.listdir(data_dir) if os.path.isdir(data_dir) else 'dir not found'}")
        sys.exit(3)

    if desc_index is None or desc_idmap is None:
        eprint("\n❌ Description index or idmap failed to load. Please verify the paths and that the files exist.")
        eprint(f"Expected desc index: {args.desc_index}")
        eprint(f"Expected desc idmap: {args.desc_idmap}")
        data_dir = os.path.dirname(args.desc_index) or "."
        eprint(f"Files in '{data_dir}': {os.listdir(data_dir) if os.path.isdir(data_dir) else 'dir not found'}")
        sys.exit(4)

    # Load SBERT model (this may download model files on first run)
    eprint("📥 Loading embedding model (SentenceTransformer 'all-mpnet-base-v2')...")
    try:
        model = SentenceTransformer("all-mpnet-base-v2")
    except Exception as e:
        eprint(f"❌ Failed to load SentenceTransformer model: {e}")
        eprint("Please check your internet connection or install necessary dependencies.")
        sys.exit(5)

    # Load input records (support either JSONL or single JSON)
    eprint("📂 Loading input records...")
    items = load_jsonl(args.input) # Try loading as JSONL first

    if not items: # If JSONL didn't work or was empty, try loading as a single JSON object
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    items = [data] # Wrap single dict in a list for consistent processing
                else:
                    eprint("❌ Unrecognized JSON structure in input file (expected list or dictionary).")
                    sys.exit(7)
        except json.JSONDecodeError as e:
            eprint(f"❌ Failed to parse input file as JSONL or single JSON: {e}")
            eprint("Please ensure your input file is valid JSONL or a single JSON object (list/dict).")
            sys.exit(7)
        except Exception as e:
            eprint(f"❌ An unexpected error occurred while loading input file: {e}")
            sys.exit(7)

    if not items:
        eprint("❌ No records loaded from input file after trying JSONL and single JSON. The file might be empty or malformed.")
        sys.exit(8)

    # Main processing loop
    results = []
    total_items = len(items)
    eprint(f"🔎 Processing {total_items:,} records...")
    start_time = time.time()

    for i, art in enumerate(items):
        if (i + 1) % 100 == 0 or (i + 1) == total_items:
            elapsed = time.time() - start_time
            eprint(f"  Processed {i+1}/{total_items} records ({elapsed:.2f}s elapsed)...")

        orig_id = art.get("original_publication_number") or art.get("publication_number") or art.get("original_patent")
        x_id = art.get("x_cited_publication_number") or art.get("x_cited_patent") or art.get("x_cited")

        claim_text = art.get("claim") or art.get("claims") or art.get("claim_text") or ""
        desc_text = art.get("description") or art.get("desc") or art.get("long_description") or ""
        title = art.get("title") or art.get("original_title") or ""

        # Embed text
        claim_vec = embed_text(model, claim_text)
        desc_vec = embed_text(model, desc_text)
        title_vec = embed_text(model, title)

        # Query indices
        claim_matches = query_index(claim_index, claim_idmap, claim_vec, top_k=args.topk)
        desc_matches  = query_index(desc_index,  desc_idmap,  desc_vec,  top_k=args.topk)
        
        # Optionally, query claim index with title if it's considered relevant for claim similarity
        # If you have a separate title index, you would query that instead.
        title_matches_in_claims = []
        if title_vec is not None:
            title_matches_in_claims = query_index(claim_index, claim_idmap, title_vec, top_k=args.topk)

        # Merge scores (take max for a given publication_number across all search types)
        combined = {}
        for m in claim_matches:
            combined[m["publication_number"]] = max(combined.get(m["publication_number"], 0.0), m["score"])
        for m in desc_matches:
            combined[m["publication_number"]] = max(combined.get(m["publication_number"], 0.0), m["score"])
        for m in title_matches_in_claims:
            combined[m["publication_number"]] = max(combined.get(m["publication_number"], 0.0), m["score"])


        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        rank_position = None
        for idx, (pub_id, score) in enumerate(ranked, 1):
            if x_id and pub_id == x_id:
                rank_position = idx
                break

        result = {
            "original_patent": orig_id,
            "x_cited_patent": x_id,
            "rank_position": rank_position,
            "top_match": ranked[0] if ranked else None,
            "total_matches": len(ranked)
        }
        results.append(result)
        # eprint(f"🔍 {orig_id} vs {x_id} → rank = {rank_position}") # Too verbose for many records

    # Save
    save_jsonl(args.output, results)
    end_time = time.time()
    eprint(f"\n✅ Done. Processed {total_items:,} records in {end_time - start_time:.2f} seconds.")
    eprint(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()