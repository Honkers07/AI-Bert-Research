import json
import numpy as np
import torch
import time
import faiss
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# === SETTINGS ===
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
MAX_TOKENS = 512
BATCH_SIZE = 16

DESCRIPTION_FILE = r"C:\\Users\\erico\\Downloads\\descriptions_matched.jsonl"
CLAIM_FILE = r"C:\\Users\\erico\\Downloads\\claims_matched.jsonl"

# === OUTPUT DIRECTORY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Output indices in /data
OUT_EC_DESC = os.path.join(DATA_DIR, "index_desc_euclidean.faiss")
OUT_EC_CLAIM = os.path.join(DATA_DIR, "index_claim_euclidean.faiss")
OUT_CS_DESC = os.path.join(DATA_DIR, "index_desc_cosine.faiss")
OUT_CS_CLAIM = os.path.join(DATA_DIR, "index_claim_cosine.faiss")
OUTPUT_METADATA = os.path.join(DATA_DIR, "patent_metadata.json")

print("📦 Loading model and tokenizer...")
model = SentenceTransformer(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

VECTOR_DIM = model.get_sentence_embedding_dimension()

index_desc_ec = faiss.IndexFlatL2(VECTOR_DIM)
index_claim_ec = faiss.IndexFlatL2(VECTOR_DIM)
index_desc_cs = faiss.IndexFlatIP(VECTOR_DIM)
index_claim_cs = faiss.IndexFlatIP(VECTOR_DIM)

metadata_store = []

def chunk_text(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.tokenize(text)
    chunks = []
    current = []
    for token in tokens:
        current.append(token)
        if len(current) == max_tokens:
            chunk_text = tokenizer.convert_tokens_to_string(current)
            chunks.append(chunk_text)
            current = []
    if current:
        chunk_text = tokenizer.convert_tokens_to_string(current)
        chunks.append(chunk_text)
    return chunks

def embed_texts(texts):
    all_embeddings = []
    for text in texts:
        token_len = len(tokenizer.tokenize(text))
        if token_len <= MAX_TOKENS:
            embedding = model.encode(text, convert_to_numpy=True)
        else:
            chunks = chunk_text(text)
            chunk_embeddings = model.encode(chunks, convert_to_numpy=True, batch_size=8)
            embedding = np.mean(chunk_embeddings, axis=0)
        all_embeddings.append(embedding)
    return np.array(all_embeddings).astype("float32")

print("📖 Loading descriptions...")
descriptions = []
with open(DESCRIPTION_FILE, "r", encoding="utf-8") as f_desc:
    for line in f_desc:
        data = json.loads(line)
        descriptions.append(data)

print("📖 Loading claims...")
claims_dict = {}
with open(CLAIM_FILE, "r", encoding="utf-8") as f_claim:
    for line in f_claim:
        data = json.loads(line)
        pub_num = data.get("publication_number")
        claims_dict[pub_num] = data

total_patents = len(descriptions)
print(f"🚀 Embedding {total_patents} descriptions + claims (total: {2 * total_patents} vectors)")

items_processed = 0
start_time = time.time()

not_found_claims = []

batch_desc = []
batch_claim = []
batch_meta_desc = []
batch_meta_claim = []

for patent in descriptions:
    pub_num = patent.get("publication_number")
    desc_text = patent.get("description", "")
    title = patent.get("title", "")

    claim_data = claims_dict.get(pub_num)
    claim_text = claim_data.get("claim_text", "") if claim_data else None

    if claim_text is None:
        not_found_claims.append(pub_num)

    batch_desc.append(desc_text)
    batch_meta_desc.append({
        "publication_number": pub_num,
        "title": title,
        "section": "description"
    })

    batch_claim.append(claim_text if claim_text else "")
    batch_meta_claim.append({
        "publication_number": pub_num,
        "title": title,
        "section": "claim"
    })

    if len(batch_desc) >= BATCH_SIZE:
        # Embed descriptions
        desc_embeddings = embed_texts(batch_desc)
        index_desc_ec.add(desc_embeddings)
        norms_desc = np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
        index_desc_cs.add(desc_embeddings / norms_desc)

        # Embed claims
        claim_embeddings = embed_texts(batch_claim)
        index_claim_ec.add(claim_embeddings)
        norms_claim = np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
        index_claim_cs.add(claim_embeddings / norms_claim)

        metadata_store.extend(batch_meta_desc + batch_meta_claim)

        items_processed += len(batch_desc) + len(batch_claim)
        elapsed = time.time() - start_time
        speed = items_processed / elapsed if elapsed > 0 else 0
        percent = (items_processed / (2 * total_patents)) * 100
        eta_seconds = ((2 * total_patents) - items_processed) / speed if speed > 0 else 0
        eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

        print(f"\rProcessed {items_processed}/{2 * total_patents} vectors ({percent:.2f}%) | Speed: {speed:.2f} items/s | ETA: {eta}", end="", flush=True)

        batch_desc, batch_claim = [], []
        batch_meta_desc, batch_meta_claim = [], []

# Final batch
if batch_desc:
    desc_embeddings = embed_texts(batch_desc)
    index_desc_ec.add(desc_embeddings)
    norms_desc = np.linalg.norm(desc_embeddings, axis=1, keepdims=True)
    index_desc_cs.add(desc_embeddings / norms_desc)

    claim_embeddings = embed_texts(batch_claim)
    index_claim_ec.add(claim_embeddings)
    norms_claim = np.linalg.norm(claim_embeddings, axis=1, keepdims=True)
    index_claim_cs.add(claim_embeddings / norms_claim)

    metadata_store.extend(batch_meta_desc + batch_meta_claim)

    items_processed += len(batch_desc) + len(batch_claim)
    elapsed = time.time() - start_time
    speed = items_processed / elapsed if elapsed > 0 else 0
    percent = (items_processed / (2 * total_patents)) * 100
    eta_seconds = ((2 * total_patents) - items_processed) / speed if speed > 0 else 0
    eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    print(f"\\rProcessed {items_processed}/{2 * total_patents} vectors ({percent:.2f}%) | Speed: {speed:.2f} items/s | ETA: {eta}", end="", flush=True)

print("\\n💾 Saving FAISS indices and metadata...")
faiss.write_index(index_desc_ec, OUT_EC_DESC)
faiss.write_index(index_claim_ec, OUT_EC_CLAIM)
faiss.write_index(index_desc_cs, OUT_CS_DESC)
faiss.write_index(index_claim_cs, OUT_CS_CLAIM)

with open(OUTPUT_METADATA, "w", encoding="utf-8") as f_out:
    json.dump(metadata_store, f_out, ensure_ascii=False, indent=2)

if not_found_claims:
    print(f"\\n⚠️ Warning: {len(not_found_claims)} patents had no matching claim found:")
    for pub_num in not_found_claims:
        print(pub_num)

print("✅ Done! All indices and metadata saved.")