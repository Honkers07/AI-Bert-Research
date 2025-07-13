import json
import numpy as np
import torch
import time
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# === SETTINGS ===
MODEL_NAME = "AI-Growth-Lab/PatentSBERTa"
MAX_TOKENS = 512
BATCH_SIZE = 16  # Tune based on your GPU VRAM
INPUT_JSONL = r"C:\Users\erico\Downloads\g02b_keywords_filtered.jsonl"
OUTPUT_INDEX_EC = "patent_index_euclidean.faiss"
OUTPUT_INDEX_CS = "patent_index_cosine.faiss"
OUTPUT_METADATA = "patent_metadata.json"

print("📦 Loading model and tokenizer...")
model = SentenceTransformer(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

VECTOR_DIM = model.get_sentence_embedding_dimension()

index_ec = faiss.IndexFlatL2(VECTOR_DIM)
index_cs = faiss.IndexFlatIP(VECTOR_DIM)

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

# Count total lines to get total items
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    total_items = sum(1 for _ in f)

print(f"🚀 Processing {total_items} patents and building vector indices...")

batch_texts = []
batch_metadata = []
items_processed = 0

start_time = time.time()

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        claim_text = " ".join(data.get("claims", []))
        batch_texts.append(claim_text)
        batch_metadata.append({
            "publication_number": data.get("publication_number"),
            "title": data.get("title"),
            "publication_date": data.get("publication_date")
        })

        if len(batch_texts) >= BATCH_SIZE:
            embeddings = embed_texts(batch_texts)

            # Add Euclidean embeddings
            index_ec.add(embeddings)

            # Normalize embeddings for cosine and add
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_norm = embeddings / norms
            index_cs.add(embeddings_norm)

            metadata_store.extend(batch_metadata)

            items_processed += len(batch_texts)
            elapsed = time.time() - start_time
            speed = items_processed / elapsed if elapsed > 0 else 0
            percent = (items_processed / total_items) * 100
            eta_seconds = (total_items - items_processed) / speed if speed > 0 else 0

            # Format ETA as H:MM:SS
            eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

            # Print status line, overwrite previous line
            print(f"\rProcessed {items_processed}/{total_items} items ({percent:.2f}%) | Speed: {speed:.2f} items/s | ETA: {eta}", end="", flush=True)

            batch_texts, batch_metadata = [], []

# Process last batch if any
if batch_texts:
    embeddings = embed_texts(batch_texts)
    index_ec.add(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    index_cs.add(embeddings_norm)
    metadata_store.extend(batch_metadata)

    items_processed += len(batch_texts)
    elapsed = time.time() - start_time
    speed = items_processed / elapsed if elapsed > 0 else 0
    percent = (items_processed / total_items) * 100
    eta_seconds = (total_items - items_processed) / speed if speed > 0 else 0
    eta = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

    print(f"\rProcessed {items_processed}/{total_items} items ({percent:.2f}%) | Speed: {speed:.2f} items/s | ETA: {eta}", end="", flush=True)

print("\n💾 Saving FAISS indices and metadata...")
faiss.write_index(index_ec, OUTPUT_INDEX_EC)
faiss.write_index(index_cs, OUTPUT_INDEX_CS)

with open(OUTPUT_METADATA, "w", encoding="utf-8") as f:
    json.dump(metadata_store, f, ensure_ascii=False, indent=2)

print("✅ Done! Your indices and metadata are ready.")
