import json
import faiss
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

# === CONFIG ===
CLAIM_INDEX_PATH = "index_claim_euclidean.faiss"
DESC_INDEX_PATH = "index_desc_euclidean.faiss"
METADATA_PATH = "patent_metadata.json"
N_COMPONENTS = 2  # PCA to 2D

# === LOAD FAISS INDICES ===
print("📦 Loading FAISS indices...")
index_claim = faiss.read_index(CLAIM_INDEX_PATH)
index_desc = faiss.read_index(DESC_INDEX_PATH)

embeddings_claim = index_claim.reconstruct_n(0, index_claim.ntotal)
embeddings_desc = index_desc.reconstruct_n(0, index_desc.ntotal)

# === LOAD METADATA ===
print("📄 Loading metadata...")
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Split metadata by section tag
metadata_desc = [m for m in metadata if m.get("section") == "description"]
metadata_claim = [m for m in metadata if m.get("section") == "claim"]

assert len(metadata_desc) == len(embeddings_desc), "❌ Description metadata length mismatch"
assert len(metadata_claim) == len(embeddings_claim), "❌ Claim metadata length mismatch"

# === RUN PCA ===
print("📊 Running PCA on description embeddings...")
pca_desc = PCA(n_components=N_COMPONENTS)
reduced_desc = pca_desc.fit_transform(embeddings_desc)

print("📊 Running PCA on claim embeddings...")
pca_claim = PCA(n_components=N_COMPONENTS)
reduced_claim = pca_claim.fit_transform(embeddings_claim)

# === PREPARE DATA ===
def build_plot_data(reduced, metadata):
    return {
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "Title": [m.get("title", "No Title") for m in metadata],
        "Publication Number": [m.get("publication_number", "N/A") for m in metadata],
    }

plot_data_desc = build_plot_data(reduced_desc, metadata_desc)
plot_data_claim = build_plot_data(reduced_claim, metadata_claim)

# === PLOT DESCRIPTION ===
print("📈 Plotting description embeddings...")
fig_desc = px.scatter(
    plot_data_desc,
    x="x",
    y="y",
    hover_data=["Title", "Publication Number"],
    title="Patent Description Embeddings (PCA)",
    labels={"x": "PC1", "y": "PC2"}
)
fig_desc.show()

# === PLOT CLAIMS ===
print("📈 Plotting claim embeddings...")
fig_claim = px.scatter(
    plot_data_claim,
    x="x",
    y="y",
    hover_data=["Title", "Publication Number"],
    title="Patent Claim Embeddings (PCA)",
    labels={"x": "PC1", "y": "PC2"}
)
fig_claim.show()
