import json
import faiss
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

# === CONFIG ===
INDEX_PATH = "patent_index_euclidean.faiss"
METADATA_PATH = "patent_metadata.json"
N_COMPONENTS = 2  # for 2D visualization

# === LOAD FAISS INDEX ===
print("📦 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)

# Extract raw vectors
embeddings = index.reconstruct_n(0, index.ntotal)

# === LOAD METADATA ===
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# === RUN PCA ===
print("📊 Running PCA...")
pca = PCA(n_components=N_COMPONENTS)
reduced_embeddings = pca.fit_transform(embeddings)

# === PREPARE DATA FOR PLOTTING ===
titles = [m.get("title", "No Title") for m in metadata]
pub_dates = [m.get("publication_date", "Unknown") for m in metadata]
pub_nums = [m.get("publication_number", "N/A") for m in metadata]

plot_data = {
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
    "Title": titles,
    "Publication Date": pub_dates,
    "Publication Number": pub_nums,
}

# === PLOT ===
print("📈 Creating interactive plot...")
fig = px.scatter(
    plot_data,
    x="x",
    y="y",
    hover_data=["Title", "Publication Date", "Publication Number"],
    title="Patent Embeddings (PCA Visualization)",
    labels={"x": "PC1", "y": "PC2"}
)
fig.show()
