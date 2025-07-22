```markdown
# Patent Similarity Search for Quantum Photonic Circuits

This application helps researchers and inventors explore related patents in the field of **Quantum Photonic Circuits** using **vector similarity search** on BERT-based embeddings. Users input a patent **title**, **claim**, and **description**, and the app returns the most relevant patents ranked by similarity.

Built using:
- **Sentence-BERT** (`PatentSBERTa`)
- **FAISS** for vector search
- **FastAPI** backend
- **React + MUI** frontend

---

## Features

- Combine **claim** and **description** vectors with adjustable **weights**
- Choose between **cosine** or **euclidean** similarity
- Dynamically select the **number of top results** to return
- Clean, responsive UI with interactive sliders and controls

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI-Bert-Research.git
cd AI-Bert-Research
```

---

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Make sure the following files are in the `backend/data/` folder:

- `index_claim_cosine.faiss`
- `index_claim_euclidean.faiss`
- `index_desc_cosine.faiss`
- `index_desc_euclidean.faiss`
- `patent_metadata.json`

---

### 3. Frontend Setup

In a new terminal tab/window:

```bash
cd frontend
npm install
npm run dev
```

This will start the React frontend on `localhost:3000`.

---

## Usage

1. Enter a **title**, **claim**, and/or **description** in the input fields.
2. Use the sliders to adjust:
   - Claim vs. Description Weight
   - Similarity Mode: Cosine or Euclidean
   - Number of Similar Patents to Return
3. Click **Search**.
4. View dynamically ranked patent results based on your settings.

---

## Requirements

- **Python** ≥ 3.9
- **Node.js** ≥ 16
- Required Python packages (see `backend/requirements.txt`):
  - `fastapi`
  - `uvicorn`
  - `faiss-cpu`
  - `sentence-transformers`
  - `numpy`
- Required JS packages (installed via `npm install` in frontend)

---

## Notes

- This app is designed to run **locally** and does not require cloud APIs.
- You must generate the FAISS indices and metadata beforehand.
- FAISS indices assume **cosine** similarity by default unless configured otherwise.


