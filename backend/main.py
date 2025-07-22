# main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search import search_patents

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production!
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query_title: str
    query_claim: str
    query_description: str
    type: str  # "cosine" or "euclidean"
    claimWeight: float  # 0.1 - 1.0
    topK: int  # e.g., 5, 10, 15

@app.post("/search")
async def search(req: SearchRequest):
    results = search_patents(
        req.query_title,
        req.query_claim,
        req.query_description,
        similarity_mode=req.type,
        claim_weight=req.claimWeight,
        top_k=req.topK
    )
    return {"results": results}
