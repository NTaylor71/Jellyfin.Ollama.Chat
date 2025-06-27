from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from src.faiss_service.faiss_index import FaissIndex

app = FastAPI()
faiss_index = FaissIndex()

# ---- Data Models ----

class VectorItem(BaseModel):
    id: str = None
    vector: List[float]
    metadata: dict = {}

class AddRequest(BaseModel):
    vectors: List[VectorItem]

class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 5

class DeleteRequest(BaseModel):
    ids: List[str]

# ---- Health Endpoint ----

@app.get("/health")
async def health():
    return {"status": "ok"}

# ---- Add Vectors ----

@app.post("/add")
async def add_vectors(req: AddRequest):
    if not req.vectors:
        raise HTTPException(status_code=400, detail="No vectors provided.")

    ids = faiss_index.add_vectors([v.dict() for v in req.vectors])
    faiss_index.save()
    return {"status": "success", "added": len(ids), "ids": ids}

# ---- Search Vectors ----

@app.post("/search")
async def search_vectors(req: SearchRequest):
    results = faiss_index.search(req.query_vector, req.top_k)
    return {"results": results}

# ---- Delete Vectors ----
# (Optional for future implementation)

@app.post("/delete")
async def delete_vectors(req: DeleteRequest):
    # For now, FAISS FlatL2 does not support delete natively.
    # Future: Rebuild index without the deleted vectors if needed.
    return {"status": "delete_not_supported"}
