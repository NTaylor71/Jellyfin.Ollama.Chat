"""
Example usage

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "movies about virtual reality"}'

"""
import os
import json
from fastapi import Body, FastAPI
from typing import List
from src.rag.formatter import render_media_text_block
from langchain.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from uuid import uuid4
from pydantic import BaseModel
import redis.asyncio as redis


app = FastAPI()

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

# ---- Models ----

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"  # optional session/user tracking

# ---- Endpoints ----

@app.post("/chat")
async def submit_chat(req: ChatRequest):
    job_id = str(uuid4())
    job = {
        "job_id": job_id,
        "user_id": req.user_id,
        "query": req.query,
    }
    await r.rpush("chat:queue", json.dumps(job))
    return {"job_id": job_id}


@app.get("/chat/result/{job_id}")
async def get_result(job_id: str):
    key = f"chat:result:{job_id}"
    err = f"chat:error:{job_id}"

    result = await r.get(key)
    if result:
        return {"job_id": job_id, "result": result.decode("utf-8")}

    error = await r.get(err)
    if error:
        return {"job_id": job_id, "error": error.decode("utf-8")}

    return {"job_id": job_id, "status": "pending"}


class MediaEntry(BaseModel):
    title: str
    year: int
    genres: List[str] = []
    tagline: str = ""
    overview: str = ""
    actors: List[str] = []
    certificate: str = ""
    media_type: str = "Movie"
    language: str = "English"


class IngestRequest(BaseModel):
    entries: List[MediaEntry]
    replace: bool = False


@app.post("/api/ingest")
async def ingest_metadata(req: IngestRequest = Body(...)):
    print(f"📥 Ingesting {len(req.entries)} entries...")

    # Render media blocks
    rendered = []
    for e in req.entries:
        doc_text = render_media_text_block(e.dict())
        rendered.append({"id": str(uuid4()), "text": doc_text})

    # Embed
    embedder = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("OLLAMA_MODEL", "mistral")
    )
    texts = [d["text"] for d in rendered]
    ids = [d["id"] for d in rendered]
    vectors = embedder.embed_documents(texts)

    # Store in Qdrant
    qdrant = Qdrant(
        url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
        collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag"),
        prefer_grpc=False
    )

    if req.replace:
        print("♻️ Replacing collection content...")
        qdrant.client.delete_collection(req.entries[0].collection_name, wait=True)

    qdrant.add_texts(texts=texts, metadatas=None, ids=ids, embeddings=vectors)

    return {"status": "success", "count": len(rendered)}
