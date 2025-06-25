import os
import json
from fastapi import Body, FastAPI
from typing import List
from uuid import uuid4
from pydantic import BaseModel
import redis.asyncio as redis
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src.rag.formatter import render_media_text_block

app = FastAPI()

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
r = redis.from_url(REDIS_URL)

# ---- Data Models ----

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

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

# ---- API Endpoints ----

@app.post("/chat")
async def submit_chat(req: ChatRequest):
    job_id = str(uuid4())
    await r.rpush("chat:queue", json.dumps({
        "job_id": job_id,
        "user_id": req.user_id,
        "query": req.query
    }))
    return {"job_id": job_id}

@app.get("/chat/result/{job_id}")
async def get_result(job_id: str):
    key = f"chat:result:{job_id}"
    err = f"chat:error:{job_id}"
    if result := await r.get(key):
        return {"job_id": job_id, "result": result.decode()}
    if error := await r.get(err):
        return {"job_id": job_id, "error": error.decode()}
    return {"job_id": job_id, "status": "pending"}


@app.post("/api/ingest")
async def ingest_metadata(req: IngestRequest = Body(...)):
    print(f"📥 Ingesting {len(req.entries)} entries...")

    # Generate texts via Jinja2
    rendered = []
    for e in req.entries:
        doc_text = render_media_text_block(e.dict())
        rendered.append({"id": str(uuid4()), "text": doc_text})

    texts = [r["text"] for r in rendered]
    ids = [r["id"] for r in rendered]

    # Create Ollama embeddings
    embedder = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("OLLAMA_MODEL", "mistral")
    )

    # Qdrant client setup
    qdrant_url = os.getenv("VECTORDB_URL", "http://vectordb:6333")
    collection_name = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")

    qdrant_client = QdrantClient(url=qdrant_url)

    if req.replace:
        print("♻️ Replacing collection...")
        qdrant_client.delete_collection(collection_name)

    # Use from_texts to ingest
    store = QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embedder,
        url=qdrant_url,
        collection_name=collection_name
    )

    print(f"✅ Stored {len(texts)} documents in collection '{collection_name}'")
    return {"status": "success", "count": len(rendered)}
