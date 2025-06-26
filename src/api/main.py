import os
import json
import asyncio
import httpx
from fastapi import Body, FastAPI
from typing import List
from uuid import uuid4
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.rag.formatter import render_media_text_block
from src.config import r

app = FastAPI()

# ---- Startup Dependency Checks ----

async def wait_for_service(name: str, url: str, check_key: str = "status", success_value: str = "ok", max_attempts: int = 30, delay: float = 2.0):
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    json_response = response.json()
                    if check_key in json_response and json_response[check_key] == success_value:
                        print(f"[✅] {name} is ready.")
                        return
                    elif check_key == "status_code":
                        print(f"[✅] {name} responded with HTTP 200.")
                        return
        except Exception:
            pass

        print(f"[⏳] Waiting for {name}... attempt {attempt}/{max_attempts}")
        await asyncio.sleep(delay)

    raise RuntimeError(f"[❌] {name} did not become ready in time.")

@app.on_event("startup")
async def startup_event():
    await wait_for_service("Qdrant", "http://vectordb:6333/", check_key="status", success_value="ok")
    await wait_for_service("Ollama", "http://jellychat_ollama:11434/api/tags", check_key="status_code")

    for attempt in range(1, 31):
        try:
            if await r.ping():
                print(f"[✅] Redis is ready.")
                break
        except Exception:
            pass

        print(f"[⏳] Waiting for Redis... attempt {attempt}/30")
        await asyncio.sleep(2)
    else:
        raise RuntimeError(f"[❌] Redis did not become ready in time.")

    print("[🚀] All dependent services are ready. API fully booted.")

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

# ---- Collection Management ----

def ensure_collection(client: QdrantClient, collection_name: str, embedding_size: int, replace: bool):
    collections = [c.name for c in client.get_collections().collections]

    if replace and collection_name in collections:
        print(f"♻️ Deleting existing collection '{collection_name}'")
        client.delete_collection(collection_name)

    if collection_name not in collections:
        print(f"🆕 Creating collection '{collection_name}'")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
        )
    else:
        print(f"✅ Collection '{collection_name}' already exists")

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

    rendered = []
    for e in req.entries:
        doc_text = render_media_text_block(e.dict())
        rendered.append(doc_text)

    if not rendered:
        return {"status": "no_entries_provided"}

    # Prepare embeddings
    embedder = OllamaEmbeddings(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        model=os.getenv("OLLAMA_MODEL", "mistral")
    )

    qdrant_url = os.getenv("VECTORDB_URL", "http://vectordb:6333")
    collection_name = os.getenv("VECTORDB_COLLECTION", "jellyfin_rag")

    qdrant_client = QdrantClient(url=qdrant_url)

    ensure_collection(qdrant_client, collection_name, embedding_size=4096, replace=req.replace)

    vectorstore = QdrantVectorStore.from_texts(
        texts=rendered,
        embedding=embedder,
        url=qdrant_url,
        collection_name=collection_name
    )

    print(f"✅ Stored {len(rendered)} documents in collection '{collection_name}'")
    return {"status": "success", "count": len(rendered)}
