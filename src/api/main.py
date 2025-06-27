import asyncio
import json
from fastapi import FastAPI
from uuid import uuid4

from src.config import r
from src.api.healthcheck import router as health_router

app = FastAPI()

# ---- Health Endpoint ----
app.include_router(health_router)

# ---- Startup Dependency Checks ----

async def wait_for_service(name: str, url: str, max_attempts: int = 30, delay: float = 2.0):
    import httpx

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code == 200:
                    print(f"[✅] {name} is ready.")
                    return
        except Exception:
            pass

        print(f"[⏳] Waiting for {name}... attempt {attempt}/{max_attempts}")
        await asyncio.sleep(delay)

    raise RuntimeError(f"[❌] {name} did not become ready in time.")

@app.on_event("startup")
async def startup_event():
    await wait_for_service("FAISS Service", "http://faiss_service:6333/health")
    await wait_for_service("Ollama", "http://ollama:11434/api/tags")

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

from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str
    user_id: str = "anonymous"

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
