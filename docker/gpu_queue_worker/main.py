import os
import asyncio
import json
import redis.asyncio as redis
import httpx
from src.config import REDIS_URL
from src.config import (
    GPU_QUEUE,
    GPU_RESULT_PREFIX,
    GPU_ERROR_PREFIX,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL
)

async def gpu_worker_loop():
    """Asynchronous GPU queue worker loop."""
    r = redis.from_url(REDIS_URL)
    print("🚀 GPU Queue Worker is ready. Listening for jobs...")

    try:
        while True:
            try:
                job = await r.blpop(GPU_QUEUE, timeout=5)
                if job:
                    _, job_data = job
                    job_obj = json.loads(job_data.decode())
                    job_id = job_obj.get("job_id")
                    prompt = job_obj.get("prompt", "")

                    print(f"🧠 Processing GPU embedding job: {job_id}")

                    embedding = await get_embedding_from_ollama(prompt)

                    if embedding:
                        await r.set(f"{GPU_RESULT_PREFIX}{job_id}", json.dumps(embedding), ex=60)
                        print(f"✅ GPU embedding job {job_id} completed.")
                    else:
                        await r.set(f"{GPU_ERROR_PREFIX}{job_id}", "Embedding generation failed.", ex=60)
                        print(f"❌ GPU embedding job {job_id} failed.")

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"⚠️ GPU Worker loop error: {e}")
                await asyncio.sleep(2)

    finally:
        await r.close()
        print("🛑 Redis connection closed.")

async def get_embedding_from_ollama(prompt: str) -> list:
    """Send embedding request to Ollama and return the embedding vector."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json={"model": OLLAMA_MODEL, "prompt": prompt})
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            return embedding
        except Exception as e:
            print(f"❌ Ollama embedding request failed: {e}")
            return []

if __name__ == "__main__":
    asyncio.run(gpu_worker_loop())
