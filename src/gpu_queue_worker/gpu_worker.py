# /src/gpu_queue_worker/gpu_worker.py

import asyncio
import json
import httpx
from src.config import OLLAMA_BASE_URL, r
from src.config import GPU_QUEUE, GPU_RESULT_PREFIX, GPU_ERROR_PREFIX


VECTOR_DIM = 4096

import httpx
import asyncio

async def check_ollama_model_ready(max_retries=10, delay=5):
    """
    Check if Ollama model is ready. Will retry up to `max_retries` times with `delay` seconds between retries.
    """
    retries = 0
    while retries < max_retries:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                # Attempt to get the model's status
                response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if response.status_code == 200:
                    print("✅ Ollama model is ready.")
                    return True
                else:
                    print(f"❌ Ollama returned an unexpected status: {response.status_code}")
            except httpx.RequestError as e:
                print(f"⚠️ Error checking Ollama status: {e}")

        retries += 1
        print(f"⏳ Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
        await asyncio.sleep(delay)

    print("❌ Ollama model is not ready after maximum retries.")
    return False


async def gpu_worker_loop():
    """Asynchronous GPU queue worker loop."""
    print("🚀 GPU Queue Worker is ready. Listening for GPU jobs...")

    try:
        while True:
            # Wait for the Ollama model to be ready before processing any job
            model_ready = await check_ollama_model_ready(max_retries=5, delay=10)
            if not model_ready:
                print("❌ Exiting loop: Model not ready after retries.")
                continue  # You can break here if you want to exit the loop, or keep trying indefinitely

            try:
                job = await r.blpop(GPU_QUEUE, timeout=5)
                if job:
                    _, job_data = job
                    job_obj = json.loads(job_data.decode())
                    job_id = job_obj.get("job_id")
                    prompt = job_obj.get("prompt", "")

                    print(f"🔧 Processing GPU job: {job_id}")

                    try:
                        embedding = await get_ollama_embedding(prompt)

                        if embedding:
                            await r.set(f"{GPU_RESULT_PREFIX}{job_id}", json.dumps(embedding))
                            print(f"✅ GPU job {job_id} completed successfully.")
                        else:
                            await r.set(f"{GPU_ERROR_PREFIX}{job_id}", "Embedding failed or dimension mismatch.")
                            print(f"❌ GPU job {job_id} failed due to embedding issue.")

                    except Exception as e:
                        await r.set(f"{GPU_ERROR_PREFIX}{job_id}", str(e))
                        print(f"❌ GPU job {job_id} failed: {e}")

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"⚠️ GPU Worker loop error: {e}")
                await asyncio.sleep(2)

    finally:
        await r.close()
        print("🛑 Redis connection closed in GPU worker.")


async def get_ollama_embedding(text: str) -> list:
    """Query Ollama to generate embedding for the provided text."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json={"model": "mistral", "prompt": text})
            response.raise_for_status()
            embedding = response.json().get("embedding", [])
            if len(embedding) != VECTOR_DIM:
                print(f"⚠️ Embedding dimension mismatch. Expected {VECTOR_DIM}, got {len(embedding)}.")
                return []
            return embedding
        except Exception as e:
            print(f"❌ Ollama embedding request failed: {e}")
            return []
