import asyncio
import json
import httpx
from src.config import (
    OLLAMA_EMBED_BASE_URL,
    OLLAMA_CHAT_BASE_URL,
    r,
    REDIS_QUEUE,
    RESULT_PREFIX,
    ERROR_PREFIX,
    OLLAMA_EMBED_MODEL,
    OLLAMA_CHAT_MODEL
)
from src.ollama_service.embedding_client import get_ollama_embedding
from src.ollama_service.chat_client import get_ollama_chat_response

async def check_ollama_model_ready(base_url, max_retries=10, delay=5):
    retries = 0
    while retries < max_retries:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    print(f"✅ Ollama model at {base_url} is ready.")
                    return True
                else:
                    print(f"❌ Ollama at {base_url} returned an unexpected status: {response.status_code}")
            except httpx.RequestError as e:
                print(f"⚠️ Error checking Ollama status at {base_url}: {e}")

        retries += 1
        print(f"⏳ Retrying {base_url} in {delay} seconds... (Attempt {retries}/{max_retries})")
        await asyncio.sleep(delay)

    print(f"❌ Ollama model at {base_url} is not ready after maximum retries.")
    return False


async def gpu_worker_loop():
    """Asynchronous Redis queue worker loop for multi-purpose tasks."""
    print("🚀 Redis Queue Worker is ready. Listening for jobs...")

    try:
        while True:
            try:
                job = await r.blpop(REDIS_QUEUE, timeout=5)
                if job:
                    _, job_data = job
                    job_obj = json.loads(job_data.decode())
                    job_id = job_obj.get("job_id")
                    task_type = job_obj.get("task_type", "embedding")
                    prompt = job_obj.get("prompt", "")

                    print(f"🔧 Processing job: {job_id} | Type: {task_type}")

                    if task_type == "embedding":
                        model_ready = await check_ollama_model_ready(OLLAMA_EMBED_BASE_URL, max_retries=5, delay=10)
                        if not model_ready:
                            await r.set(f"{ERROR_PREFIX}{job_id}", "Embedding model not ready.")
                            print(f"❌ Embedding model not ready for job {job_id}.")
                            continue

                        embedding = await get_ollama_embedding(prompt, model=OLLAMA_EMBED_MODEL)
                        if embedding:
                            await r.set(f"{RESULT_PREFIX}{job_id}", json.dumps(embedding))
                            print(f"✅ Embedding job {job_id} completed successfully.")
                        else:
                            await r.set(f"{ERROR_PREFIX}{job_id}", "Embedding failed or returned empty.")
                            print(f"❌ Embedding job {job_id} failed.")

                    elif task_type == "chat":
                        model_ready = await check_ollama_model_ready(OLLAMA_CHAT_BASE_URL, max_retries=5, delay=10)
                        if not model_ready:
                            await r.set(f"{ERROR_PREFIX}{job_id}", "Chat model not ready.")
                            print(f"❌ Chat model not ready for job {job_id}.")
                            continue

                        messages = job_obj.get("messages", [{"role": "user", "content": prompt}])
                        response = await get_ollama_chat_response(messages, model=OLLAMA_CHAT_MODEL)
                        if response:
                            await r.set(f"{RESULT_PREFIX}{job_id}", json.dumps({"response": response}))
                            print(f"✅ Chat job {job_id} completed successfully.")
                        else:
                            await r.set(f"{ERROR_PREFIX}{job_id}", "Chat response was empty.")
                            print(f"❌ Chat job {job_id} failed: Empty response.")

                    else:
                        await r.set(f"{ERROR_PREFIX}{job_id}", f"Unknown task type: {task_type}")
                        print(f"❌ Unknown task type for job {job_id}: {task_type}")

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Worker loop error: {e}")
                await asyncio.sleep(2)

    finally:
        await r.close()
        print("🛑 Redis connection closed in worker.")
