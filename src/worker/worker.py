import os
import json
import asyncio
import redis.asyncio as redis
import httpx
from src.config import r

QUEUE_NAME = os.getenv("REDIS_QUEUE", "chat:queue")
RESULT_PREFIX = "chat:result:"
ERROR_PREFIX = "chat:error:"
API_URL = os.getenv("API_URL", "http://jellychat_ollama:11434/api/chat")


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


async def handle_job(job: dict) -> None:
    job_id = job.get("job_id")
    query = job.get("query")

    if not job_id or not query:
        print(f"⚠️ Invalid job structure: {job}")
        return

    try:
        print(f"💬 Processing job: {job_id} | Query: {query}")
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(API_URL, json={"model": "mistral", "prompt": query})
            response.raise_for_status()
            answer = response.json().get("response", "").strip()

        await r.set(f"{RESULT_PREFIX}{job_id}", answer)
        print(f"✅ Job complete: {job_id}")

    except Exception as e:
        await r.set(f"{ERROR_PREFIX}{job_id}", str(e))
        print(f"❌ Job failed: {job_id} | Error: {str(e)}")


async def worker_loop() -> None:
    await wait_for_service("Ollama", os.getenv("OLLAMA_BASE_URL", "http://ollama:11434") + "/api/tags", check_key="status_code")

    print(f"🚀 Worker listening on queue: {QUEUE_NAME}")
    while True:
        try:
            job_json = await r.blpop(QUEUE_NAME, timeout=0)
            if job_json:
                _, job_data = job_json
                job = json.loads(job_data)
                await handle_job(job)
        except Exception as e:
            print(f"❌ Worker loop error: {str(e)}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(worker_loop())
