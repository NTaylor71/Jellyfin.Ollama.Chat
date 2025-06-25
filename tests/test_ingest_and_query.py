import os
import json
import asyncio
import redis.asyncio as redis
import httpx
from uuid import uuid4

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
API_URL = os.getenv("API_URL", "http://localhost:8000")
QUEUE = "chat:queue"
RESULT_PREFIX = "chat:result:"
TIMEOUT = 30

sample_entry = {
    "title": "The Matrix",
    "year": 1999,
    "genres": ["Sci-Fi", "Action"],
    "overview": "A hacker discovers reality is a simulation.",
    "actors": ["Keanu Reeves"],
    "tagline": "Welcome to the Real World",
    "certificate": "R",
    "media_type": "Movie",
    "language": "English"
}

async def ingest_sample():
    print("📥 Ingesting sample...")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{API_URL}/api/ingest", json={
            "replace": True,
            "entries": [sample_entry]
        })
        assert resp.status_code == 200, resp.text
        print("✅ Ingest complete")

async def query_sample():
    job_id = str(uuid4())
    payload = {"job_id": job_id, "user_id": "test", "query": "movies about simulated reality"}

    r = redis.from_url(REDIS_URL)
    await r.rpush(QUEUE, json.dumps(payload))

    print(f"🧠 Sent query job: {job_id}")

    for _ in range(TIMEOUT):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        if result:
            print(f"\n✅ Got result:\n{result.decode('utf-8')}")
            return
        await asyncio.sleep(1)

    print("\n⚠️ No response after timeout.")

if __name__ == "__main__":
    asyncio.run(ingest_sample())
    asyncio.run(query_sample())
