import json
import asyncio
import httpx
import redis.asyncio as redis
from uuid import uuid4
from src.config import REDIS_URL, REDIS_QUEUE, RESULT_PREFIX, ERROR_PREFIX, TIMEOUT_SEC, FAISS_URL
from src.data.sample_entries import get_sample_entries

sample_vectors = get_sample_entries()

async def ingest_sample():
    print("📥 Ingesting sample vectors into FAISS Service...")

    # Ensure each vector has a 'document' field
    for vector in sample_vectors:
        metadata = vector.get("metadata", {})
        vector["document"] = f"{metadata.get('title', 'Unknown')} ({metadata.get('year', 'Unknown')})"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(f"{FAISS_URL}/add", json={"vectors": sample_vectors})
        assert response.status_code == 200, f"Ingest failed: {response.text}"
        print(f"✅ Ingest response: {response.json()}")

async def query_sample():
    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())
    query_vector = [0.1 * i for i in range(4096)]

    payload = {
        "job_id": job_id,
        "user_id": "e2e_test",
        "query": "List famous sci-fi movies with virtual reality.",
        "query_vector": query_vector
    }

    print(f"📤 Submitting query job to Redis: {job_id}")
    await r.rpush(REDIS_QUEUE, json.dumps(payload))

    print(f"⏳ Waiting for result... (up to {TIMEOUT_SEC}s)")
    for i in range(TIMEOUT_SEC):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        error = await r.get(f"{ERROR_PREFIX}{job_id}")

        if result:
            print(f"\n✅ End-to-End Success:\n{result.decode('utf-8')}")
            return
        elif error:
            print(f"\n❌ End-to-End Error:\n{error.decode('utf-8')}")
            return

        await asyncio.sleep(1)

    print(f"\n⚠️ Timeout: No result for job {job_id} after {TIMEOUT_SEC} seconds.")

if __name__ == "__main__":
    asyncio.run(ingest_sample())
    asyncio.run(query_sample())
