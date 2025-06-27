import os
import json
import asyncio
import redis.asyncio as redis
from uuid import uuid4
from src.data.sample_entries import get_sample_vectors  # ✅ Unified sample data

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INGEST_QUEUE = "chat:ingest"
TIMEOUT_SEC = 30

async def submit_ingestion_job():
    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())

    vectors = get_sample_vectors()  # ✅ Always use unified source

    job_payload = {
        "job_id": job_id,
        "vectors": vectors
    }

    print(f"📦 Submitting ingestion job to Redis queue: {job_id}")
    await r.rpush(INGEST_QUEUE, json.dumps(job_payload))
    print(f"⏳ Waiting for ingestion to process the job... (up to {TIMEOUT_SEC}s)")

    await asyncio.sleep(5)

    print(f"✅ Ingestion job {job_id} submitted and presumed processed.")

if __name__ == "__main__":
    asyncio.run(submit_ingestion_job())
