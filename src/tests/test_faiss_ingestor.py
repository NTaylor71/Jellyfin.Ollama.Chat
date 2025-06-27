import os
import json
import asyncio
import redis.asyncio as redis
from uuid import uuid4

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
INGEST_QUEUE = "chat:ingest"

sample_vectors = [
    {
        "id": str(uuid4()),
        "vector": [0.1 * i for i in range(4096)],  # Example vector
        "metadata": {"title": "The Matrix", "year": 1999}
    },
    {
        "id": str(uuid4()),
        "vector": [0.2 * i for i in range(4096)],
        "metadata": {"title": "Pulp Fiction", "year": 1994}
    }
]

async def submit_ingest_job():
    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())
    payload = {
        "job_id": job_id,
        "vectors": sample_vectors
    }

    print(f"📤 Submitting ingest job: {job_id}")
    await r.rpush(INGEST_QUEUE, json.dumps(payload))
    print(f"✅ Ingest job submitted to queue: {INGEST_QUEUE}")

if __name__ == "__main__":
    asyncio.run(submit_ingest_job())
