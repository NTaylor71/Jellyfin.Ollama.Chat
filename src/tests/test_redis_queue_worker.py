import os
import json
import asyncio
import redis.asyncio as redis
from uuid import uuid4
from src.data.sample_entries import get_sample_vectors
from src.config import FAISS_VECTOR_DIM, REDIS_URL, REDIS_QUEUE, RESULT_PREFIX, ERROR_PREFIX, TIMEOUT_SEC  # Correct import

sample_query = "List famous sci-fi movies with virtual reality."

async def submit_ingestion_job():
    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())

    # Get real sample vectors from the data
    vectors = get_sample_vectors()

    # Prepare job payload for ingestion
    job_payload = {
        "job_id": job_id,
        "vectors": vectors
    }

    print(f"📦 Submitting ingestion job to Redis queue: {job_id}")
    await r.rpush(REDIS_QUEUE, json.dumps(job_payload))  # Use REDIS_QUEUE from config
    print(f"⏳ Waiting for ingestion to process the job... (up to 5s)")

    await asyncio.sleep(5)

    print(f"✅ Ingestion job {job_id} submitted and presumed processed.")

async def submit_and_listen():
    await submit_ingestion_job()

    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())
    job_payload = {
        "job_id": job_id,
        "user_id": "test_worker",
        "query": sample_query
    }

    print(f"📤 Submitting search job to Redis queue: {job_id}")
    await r.rpush(REDIS_QUEUE, json.dumps(job_payload))  # Use REDIS_QUEUE from config

    print(f"⏳ Waiting for Redis Queue Worker to process job... (up to {TIMEOUT_SEC}s)")
    for i in range(TIMEOUT_SEC):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        error = await r.get(f"{ERROR_PREFIX}{job_id}")

        if result:
            print(f"\n✅ Result for job {job_id}:\n{result.decode('utf-8')}")
            return
        if error:
            print(f"\n❌ Error for job {job_id}:\n{error.decode('utf-8')}")
            return

        await asyncio.sleep(1)

    print(f"❌ Timeout waiting for result for job {job_id}.")

if __name__ == "__main__":
    asyncio.run(submit_and_listen())
