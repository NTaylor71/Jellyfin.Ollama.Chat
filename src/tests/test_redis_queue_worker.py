import os
import json
import asyncio
import redis.asyncio as redis
from uuid import uuid4

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE_NAME = "chat:queue"
RESULT_PREFIX = "chat:result:"
ERROR_PREFIX = "chat:error:"
TIMEOUT_SEC = 60

sample_query = "List famous sci-fi movies with virtual reality."

async def submit_and_listen():
    r = redis.from_url(REDIS_URL)
    job_id = str(uuid4())
    job_payload = {
        "job_id": job_id,
        "user_id": "test_worker",
        "query": sample_query
    }

    print(f"📤 Submitting job to Redis queue: {job_id}")
    await r.rpush(QUEUE_NAME, json.dumps(job_payload))

    print(f"⏳ Waiting for Redis Queue Worker to process job... (up to {TIMEOUT_SEC}s)")
    for i in range(TIMEOUT_SEC):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        error = await r.get(f"{ERROR_PREFIX}{job_id}")

        if result:
            print(f"\n✅ Result for job {job_id}:\n{result.decode('utf-8')}")
            return
        elif error:
            print(f"\n❌ Error for job {job_id}:\n{error.decode('utf-8')}")
            return

        await asyncio.sleep(1)

    print(f"\n⚠️ Timeout: No result for job {job_id} after {TIMEOUT_SEC} seconds.")

if __name__ == "__main__":
    asyncio.run(submit_and_listen())
