# /src/redis_queue_worker/main.py

import os
import asyncio
import json
import redis.asyncio as redis
from src.config import REDIS_QUEUE, RESULT_PREFIX, ERROR_PREFIX
from src.redis_queue_worker.worker import process_job

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

async def worker_loop():
    """Asynchronous Redis queue worker loop."""
    r = redis.from_url(REDIS_URL)
    print("🚀 Redis Queue Worker is ready. Listening for jobs...")

    try:
        while True:
            try:
                job = await r.blpop(REDIS_QUEUE, timeout=5)
                if job:
                    _, job_data = job
                    job_obj = json.loads(job_data.decode())
                    job_id = job_obj.get("job_id")

                    print(f"🔧 Processing job: {job_id}")
                    try:
                        result = await process_job(job_obj)
                        await r.set(f"{RESULT_PREFIX}{job_id}", result)
                        print(f"✅ Job {job_id} completed successfully.")
                    except Exception as e:
                        await r.set(f"{ERROR_PREFIX}{job_id}", str(e))
                        print(f"❌ Job {job_id} failed: {e}")

                await asyncio.sleep(0.1)

            except Exception as e:
                print(f"⚠️ Worker loop error: {e}")
                await asyncio.sleep(2)

    finally:
        await r.close()
        print("🛑 Redis connection closed.")

if __name__ == "__main__":
    asyncio.run(worker_loop())
