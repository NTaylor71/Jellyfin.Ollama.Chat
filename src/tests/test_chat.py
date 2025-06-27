import os
import json
import asyncio
import redis.asyncio as redis
from uuid import uuid4
from src.config import REDIS_URL, REDIS_QUEUE, RESULT_PREFIX, ERROR_PREFIX, TIMEOUT_SEC

query = "What are some 90s dystopian sci-fi movies?"
job_id = str(uuid4())

payload = {
    "job_id": job_id,
    "user_id": "test",
    "query": query
}

async def test_roundtrip():
    r = redis.from_url(REDIS_URL)
    print(f"📤 Submitting job: {job_id}")
    await r.rpush(REDIS_QUEUE, json.dumps(payload))

    print(f"⏳ Waiting for result... (up to {TIMEOUT_SEC}s)")
    for i in range(TIMEOUT_SEC):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        error = await r.get(f"{ERROR_PREFIX}{job_id}")

        if result:
            print(f"\n✅ Success:\n{result.decode('utf-8')}")
            return
        elif error:
            print(f"\n❌ Error:\n{error.decode('utf-8')}")
            return

        await asyncio.sleep(1)

    print(f"\n⚠️ Timeout: no result after {TIMEOUT_SEC} seconds.")

if __name__ == "__main__":
    asyncio.run(test_roundtrip())
