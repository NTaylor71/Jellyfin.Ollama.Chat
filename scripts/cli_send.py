import os
import json
import redis.asyncio as redis
import asyncio
from uuid import uuid4

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
QUEUE = "chat:queue"
RESULT_PREFIX = "chat:result:"
ERROR_PREFIX = "chat:error:"
TIMEOUT = 60  # seconds

r = redis.from_url(REDIS_URL)

async def send_query(query: str):
    job_id = str(uuid4())
    job = {"job_id": job_id, "user_id": "cli", "query": query}
    await r.rpush(QUEUE, json.dumps(job))
    print(f"⏳ Submitted. Waiting for reply to job_id: {job_id}")

    for _ in range(TIMEOUT):
        result = await r.get(f"{RESULT_PREFIX}{job_id}")
        error = await r.get(f"{ERROR_PREFIX}{job_id}")

        if result:
            print(f"\n✅ Result:\n{result.decode('utf-8')}\n")
            return
        elif error:
            print(f"\n❌ Error:\n{error.decode('utf-8')}\n")
            return
        await asyncio.sleep(1)

    print(f"\n⚠️ Timeout: No response after {TIMEOUT}s.\n")

async def main():
    print("💬 FAISS RAG CLI (Ctrl+C to quit)")
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            await send_query(query)
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    asyncio.run(main())
