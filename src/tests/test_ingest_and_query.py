import json
import asyncio
import httpx
from uuid import uuid4
from src.config import API_URL, QUEUE, RESULT_PREFIX, r  # Import from config.py
from src.data.sample_entries import get_sample_entries  # Import centralized data source

TIMEOUT = 30

# Get sample entries from centralized data source
sample_entries = get_sample_entries()

async def ingest_sample():
    print("📥 Ingesting sample...")

    # Ensure each entry is formatted for ingestion
    for entry in sample_entries:
        entry["document"] = f"{entry['title']} ({entry['year']})"  # Assuming this is the format required

    # TODO : timeout=None : remove once Redis queue is a thing
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{API_URL}/api/ingest", json={
            "replace": True,
            "entries": sample_entries
        })
        assert resp.status_code == 200, resp.text
        print("✅ Ingest complete")

async def query_sample():
    job_id = str(uuid4())
    payload = {"job_id": job_id, "user_id": "test", "query": "movies about simulated reality"}

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
